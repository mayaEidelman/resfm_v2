import sys
# import time
import torch
import math
# import time
from utils import general_utils
import loss_functions
import evaluation
import copy

from utils.general_utils import save_metrics_excel
from utils import path_utils, dataset_utils, plot_utils
from utils.metrics_utils import OutliersMetrics, CalcMeanBatchMetrics, nanMetrics
from time import time
import pandas as pd
from utils.Phases import Phases
from utils.path_utils import path_to_exp
# from torch.utils.tensorboard import SummaryWriter
import wandb
import os
from torch.distributed import all_gather_object
from torch.nn import functional as F

from lightning.fabric import Fabric


os.environ["WANDB__SERVICE_WAIT"] = "300"
OPTIMIZER_TYPES = {
    'Adam': torch.optim.Adam,
}

OUTPUT_MODES_TYPES = {
    1: "ESFM",
    2: "Outliers Removal",
    3: "RESFM",
    4: "3mlps_ablation",

}

# fabric = Fabric(accelerator="cuda", devices="auto", strategy="ddp")
# # fabric = Fabric(accelerator="cuda", devices=1)
# fabric.launch()


def epoch_evaluation(data_loader, model, conf, epoch, phase, save_predictions=False, bundle_adjustment=True):
    metrics_list = []
    errors = None
    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            # A batch of scenes
            for curr_data in batch_data:
                # One scene
                # Get predictions
                begin_time = time()
                pred_cam, pred_outliers = model(curr_data)

                # Eval results
                metrics = {}
                if pred_cam is not None:
                    outputs = evaluation.prepare_predictions(curr_data, pred_cam, conf, bundle_adjustment, phase, curr_epoch=epoch)
                    errors, errors_per_cam, outputs = evaluation.compute_errors(outputs, conf, bundle_adjustment)
                    metrics.update(errors)
                    if pred_outliers is not None:
                        outputs["outliers_pred"] = pred_outliers.cpu().numpy()
                else:
                    outputs = {}

                if pred_outliers is not None:
                    metrics.update(OutliersMetrics(pred_outliers, curr_data))
                    outliersOutputs = evaluation.prepare_outliers_predictions(curr_data, pred_outliers, conf)

                # Get scene statistics on final evaluation
                if epoch is None or phase is Phases.FINE_TUNE:
                    stats = dataset_utils.get_data_statistics(curr_data, outputs)
                    metrics.update(stats)


                metrics['Scene'] = curr_data.scan_name
                metrics_list.append(metrics)

                if save_predictions:
                    if pred_outliers is not None:
                        dataset_utils.save_outliers(outliersOutputs, conf, curr_epoch=epoch, phase=phase)
                    if errors is not None:
                        errors.update(errors_per_cam)
                        plot_utils.plot_cameras_before_and_after_ba(outputs, errors, conf, phase, scan=curr_data.scan_name, epoch=epoch, bundle_adjustment=bundle_adjustment)

    df_metrics = evaluation.organize_errors(metrics_list)
    model.train()

    return df_metrics, metrics_list

def epoch_train(conf, train_data, model, loss_func, optimizer, scheduler, epoch, phase):
    model.train()
    train_losses = []
    train_metrics = []

    for train_batch in train_data:
        batch_loss = torch.tensor([0.0], device=train_data.device)
        optimizer.zero_grad()

        for i, curr_data in enumerate(train_batch):
            if not dataset_utils.is_valid_sample(curr_data, phase=phase, min_pts_per_cam=0):
                print(f"{fabric.global_rank}: {epoch} {curr_data.scan_name} has a camera with not enough points")
                continue

            pred_cam, pred_outliers = model(curr_data)

            if pred_outliers is not None and pred_cam is None:
                pred_weights_M = torch.zeros(curr_data.x.shape[0], curr_data.x.shape[1], device=curr_data.x.device)
                pred_weights_M[curr_data.x.indices[0, :], curr_data.x.indices[1, :]] = pred_outliers.squeeze(dim=-1)
                loss = loss_func(pred_outliers, pred_weights_M, curr_data, epoch)

            elif pred_cam is not None and pred_outliers is None:
                loss = loss_func(pred_cam, curr_data, epoch)

            else:
                pred_weights_M = torch.zeros(curr_data.x.shape[0], curr_data.x.shape[1], device=curr_data.x.device)
                pred_weights_M[curr_data.x.indices[0, :], curr_data.x.indices[1, :]] = pred_outliers.squeeze(dim=-1)
                loss = loss_func(pred_cam, pred_outliers, pred_weights_M, curr_data, epoch)

            metrics = OutliersMetrics(pred_outliers, curr_data) if pred_outliers is not None else {}

            batch_loss += loss
            train_losses.append(loss.item())
            train_metrics.append(metrics)

        if batch_loss.item() > 0:
            fabric.backward(batch_loss)
            optimizer.step()

        else:
            print('Hi')
            print("Negative or zero batch loss:", batch_loss.item())
            metrics = nanMetrics()
            train_metrics = [metrics]
            train_losses = [torch.tensor([0.0], device=train_data.device).item()]

    scheduler.step()

    mean_loss = torch.tensor(train_losses).mean()
    train_metrics = CalcMeanBatchMetrics(train_metrics, None)

    return mean_loss, train_losses, [train_metrics]

def train(conf, train_data, model, phase, validation_data=None, test_data=None, fabri=None):
    global fabric
    fabric = fabri

    num_of_epochs = conf.get_int('train.num_of_epochs')
    eval_intervals = conf.get_int('train.eval_intervals', default=500)
    validation_metric = conf.get_list('train.validation_metric') if phase is not Phases.FINE_TUNE else conf.get_list('train.validation_metric_fine_tuning')

    # === Loss Function ===
    if phase in [Phases.FINE_TUNE, Phases.SHORT_OPTIMIZATION]:
        conf["resuming_epoch"] = -1
        try:
            loss_name = conf.get_string('loss.func_tuning')
        except KeyError:
            print("[Warning] Missing 'loss.func_tuning' in config — falling back to 'loss.func'")
            loss_name = conf.get_string('loss.func')
    else:
        loss_name = conf.get_string('loss.func')
    # Get loss function from the loss_functions module
    loss_func = getattr(loss_functions, loss_name)(conf)


    # === Optimizer & Scheduler ===
    lr = conf.get_float('train.lr')
    scheduler_milestone = conf.get_list('train.scheduler_milestone')
    gamma = conf.get_float('train.gamma', default=0.1)
    optim_type = conf.get_string('train.optim_type', default='Adam')
    optimizer = OPTIMIZER_TYPES[optim_type](model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestone, gamma=gamma)

    print(f'Training with optimizer of type {type(optimizer)}')


    # === Resume Training (Optional) ===
    try:
        if conf['resume']:
            if phase is Phases.OPTIMIZATION:
                path, epoch = path_utils.path_to_model_resume_optimizing(conf, scan=conf.dataset.scan)
            elif phase is Phases.TRAINING:
                path, epoch = path_utils.path_to_model_resume_learning(conf)
            checkpoint = torch.load(path)
            conf["resuming_epoch"] = checkpoint['epoch']
            if checkpoint['epoch'] >= num_of_epochs:
                sys.exit()
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("The model is resuming from checkpoint:", path)
    except:
        pass


    # === Setup Fabric and Dataloaders ===
    model, optimizer = fabric.setup(model, optimizer)
    train_data = fabric.setup_dataloaders(train_data, use_distributed_sampler=True)
    if test_data is not None:
        validation_data = fabric.setup_dataloaders(validation_data)
        test_data = fabric.setup_dataloaders(test_data)

    # === WandB Logging ===
    if fabric.global_rank == 0:
        mode = 'online' if conf['wandb'] else 'disabled'
        name = conf.get_string('exp_name')
        if phase is Phases.OPTIMIZATION:
            name += '__' + conf['dataset.scan']
        logger = wandb.init(
            mode=mode,
            name=name,
            project="RESfM",
            dir=path_utils.path_to_wandb_logs(conf, phase),
        )
        print(f'Starting a logger at {path_utils.path_to_wandb_logs(conf, phase, scan=None)}')

    # === Tracking Best Metrics ===
    best_validation_metric = math.inf if phase is Phases.FINE_TUNE else -math.inf
    best_epoch = 0
    converge_time = -1
    begin_time = time()

    # === Training Loop ===
    for epoch in range(conf["resuming_epoch"] + 1, num_of_epochs):
        ba_during_training = not conf.get_bool('ba.only_last_eval') and conf.get_bool('ba.run_ba', default=True)

        mean_train_loss, train_losses, train_metrics = epoch_train(conf, train_data, model, loss_func, optimizer, scheduler, epoch, phase=phase)
        mean_train_loss = fabric.all_reduce(mean_train_loss, reduce_op="mean")
        train_metrics = fabric.all_gather(train_metrics)

        if fabric.global_rank == 0:
            wandb.log({"training loss": mean_train_loss}, step=epoch)
            wandb.log({"LR": scheduler.get_last_lr()[-1]}, step=epoch)
            if conf.get_int('train.output_mode', default=3) != 1:
                train_metrics_means = CalcMeanBatchMetrics(train_metrics, phase)
                wandb.log(train_metrics_means, step=epoch)
            if epoch % 100 == 0:
                print(f'{fabric.global_rank}:{epoch} Train Loss: {mean_train_loss}')


        # === Evaluation ===
        if epoch % eval_intervals == 0 or epoch == num_of_epochs - 1:
            if phase is Phases.TRAINING:
                validation_metrics, metrics_list = epoch_evaluation(validation_data, model, conf, epoch, Phases.VALIDATION, save_predictions=True, bundle_adjustment=ba_during_training)
            else:
                validation_metrics, metrics_list = epoch_evaluation(train_data, model, conf, epoch, phase, save_predictions=True, bundle_adjustment=ba_during_training)

            metrics_list_all = metrics_list
            if fabric.world_size > 1:
                metrics_list_all2 = [None for _ in range(fabric.world_size)]
                all_gather_object(metrics_list_all2, metrics_list)
                metrics_list_all = []
                [metrics_list_all.extend(e) for e in metrics_list_all2]

            if fabric.global_rank == 0:
                validation_metrics = evaluation.organize_errors(metrics_list_all)
                validation_metrics2 = validation_metrics.copy()
                validation_metrics2.insert(loc=0, column='epoch', value=[epoch] * len(validation_metrics2))

                if phase is Phases.TRAINING:
                    save_metrics_excel(conf, phase=Phases.VALIDATION, df=validation_metrics, epoch=epoch, append=False)
                    general_utils.write_results(conf, validation_metrics2, file_name="Validation_over_epochs", append=True)
                elif phase is Phases.FINE_TUNE:
                    validation_metrics2 = validation_metrics2.drop(['Mean'])
                    # save_metrics_excel(conf, phase=Phases.FINE_TUNE, df=validation_metrics2, epoch=None, append=True)
                    general_utils.write_results(conf, validation_metrics2, file_name="Validation_over_fine_tuning", phase=phase, append=True)

                print(validation_metrics.to_string(), flush=True)

                metric = validation_metrics.loc[["Mean"], validation_metric].sum(axis=1).values.item()

                for scene in validation_metrics.index:
                    if scene != 'Mean':
                        continue
                    for metric_name in validation_metrics.columns:
                        wandb.log({"Validation_" + metric_name: validation_metrics.loc[scene, metric_name]}, step=epoch)

                path = path_utils.path_to_model(conf, phase, epoch=epoch, best=False)
                current_model = copy.deepcopy(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': current_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)

                is_better = (metric < best_validation_metric) if phase is Phases.FINE_TUNE else (metric > best_validation_metric)
                if is_better:
                    converge_time = time() - begin_time
                    best_validation_metric = metric
                    best_epoch = epoch
                    best_model = copy.deepcopy(model)
                    print(f'Updated best validation metric: {best_validation_metric} time so far: {converge_time}')
                    path = path_utils.path_to_model(conf, phase, epoch=epoch, best=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)



    # === Final Evaluation ===
    train_stat = {}
    run_ba = conf.get_bool('ba.run_ba', default=True)

    best_model_path = path_utils.path_to_model(conf, phase, epoch=None, best=True)
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    best_model = model

    print(f"Evaluating using the best model: {best_model_path}")

    if phase is not Phases.TRAINING:
        print("Evaluate training set")
        train_errors, _ = epoch_evaluation(train_data, best_model, conf, None, phase, save_predictions=True, bundle_adjustment=run_ba)
    else:
        train_errors = None

    if phase is Phases.TRAINING:
        print("Evaluate validation set")
        validation_errors, _ = epoch_evaluation(validation_data, best_model, conf, None, Phases.VALIDATION, save_predictions=True, bundle_adjustment=run_ba)
    else:
        validation_errors = None

    if phase in [Phases.TRAINING]:
        print("Evaluate test set")
        test_errors, _ = epoch_evaluation(test_data, best_model, conf, None, Phases.TEST, save_predictions=True, bundle_adjustment=run_ba)
    else:
        test_errors = None

    # === Return Statistics ===
    train_stat['Convergence time'] = converge_time
    train_stat['best_epoch'] = best_epoch
    train_stat['best_validation_metric'] = best_validation_metric
    train_stat = pd.DataFrame([train_stat])

    return train_stat, train_errors, validation_errors, test_errors


def test(conf, model, phase, train_data=None, validation_data=None, test_data=None, fabri=None, run_ba=False):
    global fabric
    fabric = fabri
    # Load best model checkpoint
    best_model_path = path_utils.path_to_model(conf, phase, epoch=None, best=True)
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    best_model = model
    print(f"Evaluating using the best model: {best_model_path}")

    # Evaluate on train data
    if train_data is not None:
        print("Evaluate training set")
        train_data = fabric.setup_dataloaders(train_data)
        train_errors, _ = epoch_evaluation(
            train_data, best_model, conf, epoch=None,
            phase=phase, save_predictions=True, bundle_adjustment=run_ba
        )
    else:
        train_errors = None

    # Evaluate on validation data
    if validation_data is not None:
        print("Evaluate validation set")
        validation_data = fabric.setup_dataloaders(validation_data)
        validation_errors, _ = epoch_evaluation(
            validation_data, best_model, conf, epoch=None,
            phase=Phases.VALIDATION, save_predictions=True, bundle_adjustment=run_ba
        )
    else:
        validation_errors = None

    # Evaluate on test data
    if test_data is not None:
        print("Evaluate test set")
        test_data = fabric.setup_dataloaders(test_data)
        test_errors, _ = epoch_evaluation(
            test_data, best_model, conf, epoch=None,
            phase=Phases.TEST, save_predictions=True, bundle_adjustment=run_ba
        )
    else:
        test_errors = None

    return train_errors, validation_errors, test_errors