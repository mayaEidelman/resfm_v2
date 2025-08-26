import cv2  # DO NOT REMOVE
import torch

from utils import general_utils, dataset_utils, path_utils
from utils.Phases import Phases
from datasets.ScenesDataSet import ScenesDataSet, collate_fn
from datasets import SceneData
from single_scene_optimization import train_single_model
import train
import copy

from lightning.fabric import Fabric

fabric = Fabric(accelerator="cuda", devices="auto", strategy="ddp")
# fabric = Fabric(accelerator="cuda", devices=1)
fabric.launch()

def main():
    # Init Experiment
    conf, device, phase = general_utils.init_exp(Phases.TRAINING.name)
    general_utils.log_code(conf) # Log code to the experiment folder (you can comment this line if you don't want to log the code)`

    # Set device
    # Get configuration
    min_sample_size = conf.get_float('dataset.min_sample_size')
    max_sample_size = conf.get_float('dataset.max_sample_size')
    batch_size = conf.get_int('dataset.batch_size')
    optimization_num_of_epochs = conf.get_int("train.optimization_num_of_epochs")
    optimization_eval_intervals = conf.get_int('train.optimization_eval_intervals')
    optimization_lr = conf.get_float('train.optimization_lr')

    if phase is not Phases.FINE_TUNE:
        # Train model
        model = general_utils.get_class("models." + conf.get_string("model.type"))(conf).to(device)
        print(f'Number of parameters: {sum([x.numel() for x in model.parameters()])}')
        print(f'Number of trainable parameters:: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

        # Create train, test and validation sets
        test_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.test_set'), conf)
        validation_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.validation_set'), conf)
        train_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.train_set'), conf)

        train_set = ScenesDataSet(train_scenes, return_all=False, min_sample_size=min_sample_size, max_sample_size=max_sample_size, phase=Phases.TRAINING)
        validation_set = ScenesDataSet(validation_scenes, return_all=True)
        test_set = ScenesDataSet(test_scenes, return_all=True)

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

        fabric.barrier()
        train_stat, train_errors, validation_errors, test_errors = train.train(conf, train_loader, model, phase, validation_loader, test_loader, fabri=fabric)
        if fabric.global_rank == 0:
            # Write results
            general_utils.write_results(conf, train_stat, file_name="Train_Stats")
            # general_utils.write_results(conf, train_errors, file_name="Train") # todo
            general_utils.write_results(conf, validation_errors, file_name="Validation")
            general_utils.write_results(conf, test_errors, file_name="Test")

        test_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.test_set'), conf)
        test_set = ScenesDataSet(test_scenes, return_all=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
        train_errors, validation_errors, test_errors = train.test(conf, model, phase, train_data=None, validation_data=None, test_data=test_loader, fabri=fabric, run_ba=False)
        general_utils.write_results(conf, test_errors, file_name="myTest")


    if fabric.global_rank == 0:
        # Send jobs for fine-tuning
        test_scans_list = conf.get_list('dataset.test_set')

        conf_test = copy.deepcopy(conf)
        conf_test['dataset']['scans_list'] = test_scans_list
        conf_test['train']['num_of_epochs'] = optimization_num_of_epochs
        conf_test['train']['eval_intervals'] = optimization_eval_intervals
        conf_test['train']['lr'] = optimization_lr

        # send jobs
        optimization_all_sets(conf_test, device, Phases.FINE_TUNE)



def optimization_all_sets(conf, device, phase):
    # Get logs directories
    scans_list = conf.get_list('dataset.scans_list')
    for i, scan in enumerate(scans_list):
        conf["dataset"]["scan"] = scan
        train_single_model(conf, device, phase)


if __name__ == "__main__":
    main()