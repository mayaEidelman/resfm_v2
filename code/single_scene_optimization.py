import cv2  # DO NOT REMOVE
from datasets import SceneData, ScenesDataSet
import train
from utils import general_utils, path_utils
from utils.Phases import Phases
import torch

from lightning.fabric import Fabric

fabric = Fabric(accelerator="cuda", devices="auto", strategy="ddp")
fabric.launch()

def train_single_model(conf, device, phase):

    if phase is Phases.FINE_TUNE:
        print("Run test before fine-tuning")
        print("models." + conf.get_string("model.type"))
        model = general_utils.get_class("models." + conf.get_string("model.type"))(conf, Phases.TEST).to(device)
        print(model)
        scene_data = SceneData.create_scene_data(conf, phase=Phases.TEST)
        scene_dataset = ScenesDataSet.ScenesDataSet([scene_data], return_all=True)
        scene_loader = torch.utils.data.DataLoader(scene_dataset, collate_fn=ScenesDataSet.collate_fn)
        _, _, test_errors = train.test(conf, model, Phases.TEST, train_data=None, validation_data=None, test_data=scene_loader, fabri=fabric, run_ba=False)
        print(test_errors.to_string(), flush=True)
        test_errors = test_errors.drop(['Mean'])
        general_utils.write_results(conf, test_errors, file_name="myTest_fineTuning", phase=phase, append=True)

    # Create data
    scene_data = SceneData.create_scene_data(conf, phase=phase)

    # Create model
    print("models." + conf.get_string("model.type"))
    model = general_utils.get_class("models." + conf.get_string("model.type"))(conf, phase).to(device)
    print(model)
    print(f'Number of parameters: {sum([x.numel() for x in model.parameters()])}')
    print(f'Number of trainable parameters:: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if phase is Phases.FINE_TUNE:
        path = path_utils.path_to_model(conf, Phases.TRAINING, epoch=None, best=True)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Fine-tuning a scene - The model is resuming from checkpoint: ", path)


    # Optimize Scene
    scene_dataset = ScenesDataSet.ScenesDataSet([scene_data], return_all=True)
    scene_loader = torch.utils.data.DataLoader(scene_dataset, collate_fn=ScenesDataSet.collate_fn)
    train_stat, train_errors, _, _ = train.train(conf, scene_loader, model, phase, fabri=fabric)
    train_errors.drop("Mean", inplace=True)
    print(train_errors.to_string(), flush=True)

    # Write results
    train_stat["Scene"] = train_errors.index
    train_stat.set_index("Scene", inplace=True)
    train_res = train_errors.join(train_stat)
    general_utils.write_results(conf, train_res, file_name="Results_" + phase.name, append=True)


if __name__ == "__main__":
    conf, device, phase = general_utils.init_exp(Phases.OPTIMIZATION.name)
    train_single_model(conf, device, phase)
