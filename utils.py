import os
import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def save_model(model: nn.Module, save_path: str, epoch: int):
    """
    Checkpoints model state
    :param model:
    :param save_path:
    :param epoch:
    :return:
    """

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file_path = os.path.join(save_path, f"model_{epoch}.pth")

    states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
    }
    torch.save(states, save_file_path)


def get_model(n_classes: int) -> nn.Module:
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
    return model
