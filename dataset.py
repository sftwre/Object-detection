import os
import cv2
import torch
from torchvision.ops import box_convert
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np


class NATODataset(Dataset):
    """
    Dataset with images of NATO exercise
    """

    def __init__(self, dir_path: str, split: str, resize_dim: int = None):

        self.dir_path = dir_path
        annot_path = os.path.join(
            dir_path, "train.json" if split == "train" else "valid.json"
        )
        self.coco = COCO(annot_path)
        self.split = split
        self.ids = list(self.coco.imgs.keys())

        # define transformations
        if split == "train":
            self.transforms = A.Compose(
                [
                    A.Resize(
                        resize_dim, resize_dim, interpolation=cv2.INTER_CUBIC, p=1
                    ),
                    A.HorizontalFlip(p=0.3),
                    A.RandomBrightnessContrast(p=0.2),
                    ToTensorV2(p=1.0),
                ],
                bbox_params=A.BboxParams(
                    format="coco",
                    min_area=160,
                    min_visibility=0.3,
                    label_fields=["class_labels"],
                ),
            )
        elif split == "valid":
            self.transforms = A.Compose(
                [
                    A.Resize(
                        resize_dim, resize_dim, interpolation=cv2.INTER_CUBIC, p=1
                    ),
                    ToTensorV2(p=1.0),
                ],
                bbox_params=A.BboxParams(
                    format="coco",
                    min_area=160,
                    min_visibility=0.3,
                    label_fields=["class_labels"],
                ),
            )

    def __getitem__(self, item):

        img_id = self.ids[item]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annons = self.coco.loadAnns(ann_ids)
        img_name = self.coco.loadImgs(img_id)[0]["file_name"].split("/")[-1]

        # load image and convert color channels to RGB
        img_path = os.path.join(self.dir_path, "images", img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0

        target = {}
        target["boxes"] = list(map(lambda x: x["bbox"], annons))
        target["labels"] = list(map(lambda x: x["category_id"], annons))
        target["area"] = list(map(lambda x: x["area"], annons))
        target["iscrowd"] = list(map(lambda x: x["iscrowd"], annons))
        target["image_id"] = torch.tensor([item])

        # apply transformations
        tfs_dict = self.transforms(
            image=img, bboxes=target["boxes"], class_labels=target["labels"]
        )

        img = tfs_dict["image"]
        target["boxes"] = torch.tensor(tfs_dict["bboxes"], dtype=torch.int64)

        # pad bbox with zeros if no boxes are present
        if np.isnan((target["boxes"]).numpy()).any() or target[
            "boxes"
        ].shape == torch.Size([0]):
            target["boxes"] = torch.zeros((0, 4), dtype=torch.int64)

        # cast targets to tensor
        target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)
        target["area"] = torch.tensor(target["area"], dtype=torch.int64)
        target["iscrowd"] = torch.tensor(target["iscrowd"], dtype=torch.int64)

        # convert box format from xywh -> xyxy
        target["boxes"] = box_convert(target["boxes"], in_fmt="xywh", out_fmt="xyxy")

        return img, target

    def __len__(self):
        return len(self.ids)
