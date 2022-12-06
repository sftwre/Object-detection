import os
import logging
import sys
import numpy as np
import utils
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn.parallel import DataParallel
from dataset import NATODataset
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from torch_coco_utils.engine import evaluate
from torch_coco_utils.utils import collate_fn


# TODO: log eval metrics in tensorboard
# TODO: create writeup of results


def main(args):

    logger = logging.getLogger()

    logger.info("[>] Loading datasets...")
    data_path = os.path.join(os.getcwd(), "data", "annotated")
    train_ds = NATODataset(dir_path=data_path, split="train", resize_dim=args.resize)
    valid_ds = NATODataset(dir_path=data_path, split="valid", resize_dim=args.resize)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    logger.info("[>] Initializing Faster R-CNN...")

    # load pre-trained Faster R-CNN model
    model = utils.get_model(args.n_classes)

    if torch.cuda.is_available():
        # ids = ",".join(list(map(lambda x: str(x), args.gpu_ids)))
        device_str = f"cuda:2"
    else:
        device_str = "cpu"

    device = torch.device(device_str)

    start_epoch = 0

    # load model from checkpoint
    if args.model_pth is not None:
        logger.info(f"[>] Loading model checkpoint {args.model_pth}...")
        checkpoint = torch.load(args.model_pth, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    torch.cuda.empty_cache()
    model.to(device)
    # model = DataParallel(model, device_ids=args.gpu_ids)

    logger.info(f"[>] Model training on {device_str}")

    # init optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    # get trainable parameters
    optimizer = torch.optim.SGD(
        parameters, lr=args.lr, momentum=0.9, weight_decay=0.0005
    )

    # init summary writer
    writer = None
    running_loss_avg_total = sys.maxsize

    if args.tensorboard:
        log_dir = f"./runs/{args.lr}LR_{args.batch_size}BS_{args.n_epochs}EPOCHS"
        writer = SummaryWriter(log_dir=log_dir)

    # lists to store epoch level metrics
    train_loss_list = []
    loss_cls_list = []
    loss_box_reg_list = []
    loss_objectness_list = []
    loss_rpn_list = []
    train_loss_list_epoch = []

    def train_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ):

        batch_loss_list = []
        batch_loss_cls_list = []
        batch_loss_box_reg_list = []
        batch_loss_objectness_list = []
        batch_loss_rpn_list = []

        model.train()
        for imgs, tgts in data_loader:

            # place inputs on gpu/cpu
            imgs = list(img.to(device) for img in imgs)
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

            loss_dict = model(imgs, tgts)

            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # store batch losses
            ttl_loss = losses.item()

            batch_loss_list.append(ttl_loss)
            batch_loss_cls_list.append(loss_dict["loss_classifier"].detach().cpu())
            batch_loss_box_reg_list.append(loss_dict["loss_box_reg"].detach().cpu())
            batch_loss_objectness_list.append(
                loss_dict["loss_objectness"].detach().cpu()
            )
            batch_loss_rpn_list.append(loss_dict["loss_rpn_box_reg"].detach().cpu())

        return (
            batch_loss_list,
            batch_loss_cls_list,
            batch_loss_box_reg_list,
            batch_loss_objectness_list,
            batch_loss_rpn_list,
        )

    # main training loop
    for epoch in range(start_epoch, args.n_epochs):
        # train
        (
            batch_loss_list,
            batch_loss_cls_list,
            batch_loss_box_reg_list,
            batch_loss_objectness_list,
            batch_loss_rpn_list,
        ) = train_epoch(model, train_loader, device, optimizer)

        train_loss_list.extend(batch_loss_list)
        loss_cls_list.append(np.mean(batch_loss_cls_list))
        loss_box_reg_list.append(np.mean(np.array(batch_loss_box_reg_list)))
        loss_objectness_list.append(np.mean(np.array(batch_loss_objectness_list)))
        loss_rpn_list.append(np.mean(np.array(batch_loss_rpn_list)))
        train_loss_list_epoch.append(np.mean(batch_loss_list))

        scalars = {}
        scalars["mean_ttl_loss"] = train_loss_list_epoch[epoch]
        scalars["cls_loss"] = loss_cls_list[epoch]
        scalars["box_reg_loss"] = loss_box_reg_list[epoch]
        scalars["objectness_loss"] = loss_objectness_list[epoch]
        scalars["rpn_loss"] = loss_rpn_list[epoch]

        # checkpoint model
        if scalars["mean_ttl_loss"] < running_loss_avg_total:
            running_loss_avg_total = scalars["mean_ttl_loss"]
            utils.save_model(model, os.path.join(os.getcwd(), "models"), epoch)

        if writer:
            # log results
            writer.add_scalars(
                "Training loss", tag_scalar_dict=scalars, global_step=epoch
            )
        else:
            # used for debugging to reduce clutter in runs directory
            logger.info(scalars)

        # eval
        model.eval()
        coco_evaluator = evaluate(model, valid_loader, device)
        coco_evaluator.summarize()

    logger.info("[X] Training complete!")
    logger.info("[>] Saving model...")
    logger.info("[X] Done!")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--n_epochs", default=10, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", default=8, type=int, help="Batch size for dataloader"
    )
    parser.add_argument("--resize", default=256, type=int, help="Resize dimension")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument(
        "--model_pth",
        required=False,
        type=str,
        default=None,
        help="checkpoint model to load",
    )
    parser.add_argument(
        "--n_classes", default=3, type=int, help="Number of object categories"
    )
    parser.add_argument(
        "--gpu_ids", nargs="+", type=int, help="Number of GPUs to utilize for training"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Flag to enable model tracking via Tensorboard",
    )
    args = parser.parse_args()

    # setup logging config
    logging.basicConfig(level=logging.INFO)

    main(args)
