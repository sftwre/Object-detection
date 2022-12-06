import os
import random
import cv2
import utils
import torch
import argparse
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import glob
from typing import List, Dict
from tqdm import tqdm


WORK_DIR = os.getcwd()
IMG_HEIGHT = 360
IMG_WIDTH = 640


def annotate_frame(outputs, detection_threshold, classes, colors, orig_image, image):
    height, width, _ = orig_image.shape
    boxes = outputs["boxes"].data.numpy()
    scores = outputs["scores"].data.numpy()
    # Filter out boxes according to `detection_threshold`.
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    draw_boxes = boxes.copy()
    # Get all the predicited class names.
    pred_classes = [classes[i] for i in outputs["labels"].cpu().numpy()]

    lw = max(round(sum(orig_image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1)  # Font thickness.

    # Draw the bounding boxes and write the class name on top of it.
    for j, box in enumerate(draw_boxes):
        p1 = (
            int(box[0] / image.shape[3] * width),
            int(box[1] / image.shape[2] * height),
        )
        p2 = (
            int(box[2] / image.shape[3] * width),
            int(box[3] / image.shape[2] * height),
        )
        class_name = pred_classes[j]
        color = colors[classes.index(class_name)]
        cv2.rectangle(
            orig_image, p1, p2, color=color, thickness=lw, lineType=cv2.LINE_AA
        )

        # For filled rectangle.
        final_label = class_name + " " + str(round(scores[j], 2))
        w, h = cv2.getTextSize(
            final_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=lw / 3, thickness=tf
        )[
            0
        ]  # text width, height
        w = int(w - (0.20 * w))
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(
            orig_image, p1, p2, color=color, thickness=-1, lineType=cv2.LINE_AA
        )
        cv2.putText(
            orig_image,
            final_label,
            (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=lw / 3.8,
            color=(255, 255, 255),
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    return orig_image


def log_predictions(args):
    global device
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(WORK_DIR, "data", args.data_path)

    # load pre-trained model
    model = utils.get_model(args.n_classes)
    model_path = os.path.join(WORK_DIR, "models", args.model_name)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    annot_path = os.path.join(WORK_DIR, "data", "annotated", "valid.json")

    with open(annot_path, "r") as file:
        data = json.load(file)

    # load validation image paths
    if args.store_video:
        file_names = glob.glob(f"{data_path}/raw/*.mp4")
    else:
        file_names = glob.glob(f"{data_path}/frames/*.jpg")
        file_names = random.sample(file_names, args.n_images)

    colors = {
        0: (226, 190, 236),
        1: (34, 14, 242),
        2: (90, 180, 239),
    }
    classes = [k["name"] for k in data["categories"]]

    if args.tensorboard:
        writer = SummaryWriter(log_dir="./runs/inference")
        write_images(
            model, data_path, file_names, writer, colors, classes, args.reshape_size
        )

    if args.store_video:
        write_video(model, data_path, file_names[0], colors, classes, args.reshape_size)


def write_images(
    model,
    data_path: str,
    file_names: List[str],
    writer: SummaryWriter,
    colors: Dict,
    classes: List,
    reshape_size: int,
):

    images = np.zeros((len(file_names), 3, IMG_HEIGHT, IMG_WIDTH))

    print("[>] Annotating frames...")
    # annotate images and save to tensorboard
    for i, img_name in tqdm(enumerate(file_names)):
        img_path = os.path.join(data_path, "images", img_name)
        img = cv2.imread(img_path)
        img_annotated = process_frame(model, img, colors, classes, reshape_size)
        images[i] = img_annotated.transpose((2, 0, 1))

    writer.add_images("Inference", images, global_step=0)
    writer.close()


def write_video(model, data_path: str, file_name: str, colors, classes, reshape_size):

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        f"{data_path}/{file_name.split('/')[-1]}_annot.mp4",
        fourcc,
        20.0,
        (IMG_WIDTH, IMG_HEIGHT),
    )

    video_path = os.path.join(data_path, file_name)
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Processing video...")

    for _ in tqdm(range(n_frames)):

        ret, frame = cap.read()

        if ret:
            annot_frame = process_frame(
                model, frame, colors=colors, classes=classes, reshape_size=reshape_size
            )
        else:
            break

        annot_frame = cv2.cvtColor(annot_frame, cv2.COLOR_RGB2BGR)
        out.write((annot_frame * 255).astype(np.uint8))

    out.release()
    cap.release()


def process_frame(model, img: np.ndarray, colors, classes, reshape_size) -> np.ndarray:

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0

    resize_dims = (reshape_size, reshape_size)
    img_resized = cv2.resize(img, resize_dims, interpolation=cv2.INTER_CUBIC)
    img_resized = torch.tensor(img_resized).to(device)
    img_resized = torch.permute(img_resized, (2, 0, 1))
    img_resized = torch.unsqueeze(img_resized, 0)

    with torch.no_grad():
        outputs = model(img_resized)

    outputs = {k: v.to("cpu") for k, v in outputs[0].items()}
    img_annotated = annotate_frame(
        outputs,
        detection_threshold=args.thresh,
        colors=colors,
        classes=classes,
        orig_image=img,
        image=img_resized,
    )
    return img_annotated


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of model stored in models directory",
    )
    parser.add_argument(
        "--reshape_size", type=int, default=256, help="height/width of reshaped image"
    )
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--n_classes", type=int, default=3, help="Number of gt classes")
    parser.add_argument(
        "--thresh", type=float, help="Cut-off confidence score for bounding box"
    )
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--store_video", action="store_true")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--n_images", type=int)

    args = parser.parse_args()
    log_predictions(args)
