import os
import cv2
import logging
import random
from argparse import ArgumentParser
from glob import glob
import json
import pandas as pd
import numpy as np


def extract_frames(data_path: str, N: int) -> None:
    """
    Extracts every Nth frame from a video file and saves to disk.
    :param data_path: Path to data directory
    :param N: frequency of frames being stored
    """

    video_path = glob(f"{data_path}/raw/*.mp4")[0]

    cap = cv2.VideoCapture(video_path)

    # create directory for frames
    frames_path = os.path.join(data_path, "frames")
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
        logger.info(f"Created new directory {frames_path}")

    logger.info(f"Video contains {cap.get(cv2.CAP_PROP_FRAME_COUNT)} total frames")

    curr_frame = 0

    logger.info("[>] Extracting frames...")

    while True:

        ret, frame = cap.read()

        if ret:
            if curr_frame % N == 0:
                file_name = f"frame{curr_frame}.jpg"
                logger.info(f"Creating... {file_name}")

                img_path = os.path.join(frames_path, file_name)
                cv2.imwrite(img_path, frame)
        else:
            break

        curr_frame += 1

    logger.info("[X] Done!")

    # Release all space and windows once done
    cap.release()


def split_dataset(work_dir: str, train_p: float) -> None:
    """
    Splits dataset into train/valid sets with the given train_p threshold and saves to disk.
    The remaining 1-train_p images are placed in the validation set.
    within respective directories.

    :param work_dir: path to current working directory
    :param train_p: percentage of dataset to hold out in training set
    :return:
    """
    data_path = os.path.join(work_dir, "data", "annotated")
    train_path = os.path.join(data_path, "train.json")
    valid_path = os.path.join(data_path, "valid.json")

    with open(os.path.join(data_path, "result.json"), "r") as fp:
        annot_data = json.load(fp)

    df_annot = pd.json_normalize(annot_data["annotations"])
    df_imgs = pd.json_normalize(annot_data["images"])

    logger.info(
        f"[>] Splitting dataset with ({train_p*100:.0f}/{(1-train_p)*100:.0f}) split..."
    )

    mask = np.random.random(df_imgs.shape[0]) < train_p

    train_df = df_imgs[mask]
    valid_df = df_imgs[~mask]

    logger.info(f"[>] Training set - {train_df.shape[0]} images")
    logger.info(f"[>] Validation set - {valid_df.shape[0]} images")

    train_merged = pd.merge(
        df_annot,
        train_df.rename(columns={"id": "image_id"}),
        how="inner",
        on="image_id",
    )
    valid_merged = pd.merge(
        df_annot,
        valid_df.rename(columns={"id": "image_id"}),
        how="inner",
        on="image_id",
    )

    annot_cols = [
        "id",
        "image_id",
        "category_id",
        "segmentation",
        "bbox",
        "ignore",
        "iscrowd",
        "area",
    ]

    # convert dataframes to json and save to disk
    train_json = {}
    train_json["images"] = json.loads(train_df.to_json(orient="records"))
    train_json["categories"] = annot_data["categories"]
    train_json["annotations"] = json.loads(
        train_merged[annot_cols].to_json(orient="records")
    )
    train_json["info"] = annot_data["info"]

    valid_json = {}
    valid_json["images"] = json.loads(valid_df.to_json(orient="records"))
    valid_json["categories"] = annot_data["categories"]
    valid_json["annotations"] = json.loads(
        valid_merged[annot_cols].to_json(orient="records")
    )
    valid_json["info"] = annot_data["info"]

    logger.info(f"[>] Saving training set to - {train_path}")
    with open(train_path, "w") as fp:
        json.dump(train_json, fp, indent=2)

    logger.info(f"[>] Saving validation set to - {valid_path}")
    with open(valid_path, "w") as fp:
        json.dump(valid_json, fp, indent=2)

    logger.info(f"[X] Done!")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dir_path", type=str, help="Data directory path")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Flag to kickoff frame extraction from a video file",
    )
    parser.add_argument(
        "--nth_frame", default=2, type=int, help="Frequency of frame to extract"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Flag to  create train/valid split of dataset",
    )
    parser.add_argument(
        "--percent_train",
        type=float,
        default=0.9,
        help="Percentage of dataset to place in training set",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger()

    work_dir = os.getcwd()
    data_path = os.path.join(work_dir, args.dir_path)

    if args.extract:
        extract_frames(data_path=data_path, N=args.nth_frame)

    if args.split:
        split_dataset(
            work_dir=work_dir,
            train_p=args.percent_train,
        )
