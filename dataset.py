"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""
import random
import traceback

import config
import numpy as np
import os
import pandas as pd
import torch
import json
import re
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression_torch_nms as nms,
    plot_image
)
import hashlib

ImageFile.LOAD_TRUNCATED_IMAGES = True

def list_files_paths(directory):
    filenames = os.listdir(directory)
    full_paths = []
    names = []
    for filename in filenames:
        ext = os.path.splitext(filename)[1]
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path) and (ext.lower() in ['.jpg', '.png']):
            full_paths.append(full_path)
            names.append(filename)
    return full_paths, names

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=8,
        transform=None,
        #ret_org_im= False,
        #ret_org_ims = False,
        #ret_flows_paths = False,
        # prev_frames_jsons_dir="",
        # vod=False,
        # number_of_support_frames = 0
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        #self.ret_org_im = ret_org_im
        #self.ret_org_ims = ret_org_ims
        #self.ret_flows_paths = ret_flows_paths
        # self.prev_frames_jsons_dir = prev_frames_jsons_dir
        # self.vod = vod
        # self.number_of_support_frames = number_of_support_frames

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # For VOD load label path and path of the 3 previous frames
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # print("Label path: ", label_path)
        annotations = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
        annotation_base = annotations[:, 0:5]
        contacts = annotations[:, 5:6]
        annotation_unit_vec_xy = annotations[:, 6:8]
        annotation_unit_vec_mag = annotations[:, 8:9]
        bboxes_base = np.roll(annotation_base, 4, axis=1)
        bboxes_base[:, :4] = np.clip(bboxes_base[:, :4], a_min=0.0000001, a_max=1.0)
        # Ensure that x y w h
        image_name = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.img_dir, image_name)
        org_image = np.array(Image.open(img_path).convert("RGB"))

        # org_images = []
        # org_images.append(org_image)

        if self.transform:
            augmentations = self.transform(image=org_image, bboxes=bboxes_base)
            image = augmentations["image"]
            bboxes_base = augmentations["bboxes"]

        # if config.DO_VOD:

        #     images = []
        #     images.append(image)
        #     flows_paths = []
        #     if "100DOH" in image_name:
        #         used_prev_100doh_frames = False
        #         if not config.NO_PREV_FRAMES_100DOH:
        #             try:
        #                 if image_name in config.prev_frames_100doh_asso_dict:
        #                     prev_frames = config.prev_frames_100doh_asso_dict[image_name]
        #                     valid_prev_frames_paths = []

        #                     last_16 = image_name.strip()[-16:]
        #                     frame_index = int(re.search(r'\d+', last_16).group())

        #                     prev_frames_filter = [f"_frame{str(frame_index-1).zfill(6)}.jpg",
        #                                           f"_frame{str(frame_index-2).zfill(6)}.jpg",
        #                                           f"_frame{str(frame_index-3).zfill(6)}.jpg",
        #                                           f"_frame{str(frame_index-4).zfill(6)}.jpg"]

        #                     for pfname, pfaf in prev_frames.items():
        #                         if pfname.strip()[-16:] in prev_frames_filter:
        #                             valid_prev_frames_paths.append({pfname: os.path.join(config.PREV_FRAMES_DIR_100DOH_PATH, pfaf)})

        #                     if len(valid_prev_frames_paths) >= config.NUM_OF_SUP_FRAMES:
        #                         list_of_prev_filtered_prev_frames = random.sample(valid_prev_frames_paths, config.NUM_OF_SUP_FRAMES)

        #                         for valid_prev_frame_path in list_of_prev_filtered_prev_frames:
        #                             key = next(iter(valid_prev_frame_path))
        #                             val = valid_prev_frame_path[key]
        #                             hashed_filename_76 = hashlib.sha256((image_name + "_" + key + "_" + "76").encode('utf-8')).hexdigest()
        #                             hashed_filename_52 = hashlib.sha256((image_name + "_" + key + "_" + "52").encode('utf-8')).hexdigest()
        #                             hashed_filename_38 = hashlib.sha256((image_name + "_" + key + "_" + "38").encode('utf-8')).hexdigest()
        #                             hashed_filename_26 = hashlib.sha256((image_name + "_" + key + "_" + "26").encode('utf-8')).hexdigest()
        #                             hashed_filename_19 = hashlib.sha256((image_name + "_" + key + "_" + "19").encode('utf-8')).hexdigest()
        #                             hashed_filename_13 = hashlib.sha256((image_name + "_" + key + "_" + "13").encode('utf-8')).hexdigest()
        #                             flows_paths.append({
        #                                 "52": [os.path.join(config.PREV_FRAMES_416_FLOWS_DIR_PATH, hashed_filename_52 + ".npz"), image_name + "_" + os.path.basename(key) + "_" + "52"],
        #                                 "26": [os.path.join(config.PREV_FRAMES_416_FLOWS_DIR_PATH, hashed_filename_26 + ".npz"), image_name + "_" + os.path.basename(key) + "_" + "26"],
        #                                 "13": [os.path.join(config.PREV_FRAMES_416_FLOWS_DIR_PATH, hashed_filename_13 + ".npz"), image_name + "_" + os.path.basename(key) + "_" + "13"],
        #                                 "76": [os.path.join(config.PREV_FRAMES_608_FLOWS_DIR_PATH, hashed_filename_76 + ".npz"), image_name + "_" + os.path.basename(key) + "_" + "76"],
        #                                 "38": [os.path.join(config.PREV_FRAMES_608_FLOWS_DIR_PATH, hashed_filename_38 + ".npz"), image_name + "_" + os.path.basename(key) + "_" + "38"],
        #                                 "19": [os.path.join(config.PREV_FRAMES_608_FLOWS_DIR_PATH, hashed_filename_19 + ".npz"), image_name + "_" + os.path.basename(key) + "_" + "19"],
        #                             })

        #                             sup_frame = np.array(Image.open(val).convert("RGB"))
        #                             if self.ret_org_im:
        #                                 org_images.append(sup_frame)
        #                             augmentations = self.transform(image=sup_frame, bboxes=[])
        #                             sup_image = augmentations["image"]
        #                             images.append(sup_image)
        #                         used_prev_100doh_frames = True
        #             except Exception as ex:
        #                 traceback.print_exc()

        #         if used_prev_100doh_frames == False:
        #             for ni in range(0, config.NUM_OF_SUP_FRAMES):
        #                 images.append(image)
        #                 if self.ret_org_im:
        #                     org_images.append(org_image)
        #             if self.ret_org_im:
        #                 org_images.append(np.array([True]))
        #     else:
        #         used_prev_dl_frames = False
        #         try:
        #             file_with_prev_frames_paths_list_path = os.path.join(config.PREV_FRAMES_DIR, os.path.splitext(os.path.basename(image_name))[0] + ".json")
        #             with open(file_with_prev_frames_paths_list_path, 'r') as openfile:
        #                 list_of_support_frames_paths = json.load(openfile)
        #             if isinstance(list_of_support_frames_paths, list) and len(list_of_support_frames_paths) >= config.NUM_OF_SUP_FRAMES:
        #                 # Choose config.NUM_OF_SUP_FRAMES: randomly from the list of support frames
        #                 list_of_prev_filtered_prev_frames = random.sample(list_of_support_frames_paths, config.NUM_OF_SUP_FRAMES)
        #                 for valid_prev_frame_path in list_of_prev_filtered_prev_frames:
        #                     valid_prev_frame_path = valid_prev_frame_path.replace("/media/drcat/Dati/dataset_dl", config.DL_PREV_FRAMES_BASE_PATH)
        #                     if not os.path.exists(valid_prev_frame_path):
        #                         sup_frame_im_fn = os.path.basename(valid_prev_frame_path)
        #                         sup_frame_im_dirname = os.path.dirname(valid_prev_frame_path)
        #                         valid_prev_frame_path = os.path.join(sup_frame_im_dirname, "0" + sup_frame_im_fn)
        #                     sup_frame = np.array(Image.open(valid_prev_frame_path).convert("RGB"))

        #                     hashed_filename_76 = hashlib.sha256((image_name + "_" + os.path.basename(valid_prev_frame_path) + "_" + "76").encode('utf-8')).hexdigest()
        #                     hashed_filename_52 = hashlib.sha256((image_name + "_" + os.path.basename(valid_prev_frame_path) + "_" + "52").encode('utf-8')).hexdigest()
        #                     hashed_filename_38 = hashlib.sha256((image_name + "_" + os.path.basename(valid_prev_frame_path) + "_" + "38").encode('utf-8')).hexdigest()
        #                     hashed_filename_26 = hashlib.sha256((image_name + "_" + os.path.basename(valid_prev_frame_path) + "_" + "26").encode('utf-8')).hexdigest()
        #                     hashed_filename_19 = hashlib.sha256((image_name + "_" + os.path.basename(valid_prev_frame_path) + "_" + "19").encode('utf-8')).hexdigest()
        #                     hashed_filename_13 = hashlib.sha256((image_name + "_" + os.path.basename(valid_prev_frame_path) + "_" + "13").encode('utf-8')).hexdigest()
        #                     flows_paths.append({
        #                         "52": [os.path.join(config.PREV_FRAMES_416_FLOWS_DIR_PATH, hashed_filename_52 + ".npz"), image_name + "_" + os.path.basename(valid_prev_frame_path) + "_" + "52"],
        #                         "26": [os.path.join(config.PREV_FRAMES_416_FLOWS_DIR_PATH, hashed_filename_26 + ".npz"), image_name + "_" + os.path.basename(valid_prev_frame_path) + "_" + "26"],
        #                         "13": [os.path.join(config.PREV_FRAMES_416_FLOWS_DIR_PATH, hashed_filename_13 + ".npz"), image_name + "_" + os.path.basename(valid_prev_frame_path) + "_" + "13"],
        #                         "76": [os.path.join(config.PREV_FRAMES_608_FLOWS_DIR_PATH, hashed_filename_76 + ".npz"), image_name + "_" + os.path.basename(valid_prev_frame_path) + "_" + "76"],
        #                         "38": [os.path.join(config.PREV_FRAMES_608_FLOWS_DIR_PATH, hashed_filename_38 + ".npz"), image_name + "_" + os.path.basename(valid_prev_frame_path) + "_" + "38"],
        #                         "19": [os.path.join(config.PREV_FRAMES_608_FLOWS_DIR_PATH, hashed_filename_19 + ".npz"), image_name + "_" + os.path.basename(valid_prev_frame_path) + "_" + "19"],
        #                     })
        #                     if self.ret_org_im:
        #                         org_images.append(sup_frame)
        #                     augmentations = self.transform(image=sup_frame, bboxes=[])
        #                     sup_image = augmentations["image"]
        #                     images.append(sup_image)
        #                 used_prev_dl_frames = True
        #         except Exception as ex:
        #             traceback.print_exc()

        #         if used_prev_dl_frames == False:
        #             for ni in range(0, config.NUM_OF_SUP_FRAMES):
        #                 images.append(image)
        #                 if self.ret_org_im:
        #                     org_images.append(org_image)
        #             if self.ret_org_im:
        #                 org_images.append(np.array([True]))
        #             # idx = 0
        #             # for sup_frame_path in list_of_support_frames_paths:
        #             #     if idx == config.NUM_OF_SUP_FRAMES:
        #             #         break
        #             #     try:
        #             #         sup_frame = np.array(Image.open(sup_frame_path).convert("RGB"))
        #             #     except:
        #             #         print("Failed to load image with path: ", sup_frame_path)
        #             #         sup_frame = None
        #             #     # If sup frame is none the image name could need a zero infront 999867.jpg -> 0999867.jpg
        #             #
        #             #     if sup_frame is None:
        #             #         sup_frame_im_fn = os.path.basename(sup_frame_path)
        #             #         sup_frame_im_dirname = os.path.dirname(sup_frame_path)
        #             #         new_sup_frame_path = os.path.join(sup_frame_im_dirname, "0" + sup_frame_im_fn)
        #             #         sup_frame = np.array(Image.open(new_sup_frame_path).convert("RGB"))
        #             #
        #             #     if self.transform:
        #             #         augmentations = self.transform(image=sup_frame, bboxes=[])
        #             #         sup_image = augmentations["image"]
        #             #     images.append(sup_image)
        #             #     idx += 1
        #     image = images
        #     # image = torch.cat(images, dim=0)

        #
        # if label_path == "100DOH_DL/labels/100DOH_repair_v_Gpk-ptHtIg0_frame000027.txt":
        #     print("Hei!")

        # print("__getitem__ cur image path: ", img_path)



        bboxes = np.hstack((bboxes_base, contacts, annotation_unit_vec_mag, annotation_unit_vec_xy))

        # print("__getitem__ init bboxes: ", bboxes)
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        #  The last dimension is 9 because we have: objectness x y w h class contact_state mag unitx unity
        targets = [torch.zeros((self.num_anchors // 3, S, S, 10)) for S in self.S]
        for box in bboxes:
            # For each ground truth bbox we check which anchor box overlap more
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            # Sort ious
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label, contact, magnitude, unitdx, unitdy = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    hoi_params = torch.tensor([contact, magnitude, unitdx, unitdy])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    targets[scale_idx][anchor_on_scale, i, j, 6:11] = hoi_params
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        # if self.ret_flows_paths:
        #     return image, tuple(targets), flows_paths

        # if self.ret_org_im:
        #     return image, tuple(targets), org_images

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS
    IMAGE_SIZE = 416
    transform = config.test_transforms
    dataset_name = "100DOH_DL"
    dataset = YOLODataset(
        dataset_name+"/test.csv",
        dataset_name+"/images/",
        dataset_name+"/labels/",
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        anchors=anchors,
        transform=transform,
    )
    S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    for x, y in loader:
        # continue
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            # print(anchor.shape)
            # print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=0.85, threshold=0.7, box_format="midpoint")
        print(boxes)
        # plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes, org_im_shape)


# if __name__ == "__main__":
#     test()
