import os.path

from eval_utils import _load_image_set_index, evaluate_detections
# from model import YOLOv3VOD
import config
import albumentations as Albu
from albumentations.core.bbox_utils import convert_bbox_to_albumentations, convert_bbox_from_albumentations
import torch.optim as optim

# from flow_module import OpticalFlowModule
from utils import (
    cells_to_bboxes,
    non_max_suppression_torch_nms,
    non_max_suppression_torch_nms_rettorcharr,
    load_checkpoint, non_max_suppression_new, cells_to_bboxes_new, non_max_suppression_torch_nms_better
)
import torch
from dataset import YOLODataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from net_utils import vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects
import cv2
import numpy as np
import math
from PIL import Image, ImageFile
import time
import json
# from nvidia_fbop import interpolate_dense_flow, compute_optical_flow, interpolate_dense_flow_rbf, create_dense_flow_from_sparse, flow_to_image, put_optical_flow_arrows_on_image, visualize_sparse_flow, compute_sparse_optical_flow, visualize_sparse_flow_both

def proper_round(num):
    dec, integ = math.modf(num)
    # print("Integ: ", integ, " Dec: ", dec)
    if dec >= 0.5:
        return (int(integ + 1))
    else:
        return (int(integ))

def yolo_to_xml_bbox_xyxy(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = proper_round((bbox[0] * w) - w_half_len)
    ymin = proper_round((bbox[1] * h) - h_half_len)
    xmax = proper_round((bbox[0] * w) + w_half_len)
    ymax = proper_round((bbox[1] * h) + h_half_len)

    # xmin = float((bbox[0] * w) - w_half_len)
    # ymin = float((bbox[1] * h) - h_half_len)
    # xmax = float((bbox[0] * w) + w_half_len)
    # ymax = float((bbox[1] * h) + h_half_len)
    return (xmin, ymin, xmax, ymax)

def xml_bbox_xyxy_to_yolo_bbox(bbox, width, height):
    # x_center, y_center width heigth
    xmin,ymin,xmax,ymax = bbox
    x = (xmin + xmax) / 2.0 / width
    y = (ymin + ymax) / 2.0 / height
    w = (xmax - xmin) / float(width)
    h = (ymax - ymin) / float(height)
    return (x, y, w, h)

def yolo_boxes_unpad_convert_to_xyxy(org_image, tf_image, transformed_bbox, crop_image_trans):
    original_height = org_image.shape[0]
    original_width = org_image.shape[1]
    crop_image_aug = crop_image_trans(image=org_image)
    cropped_image = crop_image_aug["image"]

    width_diff = tf_image.shape[1] - cropped_image.shape[1]
    height_diff = tf_image.shape[0] - cropped_image.shape[0]
    left_padding = width_diff // 2
    right_padding = width_diff - left_padding
    top_padding = height_diff // 2
    bottom_padding = height_diff - top_padding

    left_col = left_padding
    top_row = top_padding

    new_bboxes = []

    scale_orx = original_width / cropped_image.shape[1]
    scale_ory = original_height / cropped_image.shape[0]

    for bbox in transformed_bbox:
        x, y, w, h = bbox

        padded_width = tf_image.shape[1]
        padded_height = tf_image.shape[0]
        w_half_len = (w * padded_width) / 2
        h_half_len = (h * padded_height) / 2
        xmin = (x * padded_width) - w_half_len
        ymin = (y * padded_height) - h_half_len
        xmax = (x * padded_width) + w_half_len
        ymax = (y * padded_height) + h_half_len

        new_x1 = xmin - left_col
        new_y1 = ymin - top_row
        new_x2 = xmax - left_col
        new_y2 = ymax - top_row

        new_x1 *= scale_orx
        new_y1 *= scale_ory
        new_x2 *= scale_orx
        new_y2 *= scale_ory

        inv_box = (new_x1, new_y1, new_x2, new_y2)
        new_bboxes.append(inv_box)

    return new_bboxes

def yolo_boxes_unpad_convert_to_xyxy(org_image, tf_image, transformed_bbox, crop_image_trans):
    original_height = org_image.shape[0]
    original_width = org_image.shape[1]
    crop_image_aug = crop_image_trans(image=org_image)
    cropped_image = crop_image_aug["image"]

    width_diff = tf_image.shape[1] - cropped_image.shape[1]
    height_diff = tf_image.shape[0] - cropped_image.shape[0]
    left_padding = width_diff // 2
    right_padding = width_diff - left_padding
    top_padding = height_diff // 2
    bottom_padding = height_diff - top_padding

    left_col = left_padding
    top_row = top_padding

    new_bboxes = []

    scale_orx = original_width / cropped_image.shape[1]
    scale_ory = original_height / cropped_image.shape[0]

    for bbox in transformed_bbox:
        x, y, w, h = bbox

        padded_width = tf_image.shape[1]
        padded_height = tf_image.shape[0]
        w_half_len = (w * padded_width) / 2
        h_half_len = (h * padded_height) / 2
        xmin = (x * padded_width) - w_half_len
        ymin = (y * padded_height) - h_half_len
        xmax = (x * padded_width) + w_half_len
        ymax = (y * padded_height) + h_half_len

        new_x1 = xmin - left_col
        new_y1 = ymin - top_row
        new_x2 = xmax - left_col
        new_y2 = ymax - top_row

        new_x1 *= scale_orx
        new_y1 *= scale_ory
        new_x2 *= scale_orx
        new_y2 *= scale_ory

        inv_box = (new_x1, new_y1, new_x2, new_y2)
        new_bboxes.append(inv_box)

    return new_bboxes

def yolo_boxes_unpad_convert_to_xyxy_mod(org_image, yolo_input_size, transformed_bbox):
    original_height = org_image.shape[0]
    original_width = org_image.shape[1]

    # Determine the image size after applying the LongestMaxSize transform
    # Determine the scaling factor for both width and height
    scale_factor = yolo_input_size / max(original_width, original_height)

    # Calculate the new width and height
    lms_imw = round(original_width * scale_factor)
    lms_imh = round(original_height * scale_factor)

    # Compute the padding applied by the PadIfNeeded transform
    if lms_imh < yolo_input_size:
        h_pad_top = int((yolo_input_size - lms_imh) / 2.0)
        h_pad_bottom = yolo_input_size - lms_imh - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if lms_imw < yolo_input_size:
        w_pad_left = int((yolo_input_size - lms_imw) / 2.0)
        w_pad_right = yolo_input_size - lms_imw - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    new_bboxes = []

    # Scaling factor to apply to go from the image generated after applying the transformation LongestMaxSize to the original image
    scale_orx = original_width / lms_imw
    scale_ory = original_height / lms_imh

    for bbox in transformed_bbox:
        x, y, w, h = bbox

        # Transform to xyxy format from YOLO format
        tfw = w * yolo_input_size
        tfh = h * yolo_input_size
        xmin = ((2 * x * yolo_input_size) - tfw)/2
        ymin = ((2 * y * yolo_input_size) - tfh)/2
        xmax = xmin + tfw
        ymax = ymin + tfh

        # Unpad
        xmin = xmin - w_pad_left
        ymin = ymin - h_pad_top
        xmax = xmax - w_pad_left
        ymax = ymax - h_pad_top

        # Rescale
        xmin *= scale_orx
        ymin *= scale_ory
        xmax *= scale_orx
        ymax *= scale_ory

        inv_box = (xmin, ymin, xmax, ymax)
        new_bboxes.append(inv_box)

    return new_bboxes

def yolo_to_xml_bbox(bboxes, w, h):
    if bboxes.nelement() == 0:
        return  bboxes
    # x_center, y_center width heigth
    w_half_len = (bboxes[...,2] * w) / 2
    h_half_len = (bboxes[...,3] * h) / 2
    xmin = (bboxes[...,0] * w) - w_half_len
    ymin = (bboxes[...,1] * h) - h_half_len
    xmax = (bboxes[...,0] * w) + w_half_len
    ymax = (bboxes[...,1] * h) + h_half_len
    return torch.cat([xmin.unsqueeze(-1), ymin.unsqueeze(-1), xmax.unsqueeze(-1), ymax.unsqueeze(-1)], dim=1)

def do_voc_test(model):
    # config.DATASET_OLD = config.DATASET
    # config.DATASET = 'DL'
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    model.eval()
    crop_image_trans = Albu.LongestMaxSize(max_size=config.IMAGE_SIZE)
    #  num_classes is 3 because from the frcnn version we also have the __brackground__ class
    frcnn_dataset_name = ""
    # voc_datasets_base_path = "/home/drcat/pycharm_workspace/100DOH_FRCNN_MOD_FGFA_TROIA/data/VOCdevkit2007_handobj_100K/VOC2007"
    voc_datasets_base_path = "./"
    if config.DATASET == "100DOH_DL":
        frcnn_dataset_name = "100K_DL_MOD_BINARY_STATE"
    elif config.DATASET == "100DOH":
        frcnn_dataset_name = "100K_MOD_BINARY_STATE"
    elif config.DATASET == "DL":
        frcnn_dataset_name = "DL_BINARY_VOD"

    frcnn_dataset_path = os.path.join(voc_datasets_base_path, frcnn_dataset_name)
    num_classes = 3

    no_dets = 0
    image_idx = 0

    test_dataset_file = config.DATASET+"/test.csv"

    #  Load test.csv file and iterate through each line
    with open(test_dataset_file) as file:
        test_db = [line.rstrip() for line in file]

    num_images = len(test_db)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    # for idx_im, (x, y, org_x) in enumerate(tqdm(test_loader)):
    loop = tqdm(range(len(test_db)))
    for idx_im in loop:
        load_tic = time.time()
        db_line = test_db[idx_im]
        image_fn = db_line.split(",")[0]
        label_fn = db_line.split(",")[1]
        img_path = config.DATASET+"/images/"+image_fn
        org_image = np.array(Image.open(img_path).convert("RGB"))
        augmentations = config.test_transforms(image=org_image, bboxes=[])
        image = augmentations["image"]

        images = []
        images.append(image.unsqueeze(0))
        if config.DO_VOD:
            if "100DOH" in image_fn:
                for ni in range(0, config.NUM_OF_SUP_FRAMES):
                    images.append(image.unsqueeze(0))
            else:
                file_with_prev_frames_paths_list_path = os.path.join(config.PREV_FRAMES_DIR, os.path.splitext(os.path.basename(image_fn))[0] + ".json")
                with open(file_with_prev_frames_paths_list_path, 'r') as openfile:
                    list_of_support_frames_paths = json.load(openfile)
                if isinstance(list_of_support_frames_paths, list):
                    idx = 0
                    for sup_frame_path in list_of_support_frames_paths:

                        sup_frame_path = sup_frame_path.replace("/media/drcat/Dati/dataset_dl", config.DL_PREV_FRAMES_BASE_PATH)

                        if idx == config.NUM_OF_SUP_FRAMES:
                            break
                        try:
                            sup_frame = np.array(Image.open(sup_frame_path).convert("RGB"))
                        except:
                            print("Failed to load image with path: ", sup_frame_path)
                            sup_frame = None
                        # If sup frame is none the image name could need a zero infront 999867.jpg -> 0999867.jpg

                        if sup_frame is None:
                            sup_frame_im_fn = os.path.basename(sup_frame_path)
                            sup_frame_im_dirname = os.path.dirname(sup_frame_path)
                            new_sup_frame_path = os.path.join(sup_frame_im_dirname, "0" + sup_frame_im_fn)
                            sup_frame = np.array(Image.open(new_sup_frame_path).convert("RGB"))


                        augmentations = config.test_transforms(image=sup_frame, bboxes=[])
                        sup_image = augmentations["image"]
                        images.append(sup_image.unsqueeze(0))
                        idx += 1
        image = images
        x = torch.cat(image, dim=0)
        x = x.to(config.DEVICE)
        load_toc = time.time()
        det_tic = time.time()
        with torch.no_grad():
            # out, _ = model(x)
            out = model(x)
            bboxes = [[] for _ in range(x.shape[0])]
            for i in range(3):
                batch_size, A, S, _, _ = out[i].shape
                anchor = scaled_anchors[i]
                boxes_scale_i = cells_to_bboxes(out[i], anchor, S=S, is_preds=True)
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box

            # model.train()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        # if image_idx == 12:
        #     print("yo")
        misc_tic = time.time()
        # nms_boxes = non_max_suppression_torch_nms_rettorcharr(bboxes[0], iou_threshold=0.5, threshold=0.65, box_format="midpoint",)
        nms_boxes = non_max_suppression_torch_nms_better(bboxes[0], iou_threshold=0.45, threshold=0.05, box_format="midpoint",)


        if nms_boxes.nelement() == 0:
            all_boxes[2][image_idx] = empty_array
            all_boxes[1][image_idx] = empty_array
            image_idx+=1
            no_dets +=1
            continue
        # Create dets for hands
        # Select hands bboxes
        nms_boxes_hands_ids = nms_boxes[:,0] > 0
        nms_boxes_hands = nms_boxes[nms_boxes_hands_ids]
        classes = nms_boxes_hands[:,0]
        lr = torch.where(classes == 1, torch.tensor([[0]]), torch.tensor([[1]])).squeeze(0).unsqueeze(-1).float()
        scores = nms_boxes_hands[:,1].unsqueeze(-1)

        org_image_width = org_image.shape[1]
        org_image_height = org_image.shape[0]

        yolo_cls_boxes = nms_boxes_hands[:,2:6]
        yolo_cls_boxes_list = yolo_cls_boxes.detach().cpu().numpy().tolist()

        tf_image = np.uint8(x[0:1,...].squeeze(0).permute(1, 2, 0).detach().cpu()*255)

        # scaled_cls_boxes = yolo_boxes_unpad_convert_to_xyxy(org_image, tf_image, yolo_cls_boxes_list, crop_image_trans)
        scaled_cls_boxes = yolo_boxes_unpad_convert_to_xyxy_mod(org_image, config.IMAGE_SIZE, yolo_cls_boxes_list)
        scaled_cls_boxes = torch.tensor(scaled_cls_boxes)

        contacts = torch.where(nms_boxes_hands[:,6] > 0.5, torch.tensor([[1]]), torch.tensor([[0]])).squeeze(0).unsqueeze(-1).float()
        vectors = nms_boxes_hands[:,7:]
        # nc_probs is required in the frcnn version but idk why??
        nc_probs = torch.ones_like(contacts)
        # _, order = torch.sort(scores, 0, True)
        hand_dets = torch.cat((scaled_cls_boxes, scores, contacts, vectors, lr, nc_probs), 1).detach().cpu().numpy()
        if hand_dets.size==0:
            hand_dets = None
            all_boxes[2][image_idx] = empty_array
        else:
            all_boxes[2][image_idx] = hand_dets

        # Create dets for targetobjects
        # Select targetobjects bboxes
        nms_boxes_objs_ids = nms_boxes[:, 0] == 0
        nms_boxes_objs = nms_boxes[nms_boxes_objs_ids]
        classes = nms_boxes_objs[:, 0]
        lr = torch.where(classes == 1, torch.tensor([[0]]), torch.tensor([[1]])).squeeze(0).unsqueeze(-1).float()
        scores = nms_boxes_objs[:, 1].unsqueeze(-1)

        yolo_cls_boxes = nms_boxes_objs[:, 2:6]
        yolo_cls_boxes_list = yolo_cls_boxes.detach().cpu().numpy().tolist()

        # scaled_cls_boxes = yolo_boxes_unpad_convert_to_xyxy(org_image, tf_image, yolo_cls_boxes_list, crop_image_trans)
        scaled_cls_boxes = yolo_boxes_unpad_convert_to_xyxy_mod(org_image, config.IMAGE_SIZE, yolo_cls_boxes_list)
        scaled_cls_boxes = torch.tensor(scaled_cls_boxes)

        contacts = torch.where(nms_boxes_objs[:, 6] > 0.5, torch.tensor([[1]]), torch.tensor([[0]])).squeeze(0).unsqueeze(-1).float()
        vectors = nms_boxes_objs[:, 7:]
        # nc_probs is required in the frcnn version but idk why??
        nc_probs = torch.ones_like(contacts)
        # _, order = torch.sort(scores, 0, True)
        obj_dets = torch.cat((scaled_cls_boxes, scores, contacts, vectors, lr, nc_probs), 1).detach().cpu().numpy()
        if obj_dets.size == 0:
            obj_dets = None
            all_boxes[1][image_idx] = empty_array
        else:
            all_boxes[1][image_idx] = obj_dets

        misc_toc = time.time()

        nms_time = misc_toc - misc_tic
        load_time = load_toc - load_tic
        loop.set_postfix(load_time=load_time, det_time=detect_time, misc_time=nms_time)
        # print("DET TIME: ", detect_time)
        # print("MISC TIME: ", nms_time)

        # im2show = vis_detections_filtered_objects_PIL(org_image, obj_dets, hand_dets, 0.1, 0.1)
        # im2show = vis_detections_filtered_objects_PIL(np.uint8(x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255), obj_dets, hand_dets, 0.1, 0.1)
        # im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
        # im2showRGB = cv2.cvtColor(np.array(im2show), cv2.COLOR_RGBA2RGB)
        # cv2.imshow("frame", im2showRGB)
        # cv2.waitKey()
        image_idx +=1


    print("No dets: ", no_dets)
    with open(os.path.join(config.DATASET, 'test.csv'), 'r') as f:
      lines = f.readlines()
    imagenames = [os.path.splitext(x.strip().split(",")[0])[0] for x in lines]
    localdatapath = config.DATASET
    return evaluate_detections(all_boxes, frcnn_dataset_path, localdatapath, imagenames)
    # config.DATASET = config.DATASET_OLD

def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

# def do_voc_test_new(model):
#     # config.DATASET_OLD = config.DATASET
#     # config.DATASET = 'DL'
#     scaled_anchors = (
#         torch.tensor(config.ANCHORS)
#         * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
#     ).to(config.DEVICE)

#     model.eval()
#     crop_image_trans = Albu.LongestMaxSize(max_size=config.IMAGE_SIZE)
#     #  num_classes is 3 because from the frcnn version we also have the __brackground__ class
#     frcnn_dataset_name = ""
#     voc_datasets_base_path = "/home/drcat/pycharm_workspace/100DOH_FRCNN_MOD_FGFA_TROIA/data/VOCdevkit2007_handobj_100K/VOC2007"
#     if config.DATASET == "100DOH_DL":
#         frcnn_dataset_name = "100K_DL_MOD_BINARY_STATE"
#     elif config.DATASET == "100DOH":
#         frcnn_dataset_name = "100K_MOD_BINARY_STATE"
#     elif config.DATASET == "DL":
#         frcnn_dataset_name = "DL_BINARY_VOD"

#     frcnn_dataset_path = os.path.join(voc_datasets_base_path, frcnn_dataset_name)
#     num_classes = 3

#     no_dets = 0
#     image_idx = 0

#     test_dataset_file = config.DATASET+"/test.csv"

#     #  Load test.csv file and iterate through each line
#     with open(test_dataset_file) as file:
#         test_db = [line.rstrip() for line in file]

#     num_images = len(test_db)
#     all_boxes = [[[] for _ in range(num_images)]
#                  for _ in range(num_classes)]

#     empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
#     # for idx_im, (x, y, org_x) in enumerate(tqdm(test_loader)):
#     loop = tqdm(range(len(test_db)))
#     for idx_im in loop:
#         load_tic = time.time()
#         db_line = test_db[idx_im]
#         image_fn = db_line.split(",")[0]
#         label_fn = db_line.split(",")[1]
#         img_path = config.DATASET+"/images/"+image_fn
#         org_image = np.array(Image.open(img_path).convert("RGB"))
#         augmentations = config.test_transforms(image=org_image, bboxes=[])
#         image = augmentations["image"]

#         images = []
#         images.append(image.unsqueeze(0))
#         if config.DO_VOD:
#             if "100DOH" in image_fn:
#                 for ni in range(0, config.NUM_OF_SUP_FRAMES):
#                     images.append(image.unsqueeze(0))
#             else:
#                 file_with_prev_frames_paths_list_path = os.path.join(config.PREV_FRAMES_DIR, os.path.splitext(os.path.basename(image_fn))[0] + ".json")
#                 with open(file_with_prev_frames_paths_list_path, 'r') as openfile:
#                     list_of_support_frames_paths = json.load(openfile)
#                 if isinstance(list_of_support_frames_paths, list):
#                     idx = 0
#                     for sup_frame_path in list_of_support_frames_paths:

#                         sup_frame_path = sup_frame_path.replace("/media/drcat/Dati/dataset_dl", config.DL_PREV_FRAMES_BASE_PATH)

#                         if idx == config.NUM_OF_SUP_FRAMES:
#                             break
#                         try:
#                             sup_frame = np.array(Image.open(sup_frame_path).convert("RGB"))
#                         except:
#                             print("Failed to load image with path: ", sup_frame_path)
#                             sup_frame = None
#                         # If sup frame is none the image name could need a zero infront 999867.jpg -> 0999867.jpg

#                         if sup_frame is None:
#                             sup_frame_im_fn = os.path.basename(sup_frame_path)
#                             sup_frame_im_dirname = os.path.dirname(sup_frame_path)
#                             new_sup_frame_path = os.path.join(sup_frame_im_dirname, "0" + sup_frame_im_fn)
#                             sup_frame = np.array(Image.open(new_sup_frame_path).convert("RGB"))


#                         augmentations = config.test_transforms(image=sup_frame, bboxes=[])
#                         sup_image = augmentations["image"]
#                         images.append(sup_image.unsqueeze(0))
#                         idx += 1
#         image = images
#         x = torch.cat(image, dim=0)
#         x = x.to(config.DEVICE)
#         load_toc = time.time()
#         det_tic = time.time()
#         pred_for_scale = []
#         with torch.no_grad():
#             # out, _ = model(x)
#             out = model(x)

#             for i in range(3):
#                 batch_size, A, S, _, _ = out[i].shape
#                 anchor = scaled_anchors[i]
#                 boxes_scale_i = cells_to_bboxes_new(out[i], anchor, S=S, is_preds=True)
#                 pred_for_scale.append(boxes_scale_i)

#             # model.train()
#         final_preds = torch.cat(pred_for_scale, dim=1)
#         det_toc = time.time()
#         detect_time = det_toc - det_tic
#         # if image_idx == 12:
#         #     print("yo")
#         misc_tic = time.time()
#         nms_boxes = non_max_suppression_new(final_preds, conf_thres=0.9, iou_thres=0.5)[0]


#         if nms_boxes.nelement() == 0:
#             all_boxes[2][image_idx] = empty_array
#             all_boxes[1][image_idx] = empty_array
#             image_idx+=1
#             no_dets +=1
#             continue
#         # Create dets for hands
#         # Select hands bboxes
#         nms_boxes_hands_ids = nms_boxes[:,5] > 0
#         nms_boxes_hands = nms_boxes[nms_boxes_hands_ids]
#         classes = nms_boxes_hands[:,5]
#         lr = torch.where(classes == 1.0, torch.tensor([[0]]), torch.tensor([[1]])).squeeze(0).unsqueeze(-1).float()
#         scores = nms_boxes_hands[:,4].unsqueeze(-1)

#         org_image_width = org_image.shape[1]
#         org_image_height = org_image.shape[0]

#         yolo_cls_boxes = nms_boxes_hands[:,:4]
#         scaled_cls_boxes = rescale_boxes(yolo_cls_boxes, config.IMAGE_SIZE, org_image.shape[:2])
#         # yolo_cls_boxes_list = yolo_cls_boxes.detach().cpu().numpy().tolist()
#         #
#         # tf_image = np.uint8(x[0:1,...].squeeze(0).permute(1, 2, 0).detach().cpu()*255)
#         #
#         # scaled_cls_boxes = yolo_boxes_unpad_convert_to_xyxy(org_image, tf_image, yolo_cls_boxes_list, crop_image_trans)
#         # scaled_cls_boxes = torch.tensor(scaled_cls_boxes)

#         contacts = nms_boxes_hands[:, 6:7].float()
#         vectors = nms_boxes_hands[:,7:]
#         # nc_probs is required in the frcnn version but idk why??
#         nc_probs = torch.ones_like(contacts)
#         # _, order = torch.sort(scores, 0, True)
#         hand_dets = torch.cat((scaled_cls_boxes, scores, contacts, vectors, lr, nc_probs), 1).detach().cpu().numpy()
#         if hand_dets.size==0:
#             hand_dets = None
#             all_boxes[2][image_idx] = empty_array
#         else:
#             all_boxes[2][image_idx] = hand_dets

#         # Create dets for targetobjects
#         # Select targetobjects bboxes
#         nms_boxes_objs_ids = nms_boxes[:, 5] == 0.0
#         nms_boxes_objs = nms_boxes[nms_boxes_objs_ids]
#         classes = nms_boxes_objs[:, 5]
#         lr = torch.where(classes == 1.0, torch.tensor([[0]]), torch.tensor([[1]])).squeeze(0).unsqueeze(-1).float()
#         scores = nms_boxes_objs[:, 4].unsqueeze(-1)

#         org_image_width = org_image.shape[1]
#         org_image_height = org_image.shape[0]

#         yolo_cls_boxes = nms_boxes_objs[:, :4]
#         scaled_cls_boxes = rescale_boxes(yolo_cls_boxes, config.IMAGE_SIZE, org_image.shape[:2])
#         # yolo_cls_boxes_list = yolo_cls_boxes.detach().cpu().numpy().tolist()
#         #
#         # tf_image = np.uint8(x[0:1,...].squeeze(0).permute(1, 2, 0).detach().cpu()*255)
#         #
#         # scaled_cls_boxes = yolo_boxes_unpad_convert_to_xyxy(org_image, tf_image, yolo_cls_boxes_list, crop_image_trans)
#         # scaled_cls_boxes = torch.tensor(scaled_cls_boxes)

#         contacts = nms_boxes_objs[:, 6:7].float()
#         vectors = nms_boxes_objs[:, 7:]
#         # nc_probs is required in the frcnn version but idk why??
#         nc_probs = torch.ones_like(contacts)
#         # _, order = torch.sort(scores, 0, True)
#         obj_dets = torch.cat((scaled_cls_boxes, scores, contacts, vectors, lr, nc_probs), 1).detach().cpu().numpy()
#         if obj_dets.size == 0:
#             obj_dets = None
#             all_boxes[1][image_idx] = empty_array
#         else:
#             all_boxes[1][image_idx] = obj_dets

#         misc_toc = time.time()

#         nms_time = misc_toc - misc_tic
#         load_time = load_toc - load_tic
#         loop.set_postfix(load_time=load_time, det_time=detect_time, misc_time=nms_time)
#         # print("DET TIME: ", detect_time)
#         # print("MISC TIME: ", nms_time)

#         # im2show = vis_detections_filtered_objects_PIL(org_image, obj_dets, hand_dets, 0.1, 0.1)
#         # im2show = vis_detections_filtered_objects_PIL(np.uint8(x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255), obj_dets, hand_dets, 0.1, 0.1)
#         # im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
#         # im2showRGB = cv2.cvtColor(np.array(im2show), cv2.COLOR_RGBA2RGB)
#         # cv2.imshow("frame", im2showRGB)
#         # cv2.waitKey()
#         image_idx +=1


#     print("No dets: ", no_dets)
#     evaluate_detections(all_boxes, frcnn_dataset_path)
#     # config.DATASET = config.DATASET_OLD

# def do_voc_test_online_tsm(model):
#     # config.DATASET_OLD = config.DATASET
#     # config.DATASET = 'DL'
#     scaled_anchors = (
#         torch.tensor(config.ANCHORS)
#         * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
#     ).to(config.DEVICE)

#     model.eval()
#     crop_image_trans = Albu.LongestMaxSize(max_size=config.IMAGE_SIZE)
#     #  num_classes is 3 because from the frcnn version we also have the __brackground__ class
#     frcnn_dataset_name = ""
#     voc_datasets_base_path = "/home/drcat/pycharm_workspace/100DOH_FRCNN_MOD_FGFA_TROIA/data/VOCdevkit2007_handobj_100K/VOC2007"
#     if config.DATASET == "100DOH_DL":
#         frcnn_dataset_name = "100K_DL_MOD_BINARY_STATE"
#     elif config.DATASET == "100DOH":
#         frcnn_dataset_name = "100K_MOD_BINARY_STATE"
#     elif config.DATASET == "DL":
#         frcnn_dataset_name = "DL_BINARY_VOD"

#     frcnn_dataset_path = os.path.join(voc_datasets_base_path, frcnn_dataset_name)
#     num_classes = 3

#     no_dets = 0
#     image_idx = 0

#     test_dataset_file = config.DATASET+"/test.csv"

#     #  Load test.csv file and iterate through each line
#     with open(test_dataset_file) as file:
#         test_db = [line.rstrip() for line in file]

#     num_images = len(test_db)
#     all_boxes = [[[] for _ in range(num_images)]
#                  for _ in range(num_classes)]

#     empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
#     # for idx_im, (x, y, org_x) in enumerate(tqdm(test_loader)):
#     loop = tqdm(range(len(test_db)))
#     for idx_im in loop:
#         load_tic = time.time()
#         db_line = test_db[idx_im]
#         image_fn = db_line.split(",")[0]
#         label_fn = db_line.split(",")[1]
#         img_path = config.DATASET+"/images/"+image_fn
#         org_image = np.array(Image.open(img_path).convert("RGB"))
#         augmentations = config.test_transforms(image=org_image, bboxes=[])
#         image = augmentations["image"]

#         images = []
#         images.append(image.unsqueeze(0))

#         # sup_frames_to_load = config.NUM_OF_SUP_FRAMES
#         sup_frames_to_load = 4
#         dt_100doh = False

#         if config.DO_VOD:
#             if "100DOH" in image_fn:
#                 dt_100doh = True
#                 for ni in range(0, 2):
#                     images.append(image.unsqueeze(0))
#             else:
#                 file_with_prev_frames_paths_list_path = os.path.join(config.PREV_FRAMES_DIR, os.path.splitext(os.path.basename(image_fn))[0] + ".json")
#                 with open(file_with_prev_frames_paths_list_path, 'r') as openfile:
#                     list_of_support_frames_paths = json.load(openfile)
#                 if isinstance(list_of_support_frames_paths, list):
#                     idx = 0
#                     for sup_frame_path in list_of_support_frames_paths:

#                         sup_frame_path = sup_frame_path.replace("/media/drcat/Dati/dataset_dl", config.DL_PREV_FRAMES_BASE_PATH)

#                         if idx == sup_frames_to_load:
#                             break
#                         try:
#                             sup_frame = np.array(Image.open(sup_frame_path).convert("RGB"))
#                         except:
#                             print("Failed to load image with path: ", sup_frame_path)
#                             sup_frame = None
#                         # If sup frame is none the image name could need a zero infront 999867.jpg -> 0999867.jpg

#                         if sup_frame is None:
#                             sup_frame_im_fn = os.path.basename(sup_frame_path)
#                             sup_frame_im_dirname = os.path.dirname(sup_frame_path)
#                             new_sup_frame_path = os.path.join(sup_frame_im_dirname, "0" + sup_frame_im_fn)
#                             sup_frame = np.array(Image.open(new_sup_frame_path).convert("RGB"))


#                         augmentations = config.test_transforms(image=sup_frame, bboxes=[])
#                         sup_image = augmentations["image"]
#                         images.append(sup_image.unsqueeze(0))
#                         idx += 1
#         x = [image.to(config.DEVICE) for image in images]

#         # x = torch.cat(image, dim=0)
#         # x = x.to(config.DEVICE)
#         load_toc = time.time()
#         det_tic = time.time()
#         with torch.no_grad():
#             if not dt_100doh:
#                 multiplier = 2
#                 init_input_shift_buffer = [[torch.zeros((1, 4*multiplier, config.IMAGE_SIZE // 2, config.IMAGE_SIZE // 2)).to(config.DEVICE)],

#                                            [torch.zeros((1, 8*multiplier, config.IMAGE_SIZE // 4, config.IMAGE_SIZE // 4)).to(config.DEVICE),
#                                             torch.zeros((1, 8*multiplier, config.IMAGE_SIZE // 4, config.IMAGE_SIZE // 4)).to(config.DEVICE)],

#                                            [torch.zeros((1, 16*multiplier, config.IMAGE_SIZE // 8, config.IMAGE_SIZE // 8)).to(config.DEVICE),
#                                             torch.zeros((1, 16*multiplier, config.IMAGE_SIZE // 8, config.IMAGE_SIZE // 8)).to(config.DEVICE),
#                                             torch.zeros((1, 16*multiplier, config.IMAGE_SIZE // 8, config.IMAGE_SIZE // 8)).to(config.DEVICE),
#                                             torch.zeros((1, 16*multiplier, config.IMAGE_SIZE // 8, config.IMAGE_SIZE // 8)).to(config.DEVICE),
#                                             torch.zeros((1, 16*multiplier, config.IMAGE_SIZE // 8, config.IMAGE_SIZE // 8)).to(config.DEVICE),
#                                             torch.zeros((1, 16*multiplier, config.IMAGE_SIZE // 8, config.IMAGE_SIZE // 8)).to(config.DEVICE),
#                                             torch.zeros((1, 16*multiplier, config.IMAGE_SIZE // 8, config.IMAGE_SIZE // 8)).to(config.DEVICE),
#                                             torch.zeros((1, 16*multiplier, config.IMAGE_SIZE // 8, config.IMAGE_SIZE // 8)).to(config.DEVICE), ],

#                                            [torch.zeros((1, 32*multiplier, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 16)).to(config.DEVICE),
#                                             torch.zeros((1, 32*multiplier, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 16)).to(config.DEVICE),
#                                             torch.zeros((1, 32*multiplier, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 16)).to(config.DEVICE),
#                                             torch.zeros((1, 32*multiplier, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 16)).to(config.DEVICE),
#                                             torch.zeros((1, 32*multiplier, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 16)).to(config.DEVICE),
#                                             torch.zeros((1, 32*multiplier, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 16)).to(config.DEVICE),
#                                             torch.zeros((1, 32*multiplier, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 16)).to(config.DEVICE),
#                                             torch.zeros((1, 32*multiplier, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 16)).to(config.DEVICE), ],

#                                            [torch.zeros((1, 64*multiplier, config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 32)).to(config.DEVICE),
#                                             torch.zeros((1, 64*multiplier, config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 32)).to(config.DEVICE),
#                                             torch.zeros((1, 64*multiplier, config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 32)).to(config.DEVICE),
#                                             torch.zeros((1, 64*multiplier, config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 32)).to(config.DEVICE), ]]

#                 # out, _ = model(x)
#                 out_shift_buffer = init_input_shift_buffer
#                 for i in reversed(range(1, sup_frames_to_load+1)):
#                     out, out_shift_buffer = model(x[i], out_shift_buffer)
#                 out, _ = model(x[0], out_shift_buffer)
#                 x = x[0]
#             else:
#                 x = torch.cat(x, dim=0)
#                 out = model(x)
#                 x = x[0:1,...]
#             bboxes = [[] for _ in range(x.shape[0])]
#             for i in range(3):
#                 batch_size, A, S, _, _ = out[i].shape
#                 anchor = scaled_anchors[i]
#                 boxes_scale_i = cells_to_bboxes(out[i], anchor, S=S, is_preds=True)
#                 for idx, (box) in enumerate(boxes_scale_i):
#                     bboxes[idx] += box

#             # model.train()
#         det_toc = time.time()
#         detect_time = det_toc - det_tic
#         # if image_idx == 12:
#         #     print("yo")
#         misc_tic = time.time()
#         nms_boxes = non_max_suppression_torch_nms_rettorcharr(bboxes[0], iou_threshold=0.5, threshold=0.65, box_format="midpoint",)


#         if nms_boxes.nelement() == 0:
#             all_boxes[2][image_idx] = empty_array
#             all_boxes[1][image_idx] = empty_array
#             image_idx+=1
#             no_dets +=1
#             continue
#         # Create dets for hands
#         # Select hands bboxes
#         nms_boxes_hands_ids = nms_boxes[:,0] > 0
#         nms_boxes_hands = nms_boxes[nms_boxes_hands_ids]
#         classes = nms_boxes_hands[:,0]
#         lr = torch.where(classes == 1, torch.tensor([[0]]), torch.tensor([[1]])).squeeze(0).unsqueeze(-1).float()
#         scores = nms_boxes_hands[:,1].unsqueeze(-1)

#         org_image_width = org_image.shape[1]
#         org_image_height = org_image.shape[0]

#         yolo_cls_boxes = nms_boxes_hands[:,2:6]
#         yolo_cls_boxes_list = yolo_cls_boxes.detach().cpu().numpy().tolist()

#         tf_image = np.uint8(x[0:1,...].squeeze(0).permute(1, 2, 0).detach().cpu()*255)

#         scaled_cls_boxes = yolo_boxes_unpad_convert_to_xyxy(org_image, tf_image, yolo_cls_boxes_list, crop_image_trans)
#         scaled_cls_boxes = torch.tensor(scaled_cls_boxes)

#         contacts = torch.where(nms_boxes_hands[:,6] > 0.5, torch.tensor([[1]]), torch.tensor([[0]])).squeeze(0).unsqueeze(-1).float()
#         vectors = nms_boxes_hands[:,7:]
#         # nc_probs is required in the frcnn version but idk why??
#         nc_probs = torch.ones_like(contacts)
#         # _, order = torch.sort(scores, 0, True)
#         hand_dets = torch.cat((scaled_cls_boxes, scores, contacts, vectors, lr, nc_probs), 1).detach().cpu().numpy()
#         if hand_dets.size==0:
#             hand_dets = None
#             all_boxes[2][image_idx] = empty_array
#         else:
#             all_boxes[2][image_idx] = hand_dets

#         # Create dets for targetobjects
#         # Select targetobjects bboxes
#         nms_boxes_objs_ids = nms_boxes[:, 0] == 0
#         nms_boxes_objs = nms_boxes[nms_boxes_objs_ids]
#         classes = nms_boxes_objs[:, 0]
#         lr = torch.where(classes == 1, torch.tensor([[0]]), torch.tensor([[1]])).squeeze(0).unsqueeze(-1).float()
#         scores = nms_boxes_objs[:, 1].unsqueeze(-1)

#         yolo_cls_boxes = nms_boxes_objs[:, 2:6]
#         yolo_cls_boxes_list = yolo_cls_boxes.detach().cpu().numpy().tolist()

#         scaled_cls_boxes = yolo_boxes_unpad_convert_to_xyxy(org_image, tf_image, yolo_cls_boxes_list, crop_image_trans)
#         scaled_cls_boxes = torch.tensor(scaled_cls_boxes)

#         contacts = torch.where(nms_boxes_objs[:, 6] > 0.5, torch.tensor([[1]]), torch.tensor([[0]])).squeeze(0).unsqueeze(-1).float()
#         vectors = nms_boxes_objs[:, 7:]
#         # nc_probs is required in the frcnn version but idk why??
#         nc_probs = torch.ones_like(contacts)
#         # _, order = torch.sort(scores, 0, True)
#         obj_dets = torch.cat((scaled_cls_boxes, scores, contacts, vectors, lr, nc_probs), 1).detach().cpu().numpy()
#         if obj_dets.size == 0:
#             obj_dets = None
#             all_boxes[1][image_idx] = empty_array
#         else:
#             all_boxes[1][image_idx] = obj_dets

#         misc_toc = time.time()

#         nms_time = misc_toc - misc_tic
#         load_time = load_toc - load_tic
#         loop.set_postfix(load_time=load_time, det_time=detect_time, misc_time=nms_time)
#         # print("DET TIME: ", detect_time)
#         # print("MISC TIME: ", nms_time)

#         # im2show = vis_detections_filtered_objects_PIL(org_image, obj_dets, hand_dets, 0.1, 0.1)
#         # im2show = vis_detections_filtered_objects_PIL(np.uint8(x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255), obj_dets, hand_dets, 0.1, 0.1)
#         # im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
#         # im2showRGB = cv2.cvtColor(np.array(im2show), cv2.COLOR_RGBA2RGB)
#         # cv2.imshow("frame", im2showRGB)
#         # cv2.waitKey()
#         image_idx +=1


#     print("No dets: ", no_dets)
#     with open(os.path.join(config.DATASET, 'test.csv'), 'r') as f:
#         lines = f.readlines()
#     imagenames = [os.path.splitext(x.strip().split(",")[0])[0] for x in lines]
#     localdatapath = config.DATASET
#     evaluate_detections(all_boxes, frcnn_dataset_path, localdatapath, imagenames)
#     config.DATASET = config.DATASET_OLD

def resize_optical_flow(flow, new_shape=(1,2,76,76)):
    """
    Resize the optical flow tensor to the given shape.

    Parameters:
    - flow (torch.Tensor): The optical flow tensor of shape (B, 2, H, W).
    - new_shape (tuple): The target shape as (B, 2, new_H, new_W).

    Returns:
    - torch.Tensor: The resized optical flow tensor.
    """
    # Calculate the scale factors for height and width
    height_scale_factor = float(new_shape[2]) / float(flow.shape[2])
    width_scale_factor = float(new_shape[3]) / float(flow.shape[3])

    # Resize the optical flow
    resized_flow = torch.nn.functional.interpolate(flow, size=(new_shape[2], new_shape[3]), mode='bilinear', align_corners=False)

    # Adjust the magnitude of the flow vectors
    resized_flow[:, 0, :, :] *= height_scale_factor
    resized_flow[:, 1, :, :] *= width_scale_factor

    return resized_flow

def compress_and_save_npz(flow_data, file_path):
    """
    Compress and save the optical flow data using numpy's .npz format.

    Parameters:
    - flow_data (numpy array): The optical flow data.
    - file_path (str): Path to save the compressed data.

    Returns:
    - None
    """
    np.savez_compressed(file_path, optical_flow=flow_data)


def decompress_and_load_npz(file_path):
    """
    Decompress and load the optical flow data from a .npz file.

    Parameters:
    - file_path (str): Path to the compressed optical flow data.

    Returns:
    - numpy array: The decompressed optical flow data.
    """
    with np.load(file_path) as data:
        flow_data = data['optical_flow']
    return flow_data

def do_voc_test_input(model, img_path):

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    model.eval()
    crop_image_trans = Albu.LongestMaxSize(max_size=config.IMAGE_SIZE)

    num_classes = 3

    no_dets = 0
    image_idx = 0

    # for idx_im, (x, y, org_x) in enumerate(tqdm(test_loader)):
    # for idx_im in loop:
    # img_path = config.DATASET+"/images/"+image_fn
    org_image = np.array(Image.open(img_path).convert("RGB"))
    augmentations = config.test_transforms(image=org_image, bboxes=[])
    image = augmentations["image"]
    images = []
    images.append(image.unsqueeze(0))


    supp_frames_paths = ["/home/drcat/pycharm_workspace/100doh_yolov3_simple/test_ims/f0.png", "/home/drcat/pycharm_workspace/100doh_yolov3_simple/test_ims/f0.png"]
    # supp_frames_paths = ["/media/drcat/Dati/dataset_dl/Out/Check1/Check1Basket1/Check1Beh1Basket1/Check1Beh1Basket1Pass1/RightSideSequences/1/999625.jpg"]
    flows = []
    # flow_module = OpticalFlowModule('lightflownet2')
    for supp_frames_path in supp_frames_paths:
        sup_image = np.array(Image.open(supp_frames_path).convert("RGB"))

        # flow_numpy = flow_module.compute_flow(sup_image, org_image)

        # flow_torch = torch.from_numpy(flow_numpy).unsqueeze(0).permute(0, 3, 1, 2)

        # flow_numpy_resized_76 = resize_optical_flow(flow_torch, (1, 2, 76, 76)).squeeze(0).permute(1, 2, 0).numpy()
        # flow_numpy_resized_52 = resize_optical_flow(flow_torch, (1, 2, 52, 52)).squeeze(0).permute(1, 2, 0).numpy()
        # flow_numpy_resized_38 = resize_optical_flow(flow_torch, (1, 2, 38, 38)).squeeze(0).permute(1, 2, 0).numpy()
        # flow_numpy_resized_26 = resize_optical_flow(flow_torch, (1, 2, 26, 26)).squeeze(0).permute(1, 2, 0).numpy()
        # flow_numpy_resized_19 = resize_optical_flow(flow_torch, (1, 2, 19, 19)).squeeze(0).permute(1, 2, 0).numpy()
        # flow_numpy_resized_13 = resize_optical_flow(flow_torch, (1, 2, 13, 13)).squeeze(0).permute(1, 2, 0).numpy()

        # flow_numpy_resized_52_vis = flow_to_image(flow_numpy_resized_52)

        # compress_and_save_npz(flow_numpy_resized_76, "f2_b2f3_b_76.flow")
        # compress_and_save_npz(flow_numpy_resized_52, "f02f1_52.flow")
        # compress_and_save_npz(flow_numpy_resized_38, "f2_b2f3_b_76.flow")
        # compress_and_save_npz(flow_numpy_resized_26, "f02f1_26.flow")
        # compress_and_save_npz(flow_numpy_resized_19, "f2_b2f3_b_76.flow")
        # compress_and_save_npz(flow_numpy_resized_13, "f02f1_13.flow")

    #     prev_pts, next_pts = compute_sparse_optical_flow(sup_image, org_image)
    #     sup_flow = interpolate_dense_flow_rbf(prev_pts, next_pts, org_image.shape[0], org_image.shape[1])
    #     # sup_flow = compute_optical_flow(org_image, sup_image)
    #     # sup_flow_viz = visualize_sparse_flow_both(sup_image, org_image, prev_pts, next_pts)
    #     sup_flow_viz = flow_to_image(sup_flow)
    #     sup_flow_viz_Arr = put_optical_flow_arrows_on_image(sup_image, sup_flow)
    #     # flows.append(sup_flow)
        augmentations = config.test_transforms(image=sup_image, bboxes=[])
        image = augmentations["image"]
        images.append(image.unsqueeze(0))
    image = images
    x = torch.cat(image, dim=0)
    x = x.to(config.DEVICE)
    load_toc = time.time()
    det_tic = time.time()
    # Assuming you have a pre-loaded YOLOv3 model named 'yolov3_model'
    # target_layer = model.layers[14] # Adjust based on your YOLOv3 architecture
    # grad_cam = GradCAM(model, target_layer)
    # For a given input image tensor 'input_image'
    # heatmap = grad_cam.compute_heatmap(x)
    # overlayed_image = grad_cam.overlay_heatmap(heatmap, org_image)  # 'original_image' should be in BGR format
    with torch.no_grad():
        out, _ = model(x)
        # out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = scaled_anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # model.train()
    det_toc = time.time()
    detect_time = det_toc - det_tic
    # if image_idx == 12:
    #     print("yo")
    misc_tic = time.time()
    nms_boxes = non_max_suppression_torch_nms_rettorcharr(bboxes[0], iou_threshold=0.5, threshold=0.65, box_format="midpoint",)


    if nms_boxes.nelement() == 0:
        return
    # Create dets for hands
    # Select hands bboxes
    nms_boxes_hands_ids = nms_boxes[:,0] > 0
    nms_boxes_hands = nms_boxes[nms_boxes_hands_ids]
    classes = nms_boxes_hands[:,0]
    lr = torch.where(classes == 1, torch.tensor([[0]]), torch.tensor([[1]])).squeeze(0).unsqueeze(-1).float()
    scores = nms_boxes_hands[:,1].unsqueeze(-1)

    org_image_width = org_image.shape[1]
    org_image_height = org_image.shape[0]

    yolo_cls_boxes = nms_boxes_hands[:,2:6]
    yolo_cls_boxes_list = yolo_cls_boxes.detach().cpu().numpy().tolist()

    tf_image = np.uint8(x[0:1,...].squeeze(0).permute(1, 2, 0).detach().cpu()*255)

    scaled_cls_boxes = yolo_boxes_unpad_convert_to_xyxy(org_image, tf_image, yolo_cls_boxes_list, crop_image_trans)
    scaled_cls_boxes = torch.tensor(scaled_cls_boxes)

    contacts = torch.where(nms_boxes_hands[:,6] > 0.5, torch.tensor([[1]]), torch.tensor([[0]])).squeeze(0).unsqueeze(-1).float()
    vectors = nms_boxes_hands[:,7:]
    # nc_probs is required in the frcnn version but idk why??
    nc_probs = torch.ones_like(contacts)
    # _, order = torch.sort(scores, 0, True)
    hand_dets = torch.cat((scaled_cls_boxes, scores, contacts, vectors, lr, nc_probs), 1).detach().cpu().numpy()

    # Create dets for targetobjects
    # Select targetobjects bboxes
    nms_boxes_objs_ids = nms_boxes[:, 0] == 0
    nms_boxes_objs = nms_boxes[nms_boxes_objs_ids]
    classes = nms_boxes_objs[:, 0]
    lr = torch.where(classes == 1, torch.tensor([[0]]), torch.tensor([[1]])).squeeze(0).unsqueeze(-1).float()
    scores = nms_boxes_objs[:, 1].unsqueeze(-1)

    yolo_cls_boxes = nms_boxes_objs[:, 2:6]
    yolo_cls_boxes_list = yolo_cls_boxes.detach().cpu().numpy().tolist()

    scaled_cls_boxes = yolo_boxes_unpad_convert_to_xyxy(org_image, tf_image, yolo_cls_boxes_list, crop_image_trans)
    scaled_cls_boxes = torch.tensor(scaled_cls_boxes)

    contacts = torch.where(nms_boxes_objs[:, 6] > 0.5, torch.tensor([[1]]), torch.tensor([[0]])).squeeze(0).unsqueeze(-1).float()
    vectors = nms_boxes_objs[:, 7:]
    # nc_probs is required in the frcnn version but idk why??
    nc_probs = torch.ones_like(contacts)
    # _, order = torch.sort(scores, 0, True)
    obj_dets = torch.cat((scaled_cls_boxes, scores, contacts, vectors, lr, nc_probs), 1).detach().cpu().numpy()

    misc_toc = time.time()

    # nms_time = misc_toc - misc_tic
    # load_time = load_toc - load_tic

    # print("DET TIME: ", detect_time)
    # print("MISC TIME: ", nms_time)

    im2show = vis_detections_filtered_objects_PIL(org_image, obj_dets, hand_dets, 0.1, 0.1)
    # im2show = vis_detections_filtered_objects_PIL(np.uint8(x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255), obj_dets, hand_dets, 0.1, 0.1)
    # im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
    im2showRGB = cv2.cvtColor(np.array(im2show), cv2.COLOR_RGBA2RGB)
    cv2.imshow("frame", im2showRGB)
    cv2.waitKey()
    image_idx +=1

if __name__ == "__main__":
    print("a")
    # model = YOLOv3VOD(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    # optimizer = optim.Adam(
    #     model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    # )
    # optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)

    # load_checkpoint(
    #     config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
    # )

    # do_voc_test(model)