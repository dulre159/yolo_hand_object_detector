import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
import torchvision
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import time

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2, contact_score, magnitude_pred, unitdx_pred, unitdy_pred]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        # bboxes_tmp = []
        # for box in bboxes:
        #     if box[0] != chosen_box[0] or intersection_over_union(torch.tensor(chosen_box[2:6]), torch.tensor(box[2:6]), box_format=box_format,) < iou_threshold:
        #         bboxes_tmp.append(box)
        # bboxes = bboxes_tmp
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:6]),
                torch.tensor(box[2:6]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def non_max_suppression_mod(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2, contact_score, magnitude_pred, unitdx_pred, unitdy_pred]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    # Filter out boxes below the score threshold
    bboxes = [box for box in bboxes if box[1] > threshold]

    # Sort boxes by score in descending order
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    # Set to keep track of indices to remove
    boxes_to_remove = set()

    for i, box in enumerate(bboxes):
        if i in boxes_to_remove:
            continue

        bboxes_after_nms.append(box)

        for j in range(i + 1, len(bboxes)):
            if j in boxes_to_remove:
                continue

            if box[0] != bboxes[j][0]:
                iou = intersection_over_union(
                    torch.tensor(box[2:6]),
                    torch.tensor(bboxes[j][2:6]),
                    box_format=box_format,
                )

                if iou > iou_threshold:
                    boxes_to_remove.add(j)

    # Filter out boxes marked for removal
    bboxes_after_nms = [box for i, box in enumerate(bboxes_after_nms) if i not in boxes_to_remove]

    return bboxes_after_nms


import torch
import torchvision.ops as ops


def non_max_suppression_torch_nms(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2, contact_score, magnitude_pred, unitdx_pred, unitdy_pred]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]

    if len(bboxes) == 0:
        return []

    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes = torch.tensor(bboxes)

    if box_format == "midpoint":
        x1 = bboxes[..., 2] - bboxes[..., 4] / 2
        y1 = bboxes[..., 3] - bboxes[..., 5] / 2
        x2 = bboxes[..., 2] + bboxes[..., 4] / 2
        y2 = bboxes[..., 3] + bboxes[..., 5] / 2

        boxes_coords = torch.stack([x1, y1, x2, y2], dim=1)
    if box_format == "corners":
        boxes_coords = bboxes[:, 2:6]

    # max_wh = 7680
    # c = bboxes[..., 0:1] * max_wh
    # boxes_coords = boxes_coords+c
    keep = ops.nms(
        boxes_coords, bboxes[:, 1], iou_threshold
    )

    bboxes_after_nms = bboxes[keep].tolist()

    return bboxes_after_nms

def non_max_suppression_torch_nms_rettorcharr(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2, contact_score, magnitude_pred, unitdx_pred, unitdy_pred]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]

    if len(bboxes) == 0:
        return torch.tensor([])

    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes = torch.tensor(bboxes)

    if box_format == "midpoint":
        x1 = bboxes[..., 2] - bboxes[..., 4] / 2
        y1 = bboxes[..., 3] - bboxes[..., 5] / 2
        x2 = bboxes[..., 2] + bboxes[..., 4] / 2
        y2 = bboxes[..., 3] + bboxes[..., 5] / 2

        boxes_coords = torch.stack([x1, y1, x2, y2], dim=1)
    if box_format == "corners":
        boxes_coords = bboxes[:, 2:6]

    # max_wh = 7680
    # c = bboxes[..., 0:1] * max_wh
    # boxes_coords = boxes_coords+c
    keep = ops.nms(
        boxes_coords, bboxes[:, 1], iou_threshold
    )

    bboxes_after_nms = bboxes[keep]

    return bboxes_after_nms

def non_max_suppression_torch_nms_better(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2, contact_score, magnitude_pred, unitdx_pred, unitdy_pred]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]

    if len(bboxes) == 0:
        return torch.tensor([])

    # bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes = torch.tensor(bboxes, device=torch.device('cuda:0'))

    if box_format == "midpoint":
        x1 = bboxes[..., 2] - bboxes[..., 4] / 2
        y1 = bboxes[..., 3] - bboxes[..., 5] / 2
        x2 = bboxes[..., 2] + bboxes[..., 4] / 2
        y2 = bboxes[..., 3] + bboxes[..., 5] / 2

        boxes_coords = torch.stack([x1, y1, x2, y2], dim=1)
    if box_format == "corners":
        boxes_coords = bboxes[:, 2:6]

    scores = bboxes[:, 1]
    classes = bboxes[:, 0].unsqueeze(1)

    # max_wh = 7680
    # c = bboxes[..., 0:1] * max_wh
    # boxes_coords = boxes_coords+c
    keep = ops.nms(boxes_coords + classes, scores, iou_threshold)

    bboxes_after_nms = bboxes[keep]

    return bboxes_after_nms.detach().cpu()

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
def to_cpu(tensor):
    return tensor.detach().cpu()
def non_max_suppression_new(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx10 (x1, y1, x2, y2, conf, class, contact_state, asso_vec)
    """

    nc = 3  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 10), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 0] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:8] *= x[:, 0:1]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, 1:5])
        # Per la contact la sigmoid e > 0.5 è già applicato prima
        cs = x[:, 8:9]
        assvs = x[:, 9:]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:8] > conf_thres).nonzero(as_tuple=False).T
            # Nuova forma di x (x1, y1, x2, y2, conf, class, contact_state, asso_vec)
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float(), cs[i], assvs[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, 1:5] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = to_cpu(x[i])

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=7
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:7]),
                    torch.tensor(gt[3:7]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        ap = torch.trapz(precisions, recalls)
        average_precisions.append(ap)
        print("MAP CLASS ", c, ap)


    return sum(average_precisions) / len(average_precisions)

def bgr_to_matplt_rgba(bgr_color_tuple):
    rgb_color = tuple(reversed(bgr_color_tuple))
    rgba_color = (rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0, 1.0)
    return rgba_color

# def plot_image_detect(image, boxes, org_im_width, org_im_height):
def plot_image_detect(image, boxes):
    """Plots predicted bounding boxes on the image"""
    # cmap = plt.get_cmap("tab20b")
    class_labels = config.LABELS_100DOH
    colors = config.LABELS_100DOH_COLORS
    line_color = (255,0,0)

    im = np.array(image)
    height, width, _ = im.shape
    # org_im_width, org_im_height = org_im_width, org_im_height

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 10, "box should contain class pred, confidence, x, y, width, height, contact, mag, unitx, unity"
        class_pred = box[0]
        box_coord = box[2:6]
        upper_left_x = box_coord[0] - box_coord[2] / 2
        upper_left_y = box_coord[1] - box_coord[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box_coord[2] * width,
            box_coord[3] * height,
            linewidth=1,
            edgecolor=bgr_to_matplt_rgba(colors[int(class_pred)]),
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        if class_pred > 0:
            contact = box[6]
            plt.text(
                upper_left_x * width,
                upper_left_y * height,
                s=class_labels[int(class_pred)] + "_ic" if contact > 0.5 else class_labels[int(class_pred)] ,
                color="white",
                verticalalignment="top",
                bbox={"color": bgr_to_matplt_rgba(colors[int(class_pred)]), "pad": 0},
            )
        else:
            plt.text(
                upper_left_x * width,
                upper_left_y * height,
                s=class_labels[int(class_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": bgr_to_matplt_rgba(colors[int(class_pred)]), "pad": 0},
            )

        # Add association arrows
        x, y, w, h = box_coord
        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)
        x_c = int((x1 + x2) / 2)
        y_c = int((y1 + y2) / 2)
        magnitude, unitdx, unitdy = map(float, box[7:])
        contact = box[6]
        if class_pred > 0 and contact > 0.5 and magnitude > 0:
            # Rescale magnitude
            magnitude/=0.001
            # Rescale unit vector
            unitdx/=0.1
            unitdy/=0.1
            # magnitude = magnitude * (width/org_im_width)
            dx = int(unitdx * magnitude)
            dy = int(unitdy * magnitude)
            # line_color = colors[(line_idx + 1) % len(colors)]  # use a different color for each line
            p0 = (x_c, y_c)
            p1 = (x_c + dx, y_c + dy)
            plt.arrow(p0[0], p0[1], p1[0] - p0[0] , p1[1] - p0[1], head_width=0.05, head_length=0.1, length_includes_head=True, color=bgr_to_matplt_rgba(line_color), linestyle="--")

    plt.show()

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    # cmap = plt.get_cmap("tab20b")
    class_labels = config.LABELS_100DOH
    colors = config.LABELS_100DOH_COLORS
    line_color = (255,0,0)

    im = np.array(image)
    height, width, _ = im.shape
    # org_im_height, org_im_width = org_im_shape[0].detach().cpu().numpy()[0], org_im_shape[1].detach().cpu().numpy()[0]

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 10, "box should contain class pred, confidence, x, y, width, height, contact, mag, unitx, unity"
        class_pred = box[0]
        box_coord = box[2:6]
        upper_left_x = box_coord[0] - box_coord[2] / 2
        upper_left_y = box_coord[1] - box_coord[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box_coord[2] * width,
            box_coord[3] * height,
            linewidth=2,
            edgecolor=bgr_to_matplt_rgba(colors[int(class_pred)]),
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        if class_pred > 0:
            contact = box[6]
            plt.text(
                upper_left_x * width,
                upper_left_y * height,
                s=class_labels[int(class_pred)] + "_ic" if contact > 0.5 else class_labels[int(class_pred)] ,
                color="white",
                verticalalignment="top",
                bbox={"color": bgr_to_matplt_rgba(colors[int(class_pred)]), "pad": 0},
            )
        else:
            plt.text(
                upper_left_x * width,
                upper_left_y * height,
                s=class_labels[int(class_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": bgr_to_matplt_rgba(colors[int(class_pred)]), "pad": 0},
            )

        # Add association arrows
        x, y, w, h = box_coord
        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)
        x_c = int((x1 + x2) / 2)
        y_c = int((y1 + y2) / 2)
        magnitude, unitdx, unitdy = map(float, box[7:])
        contact = box[6]
        if class_pred > 0 and contact > 0.5 and magnitude > 0:
            # Rescale magnitude
            magnitude/=0.001
            # Rescale unit vector
            unitdx/=0.1
            unitdy/=0.1
            # magnitude = magnitude * (width/org_im_width)
            dx = int(unitdx * magnitude)
            dy = int(unitdy * magnitude)
            # line_color = colors[(line_idx + 1) % len(colors)]  # use a different color for each line
            p0 = (x_c, y_c)
            p1 = (x_c + dx, y_c + dy)
            plt.arrow(p0[0], p0[1], p1[0] - p0[0] , p1[1] - p0[1], head_width=0.05, head_length=0.1, length_includes_head=True, color=bgr_to_matplt_rgba(line_color), linestyle="--")

    plt.show()


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        if config.DO_VOD:
            x = torch.cat(x, dim=0)
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression_torch_nms(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        #box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        #box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])*2 -0.5
        box_predictions[..., 2:] = ((torch.sigmoid(box_predictions[..., 2:])*2)**2) * anchors
        
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:8], dim=-1).unsqueeze(-1)
        contacts = torch.sigmoid(predictions[..., 8:9])
        # mags = predictions[..., 9:10]
        # vec_xs = predictions[..., 10:11]
        # vec_ys = predictions[..., 11:12]
        magnitudedxdy_pred = predictions[...,9:]
        dxdymagnitude_pred_sub = 0.1 * F.normalize(magnitudedxdy_pred[...,1:], p=2, dim=4)
        dxdymagnitude_pred_norm = torch.cat([predictions[...,9].unsqueeze(-1), dxdymagnitude_pred_sub], dim=4)
        magnitudedxdy = dxdymagnitude_pred_norm

    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]
        contacts = predictions[..., 6:7]
        magnitudedxdy = predictions[...,7:]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    # converted_bboxes = torch.cat((best_class, scores, x, y, w_h, contacts, mags, vec_xs, vec_ys), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 10)
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h, contacts, magnitudedxdy), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 10)
    return converted_bboxes.tolist()

def cells_to_bboxes_new(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        # best_class = torch.argmax(predictions[..., 5:8], dim=-1).unsqueeze(-1)
        class_confs = torch.sigmoid(predictions[..., 5:8])
        contacts = torch.sigmoid(predictions[..., 8:9]) > 0.5
        # mags = predictions[..., 9:10]
        # vec_xs = predictions[..., 10:11]
        # vec_ys = predictions[..., 11:12]
        magnitudedxdy_pred = predictions[...,9:]
        dxdymagnitude_pred_sub = 0.1 * F.normalize(magnitudedxdy_pred[...,1:], p=2, dim=4)
        dxdymagnitude_pred_norm = torch.cat([predictions[...,9].unsqueeze(-1), dxdymagnitude_pred_sub], dim=4)
        magnitudedxdy = dxdymagnitude_pred_norm

    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]
        contacts = predictions[..., 6:7]
        magnitudedxdy = predictions[...,7:]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    # converted_bboxes = torch.cat((best_class, scores, x, y, w_h, contacts, mags, vec_xs, vec_ys), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 10)
    # converted_bboxes = torch.cat((scores, x, y, w_h,best_class, contacts, magnitudedxdy), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 10)
    converted_bboxes = torch.cat((scores, x, y, w_h, class_confs, contacts, magnitudedxdy), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 12)
    return converted_bboxes

def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        if config.DO_VOD:
            x = torch.cat(x, dim=0)
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:8][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print("\n")
    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    print("\n")
    model.train()



    def load_darknet_weights(model, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            # self.header_info = header  # Needed to write header when saving weights
            # self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        # ptr = 0
        # for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
        #     if i == cutoff:
        #         break
        #     if module_def["type"] == "convolutional":
        #         conv_layer = module[0]
        #         if module_def["batch_normalize"]:
        #             # Load BN bias, weights, running mean and running variance
        #             bn_layer = module[1]
        #             num_b = bn_layer.bias.numel()  # Number of biases
        #             # Bias
        #             bn_b = torch.from_numpy(
        #                 weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
        #             bn_layer.bias.data.copy_(bn_b)
        #             ptr += num_b
        #             # Weight
        #             bn_w = torch.from_numpy(
        #                 weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
        #             bn_layer.weight.data.copy_(bn_w)
        #             ptr += num_b
        #             # Running Mean
        #             bn_rm = torch.from_numpy(
        #                 weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
        #             bn_layer.running_mean.data.copy_(bn_rm)
        #             ptr += num_b
        #             # Running Var
        #             bn_rv = torch.from_numpy(
        #                 weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
        #             bn_layer.running_var.data.copy_(bn_rv)
        #             ptr += num_b
        #         else:
        #             # Load conv. bias
        #             num_b = conv_layer.bias.numel()
        #             conv_b = torch.from_numpy(
        #                 weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
        #             conv_layer.bias.data.copy_(conv_b)
        #             ptr += num_b
        #         # Load conv. weights
        #         num_w = conv_layer.weight.numel()
        #         conv_w = torch.from_numpy(
        #             weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
        #         conv_layer.weight.data.copy_(conv_w)
        #         ptr += num_w


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer,epoch, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "EPOCH": epoch
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, useNewLR=False):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    last_epoch = 0
    if "EPOCH" in checkpoint:
        last_epoch = checkpoint["EPOCH"]

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    if useNewLR == False:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return last_epoch


def get_loaders(train_csv_path, test_csv_path, do_return_org_image=False, do_return_org_images=False, do_return_flows_paths=False):
    from dataset import YOLODataset

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        train_csv_path,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        ret_org_im = do_return_org_image,
        ret_org_ims = do_return_org_images,
        ret_flows_paths = do_return_flows_paths,
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    # train_eval_dataset = YOLODataset(
    #     train_csv_path,
    #     transform=config.test_transforms,
    #     S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
    #     img_dir=config.IMG_DIR,
    #     label_dir=config.LABEL_DIR,
    #     anchors=config.ANCHORS,
    # )
    # train_eval_loader = DataLoader(
    #     dataset=train_eval_dataset,
    #     batch_size=config.BATCH_SIZE,
    #     num_workers=config.NUM_WORKERS,
    #     pin_memory=config.PIN_MEMORY,
    #     shuffle=False,
    #     drop_last=False,
    # )

    return train_loader, test_loader

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

def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression_torch_nms(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)



# def seed_everything(seed=42):
def seed_everything(seed=3):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
