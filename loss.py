"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn

from utils import intersection_over_union
import math
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
    
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        #self.mseObj = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        #self.bceHS = nn.BCEWithLogitsLoss()
        #class_weights = torch.tensor([5, 1, 1], device=torch.device('cuda:0'))
        #self.entropy = nn.CrossEntropyLoss(weight=class_weights)
        self.entropy = nn.CrossEntropyLoss()
        #self.entropy = FocalLoss(gamma=2.0)
        self.sigmoid = nn.Sigmoid()
        
        #Focal class loss
        #self.entropy = FocalLoss(self.entropy)
        #self.bce = FocalLoss(self.bce)
        #self.mseObj = FocalLoss(self.mseObj)
        #self.bceHS = FocalLoss(self.bceHS)
        
        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

        # HOI extension constants signifying how much to pay for each respective part of the loss
        self.lambda_contact = 1
        self.lambda_association_vector= 0.1


    def forward(self, predictions, target, anchors, flows=None, offsets=None):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        #box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3])*2 -0.5, (torch.sigmoid(predictions[..., 3:5])*2)**2 * anchors], dim=-1)
        target_boxes = target[..., 1:5][obj]
        #ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        #obj_loss_target = ious * target[..., 0:1][obj]
        ious = bbox_iou(box_preds[obj], target_boxes, CIoU=True) # iou(prediction, target)
        obj_loss_target = ious.detach() * target[..., 0:1][obj]
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), obj_loss_target)

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        #predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        #target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))  # width, height coordinates
        #box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])
        
        # CIOU Loss
        #box_preds = box_preds[obj]
        #iou = bbox_iou(box_preds, target_boxes, CIoU=True).squeeze()  # iou(prediction, target)
        box_loss = (1.0 - ious.squeeze()).mean()  # 

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        #print((predictions[..., 5:8][obj]).shape)
        #print((target[..., 5][obj].long()).shape)
        class_loss = self.entropy((predictions[..., 5:8][obj]), (target[..., 5][obj].long()),)        

        # ========================= #
        #   FOR CONTACT STATE LOSS  #
        # ========================= #

        # For contact loss we care only about minimizing loss for hand_left hand_right class
        contact_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
        hands = (target[..., 0] == 1)  & (target[..., 5] > 0)
        if len(target[..., 6][hands]) > 0:
            contact_loss = self.bce((predictions[..., 8:9][hands]), (target[..., 6][hands].unsqueeze(-1)),)

        # ============================== #
        #   FOR ASSOCIATION VECTOR LOSS  #
        # ============================== #

        # For association vector loss we care only about minimizing loss for in contact hand classes
        association_vector_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
        in_contacts = (target[..., 0] == 1)  & (target[..., 6] == 1)
        # in_contacts = (target[..., 0] == 1)  & (target[..., 6] == 1)
        inc_preds = (predictions[..., 9:][in_contacts])
        if len(inc_preds) > 0:
            dxdymagnitude_pred_sub = 0.1 * F.normalize(inc_preds[:, 1:], p=2, dim=1)
            magnitude = inc_preds[:, 0].unsqueeze(-1)
            dxdymagnitude_pred_norm = torch.cat([magnitude, dxdymagnitude_pred_sub], dim=1)
            association_vector_loss = self.mse((dxdymagnitude_pred_norm), (target[..., 7:][in_contacts]),)


        # print("\n__________________________________")
        # print("Box loss: ",self.lambda_box * box_loss)
        # print("Object loss: ",self.lambda_obj * object_loss)
        # print("No-Object loss: ",self.lambda_noobj * no_object_loss)
        # print("Class loss: ",self.lambda_class * class_loss)
        # print("Contact loss: ",self.lambda_contact * contact_loss)
        # print("Association vector loss: ",self.lambda_association_vector * association_vector_loss)
        # print("\n")

        # if math.isnan(association_vector_loss.item()):
        #     print("Yoooo!")



        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
            + self.lambda_contact * contact_loss
            + self.lambda_association_vector * association_vector_loss,

            self.lambda_box * box_loss,
            self.lambda_obj * object_loss,
            self.lambda_noobj * no_object_loss,
            self.lambda_class * class_loss,
            self.lambda_contact * contact_loss,
            self.lambda_association_vector * association_vector_loss,
        )
