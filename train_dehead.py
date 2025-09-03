"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import numpy as np
import os
import pickle
# import hashlib
import torch.optim as optim
from logger import Logger
from model_org_dehead import YOLOv3
# from model_org import YOLOv3
# from model_org import YOLOv3 as MOYOLOv3
from tqdm import tqdm
from testVOD_voc2007 import do_voc_test
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    # load_darknet_weights
)
import math
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, logger, epoch):
    loop = tqdm(train_loader, leave=True)
    # losses = []
    losses_sum = 0
    losses_count = 0

    box_losses_sum = 0
    box_losses_count = 0

    object_losses_sum = 0
    object_losses_count = 0

    no_object_losses_sum = 0
    no_object_losses_count = 0

    class_losses_sum = 0
    class_losses_count = 0

    contact_losses_sum = 0
    contact_losses_count = 0

    association_vector_losses_sum = 0
    association_vector_losses_count = 0

    for batch_idx, (x, y) in enumerate(loop):
        dataloader_len = len(train_loader)
        batches_done = dataloader_len * epoch + batch_idx
        
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            l1 = loss_fn(out[0], y0, scaled_anchors[0])
            l2 = loss_fn(out[1], y1, scaled_anchors[1])
            l3 = loss_fn(out[2], y2, scaled_anchors[2])
            tot_loss = (l1[0] + l2[0] + l3[0])
            tot_box_loss = (l1[1] + l2[1] + l3[1])
            tot_object_loss = (l1[2] + l2[2] + l3[2])
            tot_no_object_loss = (l1[3] + l2[3] + l3[3])
            tot_class_loss = (l1[4] + l2[4] + l3[4])
            tot_contact_loss = (l1[5] + l2[5] + l3[5])
            tot_association_vector_loss = (l1[6] + l2[6] + l3[6])
       #
       # print("\n__________________________________")
       #  print("Box loss: ", tot_box_loss.item())
       #  print("Object loss: ",  tot_object_loss.item())
       #  print("No-Object loss: ",  tot_no_object_loss.item())
       #  print("Class loss: ",  tot_class_loss.item())
       #  print("Contact loss: ",  tot_contact_loss.item())
       #  print("Association vector loss: ",  tot_association_vector_loss.item())
       #  print("\n")

        # losses.append(tot_loss.item())
        optimizer.zero_grad()
        scaler.scale(tot_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        losses_sum += tot_loss.item()
        losses_count += 1
        mean_loss = losses_sum / losses_count

        box_losses_sum += tot_box_loss.item()
        box_losses_count += 1
        mean_box_loss = box_losses_sum / box_losses_count

        object_losses_sum += tot_object_loss.item()
        object_losses_count += 1
        mean_object_loss = object_losses_sum / object_losses_count

        no_object_losses_sum += tot_no_object_loss.item()
        no_object_losses_count += 1
        mean_no_object_loss = no_object_losses_sum / no_object_losses_count

        class_losses_sum += tot_class_loss.item()
        class_losses_count += 1
        mean_class_loss = class_losses_sum / class_losses_count

        contact_losses_sum += tot_contact_loss.item()
        contact_losses_count += 1
        mean_contact_loss = contact_losses_sum / contact_losses_count

        association_vector_losses_sum += tot_association_vector_loss.item()
        association_vector_losses_count += 1
        mean_association_vector_loss = association_vector_losses_sum / association_vector_losses_count
        
        # Tensorboard logging
        tensorboard_log = [
                ("train/bbox_loss", float(mean_box_loss)),
                ("train/obj_loss", float(mean_object_loss)),
                ("train/class_loss", float(mean_class_loss)),
                ["train/contact_loss", float(mean_contact_loss)],
                ["train/vector_loss", float(mean_association_vector_loss)],
                ["train/nobj_loss", float(mean_no_object_loss)],
                ("train/loss", float(mean_loss))]
        logger.list_of_scalars_summary(tensorboard_log, batches_done)

        # if math.isnan(mean_loss):
        #     print("Oh no!")
        cur_lr = [x['lr'] for x in optimizer.param_groups]
        loop.set_postfix(lr=cur_lr, mean_loss=mean_loss, mean_box_loss=mean_box_loss, mean_object_loss=mean_object_loss, mean_no_object_loss=mean_no_object_loss, mean_class_loss=mean_class_loss, mean_contact_loss=mean_contact_loss, mean_association_vector_loss=mean_association_vector_loss)

def adjust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']



def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Load darknet53 pre-trained weights
    if config.LOAD_DARKNET53_IMAGENET_WEIGHTS:
        # keys_to_skip = ['layers.15.pred.1.conv.weight', 'layers.15.pred.1.conv.bias', 'layers.15.pred.1.bn.weight', 'layers.15.pred.1.bn.bias', 'layers.15.pred.1.bn.running_mean',
        #                 'layers.15.pred.1.bn.running_var', 'layers.22.pred.1.conv.weight', 'layers.22.pred.1.conv.bias', 'layers.22.pred.1.bn.weight', 'layers.22.pred.1.bn.bias',
        #                 'layers.22.pred.1.bn.running_mean', 'layers.22.pred.1.bn.running_var', 'layers.29.pred.1.conv.weight', 'layers.29.pred.1.conv.bias', 'layers.29.pred.1.bn.weight',
        #                 'layers.29.pred.1.bn.bias', 'layers.29.pred.1.bn.running_mean', 'layers.29.pred.1.bn.running_var']

        keys_to_skip = ['layers.15.pred.1.conv.weight', 'layers.15.pred.1.conv.bias', 'layers.22.pred.1.conv.weight',
                        'layers.22.pred.1.conv.bias', 'layers.29.pred.1.conv.weight', 'layers.29.pred.1.conv.bias']

        # yolov3_pretrained_on_pascal_voc = torch.load(config.DARKNET53_IMAGENET_WEIGHTS_PATH, map_location=config.DEVICE)
        yolov3_pretrained_on_pascal_voc = torch.load("./yolov3_pascal_78.1map.pth.tar", map_location=config.DEVICE)
        # model.load_state_dict(yolov3_pretrained_on_pascal_voc['state_dict'], strict=False)
        m = yolov3_pretrained_on_pascal_voc['state_dict']
        model_dict = model.state_dict()
        for k in m.keys():
            if k in keys_to_skip:
                continue

            if k in model_dict:
                pname = k
                pval = m[k]
                model_dict[pname] = pval.clone().to(model_dict[pname].device)
        # old_ckp = torch.load(config.DARKNET53_IMAGENET_WEIGHTS_PATH, map_location=config.DEVICE)
        # model.load_state_dict(old_ckp['state_dict'], strict=False)

        model.load_state_dict(model_dict)

    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
    lf = config.one_cycle(1, 0.1, config.NUM_EPOCHS)
    cos_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lf)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    logger = Logger('logs')

    train_loader, test_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
    #     )
    
    
    
    # m = hashlib.md5()
    # m.update("yolov3 dehead gn 0")
    # unique_name = str(int(m.hexdigest(), 16))[0:12]
    unique_name = "yv3_"+str(config.IMAGE_SIZE)+"_dehead_silu_gn_fl_3"
    aps_filename = unique_name + '.pkl'
    last_epoch = -1

    checkpoint = torch.load("100DOH_DL_416_YOLOV3_CKP_SILU_GN_DEHEAD_EP_LAST.pth.tar", map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    last_epoch = int(checkpoint["EPOCH"])

    # print("\n")
    # print("Evaluating...")
    # model.eval()
    # apsd = do_voc_test(model)
    # print(apsd)
    # model.train()
    # new_map = apsd['targetobject'] + apsd['hand'] + apsd['handside'] + apsd['handstate'] + apsd['objectbbox'] + apsd['all']
    # new_map /= 6

    # evaluation_metrics = [
    #     ("validation/targetobject", apsd['targetobject']),
    #     ("validation/hand", apsd['hand']),
    #     ("validation/handside", apsd['handside']),
    #     ("validation/handstate", apsd['handstate']),
    #     ("validation/objectbbox", apsd['objectbbox']),
    #     ("validation/all", apsd['all']),
    #     ("validation/mean", new_map),
    # ]
    # logger.list_of_scalars_summary(evaluation_metrics, last_epoch+1)
    # model.train()
    #
    # exit(0)

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)


    for epoch in range(last_epoch+1, config.NUM_EPOCHS):
        
        
        print("\n")
        print(f"Epoch {epoch}:")

        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, logger, epoch)
        # cos_lr_scheduler.step()

        #if epoch >= 0 and epoch % config.SAVE_EPOCHS == 0:
        #    if config.SAVE_MODEL:
                # save_checkpoint(model, optimizer, epoch, filename=config.CHECKPOINT_FILE_SAVE_NAME+"_SILU_GN_DEHEAD_EP"+str(epoch)+".pth.tar")
        save_checkpoint(model, optimizer, epoch, filename=config.CHECKPOINT_FILE_SAVE_NAME+"_SILU_GN_DEHEAD_EP_LAST.pth.tar")

        if epoch >= 0 and epoch % config.TEST_EPOCHS == 0:
            print("\n")
            print("Evaluating...")
            model.eval()
            apsd = do_voc_test(model)
            new_map = apsd['targetobject'] + apsd['hand'] + apsd['handside'] + apsd['handstate'] + apsd['objectbbox'] + apsd['all']
            new_map /= 6
            print("Total mAP: " + str(new_map))
            print(apsd)
            
            evaluation_metrics = [
                    ("validation/targetobject", apsd['targetobject']),
                    ("validation/hand", apsd['hand']),
                    ("validation/handside", apsd['handside']),
                    ("validation/handstate", apsd['handstate']),
                    ("validation/objectbbox", apsd['objectbbox']),
                    ("validation/all", apsd['all']),
                    ("validation/mean", new_map),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)
            
            if os.path.exists(aps_filename):
                with open(aps_filename, 'rb') as f:
                    loaded_apsd = pickle.load(f)
                    
                old_map = loaded_apsd['targetobject'] + loaded_apsd['hand'] + loaded_apsd['handside'] + loaded_apsd['handstate'] + loaded_apsd['objectbbox'] + loaded_apsd['all']
                old_map /= 6
                print("Old mAP: " + str(old_map))
                
                new_best=False
                if new_map > old_map:
                    new_best = True
                    
                if new_best:
                    print("There is a new best!")
                    save_checkpoint(model, optimizer, epoch, filename=config.CHECKPOINT_FILE_SAVE_NAME+"_SILU_GN_DEHEAD_EP_BEST.pth.tar")
                    with open(aps_filename, 'wb') as f:
                        pickle.dump(apsd, f)
            else:
                with open(aps_filename, 'wb') as f:
                    pickle.dump(apsd, f)
                save_checkpoint(model, optimizer, epoch, filename=config.CHECKPOINT_FILE_SAVE_NAME+"_SILU_GN_DEHEAD_EP_BEST.pth.tar")
                
                
            model.train()
    if config.SAVE_MODEL:
        save_checkpoint(model, optimizer, epoch, filename=config.CHECKPOINT_FILE_SAVE_NAME+"_SILU_GN_DEHEAD_EP"+str(epoch)+".pth.tar")
    print("\n")
    print("Evaluating...")
    model.eval()
    apsd = do_voc_test(model)
    model.train()


if __name__ == "__main__":
    main()
