import argparse
from asyncore import write
from cmath import e
from email.policy import default
from glob import glob
import logging
import os
from re import L
import sys
from tkinter import FALSE

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import ipdb
import torch.nn.functional as F
import random
from tensorboardX import SummaryWriter

# from eval import eval_net
from two_stream_unet import TwoStreamUNet, ContrastiveLoss
from rishabh import iou

torch.manual_seed(10)
# from torch.utils.tensorboard import SummaryWriter
# from utils.dataset import BasicDataset # write your own 
from utils_two_stream_unet import get_data, get_data_all_dataset, get_dict_vals, get_data_dict, get_data_dict_history
from torch.utils.data import DataLoader, random_split

tensorboard_flag = True
ROOT_FOLDER = '/data/raghavvg/NeedleMasks/' #COMMENT THIS IF IF SYSTEM CHANGES

IMAGE_LOC  = 'JPEGImages'
MASK_LOC = 'SegmentationClass'

PARENT_FOLDER_TRAIN = 'data' #os.path.join(ROOT_FOLDER,'data')
PARENT_FOLDER_TEST = 'data/test' #os.path.join(ROOT_FOLDER, 'data/test')

# LIST_OF_DATASETS_TRAIN = ['task_positives_189-2022_04_12_18_32_54-segmentation mask 1.1']
LIST_OF_DATASETS_TRAIN = [
                        'task_positives_11-2022_04_12_18_28_14-segmentation mask 1.1',
                        'task_positives_153-2022_04_12_18_32_22-segmentation mask 1.1',
                        'task_positives_189-2022_04_12_18_32_54-segmentation mask 1.1',
                        'task_positives_193-2022_04_18_19_26_03-segmentation mask 1.1',
                        'task_positives_67-2022_04_18_19_11_58-segmentation mask 1.1',
                        'task_positives_222-2022_04_19_22_31_15-segmentation mask 1.1',
                        'task_positives_230-2022_04_12_18_39_39-segmentation mask 1.1',
                        'task_1. extremity nerve with needle-2022_07_09_14_28_33-segmentation mask 1.1',
                        'task_1. extremity nerve with status post anesthetic injection-2022_07_09_15_38_21-segmentation mask 1.1',
                        'task_4. femoral nerve with vessels adjacent 1-2022_07_01_18_20_09-segmentation mask 1.1',
                        'task_4. femoral nerve with vessels adjacent 2-2022_07_08_17_03_24-segmentation mask 1.1',
                        'task_4. femoral nerve with vessels adjacent 3-2022_07_09_13_58_21-segmentation mask 1.1',
                        # 'task_positives_205-2022_04_21_17_04_12-segmentation mask 1.1' #! this is test dataset 
                        ]

LIST_OF_DATASETS_TEST = ['task_positives_205-2022_04_21_17_04_12-segmentation mask 1.1',
                    ]
iter = 16
dir_checkpoint = os.path.join('checkpoints/exp', str(iter))
if not os.path.exists(dir_checkpoint):
    os.makedirs(dir_checkpoint)

def convert_to_uint_and_transpose(img):
    # muliply by 255 and conver to numpy and uint8
    if type(img) != np.ndarray:
        img = img.numpy()
    
    img *= 255
    img = img.transpose(1,2,0)
    img = img.astype(np.uint8)

    return img

def get_weights(named_parameters, filter_name=None):
    '''
    return list of params 
    @param named_parameters: generator object for named parameters
    @param filter_name: filter name of the unwanted parameters

    '''
    weights_list = []
    for name, param in named_parameters:
        if filter_name is None:
            weights_list.append(param)
        else:
            if filter_name not in name:
                weights_list.append(param)
    
    return weights_list

def log_tensorboard(writer, global_step, type, true_masks, masks_pred, true_overlayed_imgs, pred_overlayed_imgs, imgs, flows,
                    spatial_features, temporal_features, x_spatial_shape, rand_inds):
    '''
    logger for only validation
    '''
    # choose random index to check | all 
    true_masks_ = true_masks[rand_inds]
    masks_pred_ = masks_pred[rand_inds]
    true_overlayed_imgs_ = true_overlayed_imgs[rand_inds]
    pred_overlayed_imgs_ = pred_overlayed_imgs[rand_inds]
    imgs_ = imgs[rand_inds]
    flows_ = flows[rand_inds]
    writer.add_images(type+'/images', imgs_, global_step)
    writer.add_images(type+'/flows', flows_[:,0:1,:,:], global_step)
    writer.add_images(type+'/mask_true', true_masks_, global_step)
    writer.add_images(type+'/mask_pred_sigmoid', masks_pred_, global_step)
    writer.add_images(type+'/mask_pred_', masks_pred_ > 0.5, global_step)
    writer.add_images(type+'/true_overlayed', true_overlayed_imgs_, global_step)
    writer.add_images(type+'/pred_overlayed', pred_overlayed_imgs_, global_step)
    if len(spatial_features) > 0: 
        pass
        # spatial_features_ = spatial_features[rand_inds].unsqueeze(2)
        # writer.add_images(type+'_features/spatial_features', spatial_features_.reshape(-1,1,x_spatial_shape[-2],x_spatial_shape[-1]), global_step)
    if len(temporal_features) > 0:
        pass
        # temporal_features_ = temporal_features[rand_inds].unsqueeze(2)
        # writer.add_images(type+'_features/temporal_features', temporal_features_.reshape(-1,1,x_spatial_shape[-2],x_spatial_shape[-1]), global_step)


def eval_net(net, test_data, n_classes, criterion, constrastive_criterion=None, device=None,
                 motion_flag=False, motion_criterion=None):
    '''
    pass data as dictionart and extract here 
    '''
    # images, flows, needle_masks, masks, flow_concats  = get_dict_vals(data)
    n_eval = len(test_data)
    batch_size = 10 #len(images)//4
    epoch_loss = 0
    motion_loss_list = []
    true_mask_list, pred_mask_list, loss_list, loss_v2_list, true_overlayed_imgs_list, pred_overlayed_imgs_list = [], [], [], [], [], []
    pred_v2_overlayed_imgs_list = []
    spatial_features_list, temporal_features_list = [], []
    images_list = []
    flows_list = []
    classification_loss_list = []

    with torch.no_grad():
        # for i in range(n_eval//batch_size):
        i = 0
        # while i <= (n_eval//batch_size):
        for i, data in enumerate(test_data):
      
            imgs = data['images'] #images_train[ind_] #batch['image']
            imgs_prev = data['images_prev']
            true_masks = data['needle_masks'] #needle_masks_train[ind_] #batch['mask']
            flows = data['flow_concats'] #flow_concats_train[ind_]
            flowsX = data['flow_x']
            flowsY = data['flow_y']
            flows_ = torch.cat([flowsX, flowsY], dim=1)
            needle_labels = data['needle_label']

            imgs = imgs.to(device=device)
            imgs_prev = imgs_prev.to(device=device) 
            mask_type = torch.float32 if n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            flows = flows.to(device=device) / 255
            flows_ = flows_.to(device=device, dtype=mask_type)
            needle_labels = needle_labels.to(device=device, dtype = mask_type)
            if motion_flag:
                masks_pred, x_bottle_neck, x_spatial, x_temporal, pred_feats, label_feats = net(imgs, flows, imgs_prev)
            else:
                masks_pred, x_bottle_neck, x_spatial, x_temporal = net(imgs,flows,imgs_prev) # flow in form of image 
            # masks_pred, x_spatial, x_temporal = net(imgs,flows_) # flow in form of x, y matrices

            if x_spatial is not None:
                x_spatial = x_spatial.to('cpu')
                spatial_features_list.append(x_spatial)
            if x_temporal is not None:
                x_temporal = x_temporal.to('cpu') 
                temporal_features_list.append(x_temporal)
            loss = criterion(masks_pred, true_masks) # bce with logits already has sigmoid 
            # classification_loss = criterion(needle_classification_logits, needle_labels)
            classification_loss = torch.tensor(0) #constrastive_criterion(x_bottle_neck, needle_labels)
            # loss = sigmoid_focal_loss(masks_pred, true_masks, device)
            masks_pred = torch.sigmoid(masks_pred)
            masks_pred_ = (masks_pred > 0.5).float() # additional filtering ? 
            # loss_v2 = criterion(masks_pred_, true_masks)
            # epoch_loss += loss.item()
            if motion_flag:
                motion_loss = motion_criterion(pred_feats.flatten(), label_feats.flatten())
                motion_loss_list.append(motion_loss.item())


            imgs = imgs.to('cpu') #+ 0.5 #, dtype=torch.float32        
            true_masks = true_masks.to('cpu') #, dtype=mask_type)
            flows = flows.to('cpu') #+ 0.5 #, dtype=mask_type)
            masks_pred = masks_pred.to( 'cpu')
            masks_pred_ = masks_pred_.to( 'cpu')

            images_list.append(imgs)
            flows_list.append(flows)
            loss_list.append(loss.item())
            classification_loss_list.append(classification_loss.item())
            # loss_v2_list.append(loss_v2.item())
            true_mask_list.append(true_masks)
            pred_mask_list.append(masks_pred)
            # overlayed_imgs_list.append(torch.concat([imgs.to('cpu'), true_masks, masks_pred_],dim = 1))
            tmp1 = torch.concat([imgs, 0.5*(imgs + true_masks), imgs], dim = 1) 
            tmp3 = torch.concat([imgs, 0.5*(imgs + masks_pred_), imgs], dim = 1) 
                    
            true_overlayed_imgs_list.append(tmp1)
            pred_v2_overlayed_imgs_list.append(tmp3)

        epoch_loss = np.mean(loss_list)
        epoch_classification_loss = np.mean(classification_loss_list)
        # epoch_loss_v2 = np.mean(loss_v2_list)
        images_list = torch.concat(images_list, dim=0)
        flows_list = torch.concat(flows_list, dim=0)
        true_mask_list = torch.concat(true_mask_list, dim=0)
        pred_mask_list = torch.concat(pred_mask_list, dim=0)
        true_overlayed_imgs_list = torch.concat(true_overlayed_imgs_list, dim=0)
        # pred_overlayed_imgs_list = torch.concat(pred_overlayed_imgs_list, dim=0)
        pred_v2_overlayed_imgs_list = torch.concat(pred_v2_overlayed_imgs_list, dim=0)

        if len(spatial_features_list) != 0:
            spatial_features_list = torch.concat(spatial_features_list, dim = 0)
        if len(temporal_features_list) != 0:
            temporal_features_list = torch.concat(temporal_features_list, dim = 0)

    return epoch_loss, epoch_classification_loss, true_mask_list, pred_mask_list, true_overlayed_imgs_list, pred_v2_overlayed_imgs_list, images_list \
            , spatial_features_list, temporal_features_list, flows_list

def find_contours(img, mask, threshold_flag=False):
    if threshold_flag:
        ret, mask = cv2.threshold(mask, 127, 255, 0)
    contours_pred, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2BGR)
    img = np.concatenate([img]*3, axis=-1)
    for i,c in enumerate(contours_pred):
        # mask = np.zeros(mask.shape, np.uint8)
        # cv2.drawContours(mask, [c], -1, 255, -1)
        # mean, _, _, _ = cv2.mean(mask, mask=mask)

        # Get appropriate colour for this label
        # label = 2 if mean > 1.0 else 1 # not needed as only 1 label
        colour = (0,0,255) #RGBforLabel.get(label)  
        cv2.drawContours(img,[c],-1,colour,1)
    
    return img

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              n_classes = 1,
              use_saved_data=False,
              task = 'train',
              saved_data_file = None, 
              weight_file = None,
              iter = 0,
              pure_image_flag = False,
              n_flow = 1,
              conv_layers = 1,
              flow_history_flag = False,
              learned_flow = False,
              classification_loss_flag = False,
              motion_flag=False
              ):

    global dir_checkpoint

    if not use_saved_data:
        train_data, max_flow, min_flow = get_data_dict_history(n_flow, LIST_OF_DATASETS_TRAIN, PARENT_FOLDER_TRAIN, saved_data_file,'train',flow_history_flag)
        test_data, _, _ = get_data_dict_history(n_flow, LIST_OF_DATASETS_TEST, PARENT_FOLDER_TEST, saved_data_file,'test', flow_history_flag, max_flow, min_flow)

    else:
        train_data = torch.load(os.path.join('saved_data', saved_data_file, 'train.pt')) #torch.load('saved_data/3/train.pt')
        test_data = torch.load(os.path.join('saved_data', saved_data_file, 'test.pt')) #torch.load('saved_data/3/test.pt')

    # Splitting train data into train and test 
    _len_data  = len(train_data)
    train_data, test_data = torch.utils.data.random_split(train_data, [int(0.75*_len_data), _len_data-int(0.75*_len_data)], generator=torch.Generator().manual_seed(42))
    
    # wrap data in dataloader 
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_data, batch_size=1*batch_size, shuffle=False)
    

    if task == 'train':
        dir_checkpoint = os.path.join('checkpoints/exp', str(iter) + '_conv_layers_{}_n_flow_={}'.format(conv_layers,n_flow))
        if not os.path.exists(dir_checkpoint):
            os.makedirs(dir_checkpoint)    
        
    n_val = len(train_data) #int(len(dataset) * val_percent)
    n_train = len(test_data) #- n_val

    """
    GENERATE DATA W/O WRAPPING IN DATALOADER, CAN'T USE SHUFFLE WITH FLOW or shuffle in batches
    for one image, flow has 3 previous frames, can create Bx5xHxW for flow and Bx1XHxW for US images 
    (make a dictionary and wrap dictionay in data loader)
    """
    """PARAMS for LOGGING"""

    patience = 5
    if tensorboard_flag:
        writer = SummaryWriter(comment='LR_{}_BS_{}_patience_{}_nflow_{}_conv_layers_{}_iter_{}'.format(lr, batch_size, patience, n_flow, conv_layers, iter))
    global_step = 0
    eval_step = 50

    if tensorboard_flag:
        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            Device:          {device.type}
            Images scaling:  {img_scale}''')

    # put flow network params on different lr
    if learned_flow:         
        optimizer = optim.AdamW([{'params':get_weights(net.UNet.named_parameters()) + get_weights(net.spatial_conv.named_parameters()) +\
                                     get_weights(net.temporal_conv.named_parameters())}, 
                                    {'params':get_weights(net.flownet.named_parameters()), 'lr':lr*1e-1}], lr=lr, weight_decay=1e-5) #momentum=0.9

    else:
        optimizer = optim.AdamW(get_weights(net.named_parameters(), filter_name="motion_net"), lr=lr, weight_decay=1e-6) 

    if motion_flag:
        motion_optimizer = optim.AdamW([{'params':get_weights(net.UNet.motion_net.named_parameters())}])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.9)
    
    if n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss() 
        constrastive_criterion = ContrastiveLoss()
    
    if motion_flag:
        motion_criterion = nn.MSELoss()

    rand_inds = random.sample(range(0, n_val), 20) # sampling 10 random numbers for plotting 


    ################################ VALIDATION ################################################ 

    if task == 'val':
        # only run eval and return | load weights 
        dir_checkpoint_ = os.path.join('checkpoints', 'exp', weight_file, 'CP_last.pth')
        # dir_checkpoint_ = os.path.join('/home/luyuan/thomaswe/TwoStreamUnet/checkpoints', 'exp', '13', 'CP_best.PTH')

        net.load_state_dict(torch.load(dir_checkpoint_))

        net.eval()

        val_score, true_masks_test, masks_pred_test, true_overlayed_imgs, pred_overlayed_imgs, imgs_test, spatial_features_test, temporal_features_test, flows_test \
                        = eval_net(net, test_data, n_classes, criterion, device, motion_flag=motion_flag, motion_criterion=motion_criterion)

        if net.n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(val_score))
            writer.add_scalar('Loss/test', val_score, global_step)
        else:
            logging.info('Validation Loss Coeff: {}'.format(val_score))
            writer.add_scalar('Loss/test', val_score, global_step)

        # find contours of predictions and ground truths and push on tensorboard
        for i in range(len(imgs_test)):
            pred_tmp = convert_to_uint_and_transpose(masks_pred_test[i])
            pred_tmp_ = convert_to_uint_and_transpose((masks_pred_test[i] > 0.5).float())
            img_tmp = convert_to_uint_and_transpose(imgs_test[i])
            true_mask_tmp = convert_to_uint_and_transpose(true_masks_test[i])
            img_overlayed_pred = find_contours(img_tmp, pred_tmp)  # cna also inp color
            img_overlayed_pred_ = find_contours(img_tmp, pred_tmp_)  # cna also inp color
            img_overlayed_true = find_contours(img_tmp, true_mask_tmp)  # cna also inp color

            if i%10 == 0:
                writer.add_images('test/images', imgs_test[i:i+1], i)
                writer.add_images('test/mask_true', true_masks_test[i:i+1], i)
                writer.add_images('test/mask_pred', masks_pred_test[i:i+1], i)
                writer.add_images('test/mask_pred_sigmoid', masks_pred_test[i:i+1] > 0.5, i)
                writer.add_images('test/true_overlayed',np.expand_dims(img_overlayed_true.transpose(2,0,1), axis=0), i)
                writer.add_images('test/pred_overlayed', np.expand_dims(img_overlayed_pred.transpose(2,0,1), axis=0), i)            
                writer.add_images('test/pred_overlayed_v2', np.expand_dims(img_overlayed_pred_.transpose(2,0,1), axis=0), i)            
        
        print("done evaluation ")
        writer.close()
        return 

    min_val_score = 1000 # some large number
    for epoch in range(epochs):
        net.train()
        epoch_loss = []        
        epoch_classification_loss = []
        epoch_motion_loss = []        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for i,data in enumerate(train_data):
                imgs = data['images'] #images_train[ind_] #batch['image']
                imgs_prev = data['images_prev']
                true_masks = data['needle_masks'] #needle_masks_train[ind_] #batch['mask']
                flows = data['flow_concats'] #flow_concats_train[ind_]
                flowsX = data['flow_x']
                flowsY = data['flow_y']
                flows_ = torch.cat([flowsX, flowsY], dim=1)
                needle_labels = data['needle_label']

                imgs = imgs.to(device=device)
                imgs_prev = imgs_prev.to(device=device)
                mask_type = torch.float32 if n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                flows = flows.to(device=device)/255 
                flows_ = flows_.to(device=device, dtype=mask_type)
                needle_labels = needle_labels.to(device=device, dtype=mask_type)
                if motion_flag:
                    masks_pred, x_bottle_neck, x_spatial, x_temporal, pred_feats, label_feats = net(imgs, flows, imgs_prev)
                else:
                    masks_pred, x_bottle_neck, x_spatial, x_temporal = net(imgs,flows,imgs_prev) # flow in form of image 

                loss =  criterion(masks_pred, true_masks)
            
                classification_loss = torch.tensor(0)#constrastive_criterion(x_bottle_neck, needle_labels)

                if motion_flag:
                    motion_loss = motion_criterion(pred_feats.flatten(), label_feats.flatten())
                    epoch_motion_loss.append(motion_loss.item())

                epoch_loss.append(loss.item())
                epoch_classification_loss.append(classification_loss.item())

                if tensorboard_flag:
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    writer.add_scalar('ClassificationLoss/train', classification_loss.item(), global_step)

                pbar.set_postfix(**{'loss(batch)': loss.item(), 'classification_loss(batch)': classification_loss.item(), 'lr':optimizer.param_groups[0]['lr']})

                optimizer.zero_grad()
                if motion_flag:
                    motion_optimizer.zero_grad()

                (loss).backward(retain_graph=True)
                if motion_flag:
                    motion_loss.backward()

                optimizer.step()
                if motion_flag:
                    motion_optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % eval_step == 0:
                    
                    net.eval()
                    val_score, val_classification_score ,true_masks_test, masks_pred_test, true_overlayed_imgs, pred_overlayed_imgs, imgs_test, spatial_features_test, temporal_features_test, flows_test \
                                    = eval_net(net, test_data, n_classes, criterion, device=device, motion_flag=motion_flag, motion_criterion=motion_criterion)
                    if classification_loss_flag:
                        if val_classification_score < min_val_score:
                            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_classification_best.pth')) 
                        
                        scheduler.step(val_classification_score)
                        #? if val_score not decreasing continuously for past 5 iterations then switch 
                        if epoch >= 0:
                            print("switching to training segmentation as well")
                            classification_loss_flag = False    
                        net.train()               
                        logging.info('Validation Loss: {}'.format(val_classification_score))
                    else:
                        if val_score < min_val_score:
                            min_val_score = val_score
                            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_best.pth'))
                        scheduler.step(val_score)                        
                        
                        net.train()
                        logging.info('Validation Loss: {}'.format(val_score))
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    test_iou = iou(masks_pred_test, true_masks_test)
                    writer.add_scalar('IOU/test', test_iou, global_step)
                    writer.add_scalar('Loss/test', val_score, global_step)
                    writer.add_scalar('ClassificationLoss/test', val_classification_score, global_step)
                    writer.add_images('train/images', imgs, global_step)
                    writer.add_images('train/flows', flows[:,0:1,:,:], global_step)
                    if flow_history_flag:
                        writer.add_images('train/flows_hist', flows[:,1:2,:,:], global_step)
                    if net.n_classes == 1:
                        writer.add_images('train/mask_true', true_masks, global_step)
                        masks_pred_sigmoid = torch.sigmoid(masks_pred) # don't know why we were passing via sigmoid 
                        masks_pred_ = (masks_pred_sigmoid > 0.5).float()
                        if x_spatial is not None:
                            x_spatial = x_spatial.unsqueeze(2)
                            x_spatial_shape = x_spatial.shape
                        if x_temporal is not None:
                            x_temporal = x_temporal.unsqueeze(2)
                        writer.add_images('train/mask_pred_', masks_pred_, global_step)
                        writer.add_images('train/mask_pred_sigmoid', masks_pred_sigmoid, global_step)
                        writer.add_images('train/true_overlayed' ,torch.concat([imgs, 0.5*(imgs + true_masks), imgs], dim = 1), global_step)
                        writer.add_images('train/pred_overlayed' ,torch.concat([imgs, 0.5*(imgs + masks_pred_), imgs], dim = 1), global_step)

                        train_iou = iou(masks_pred_, true_masks)
                        writer.add_scalar('IOU/train', train_iou, global_step)

                        rand_inds = random.sample(range(0, n_val), 20) # choose any random image at each logging 
                        log_tensorboard(writer, global_step, 'test', true_masks_test, masks_pred_test, true_overlayed_imgs, pred_overlayed_imgs, imgs_test, flows_test,
                                        spatial_features_test, temporal_features_test, x_spatial_shape, rand_inds)                     

            print("epoch loss : ", np.mean(epoch_loss))
            writer.add_scalar('Loss/Epoch_train', np.mean(epoch_loss), epoch)
            if motion_flag:
                writer.add_scalar('MotionLoss/Epoch_train', np.mean(epoch_motion_loss), epoch)
        if save_cp:
            try:
                os.makedirs(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if classification_loss_flag:
                torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_classification_last.pth'))
            else:
                torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_last.pth'))

    writer.close()


def sigmoid_focal_loss(inputs, targets, device, alpha = 0.75, gamma = 2, reduction = "none"):
    '''
    reference link: https://pytorch.org/vision/0.12/_modules/torchvision/ops/focal_loss.html
    '''
    # ipdb.set_trace()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", pos_weight=torch.tensor([150]).to(device))
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean" or reduction == "none":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()    

    return loss

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--use_saved_data', action='store_true' ,default=False)
    parser.add_argument('--task', type=str, default = 'train')
    parser.add_argument('--saved_data_file', type=str, default='4')
    parser.add_argument('--iter', type=str, default='0')
    parser.add_argument('--val_weights', type=str, default='0')
    parser.add_argument('--pure_images', action='store_true', default=False)
    parser.add_argument('--n_flow', type=int, default=1)
    parser.add_argument('--conv_layers', type=int, default=1)
    parser.add_argument('--multi_attn', type=int, default=0)
    parser.add_argument('--flow_history_flag', action='store_true', default=False)
    parser.add_argument('--learned_flow', action='store_true', default=False)
    parser.add_argument('--late_fusion', action='store_true', default=False)
    parser.add_argument('--classification_flag', action='store_true', default=False)
    parser.add_argument('--attention_flag', action='store_true', default=False)
    parser.add_argument('--motion_flag', type=bool, default=False)
    # parser.add_argument('--scale', type = int, default=1)
    # parser.add_argument('--log', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    # net = UNet(n_channels=3, n_classes=1, bilinear=True)
    ## TO CHECK: EFFECT OF BILINEAR AND N_CLASS = 1 ######################################## | ABLATE on increasing temporal channels 
    if not args.late_fusion:
        scale = 1
        net = TwoStreamUNet(spatial_in_channel=1, temporal_in_channel=scale*args.n_flow, out_channel=16,
                            n_classes=1, pure_images_flag=args.pure_images, 
                            conv_flag=args.conv_layers, learned_flow=args.learned_flow,
                            classification_flag = args.classification_flag, attention_flag=args.attention_flag, multi_attn=args.multi_attn,
                            device = device, motion_flag=args.motion_flag
                            )
        # freeze flownet parameters
        # ipdb.set_trace()
        # for tag, value in net.named_parameters():
        #     print("tag = " , tag , "  " , value.requires_grad)

    # print("net = \n ", net)
    # logging.info(f'Network:\n'
    #              f'\t{net.out_channel} input channels\n'
    #              f'\t{net.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    # test_train(net, device)
    # if args.task == 'train':
    # print("val_weights: ", args.val_weights)
    try:
        train_net(net=net,
                epochs=args.epochs,
                batch_size=args.batchsize,
                lr=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=args.val / 100,
                use_saved_data=args.use_saved_data,
                task = args.task,
                saved_data_file = args.saved_data_file,
                weight_file = args.val_weights,
                iter = args.iter,
                pure_image_flag = args.pure_images,
                n_flow = args.n_flow,
                conv_layers = args.conv_layers,
                flow_history_flag = args.flow_history_flag,
                learned_flow=args.learned_flow,
                classification_loss_flag=args.classification_flag, 
                motion_flag=args.motion_flag
                )
    except KeyboardInterrupt:
        # torch.save(net.state_dict(), 'INTERRUPTED.pth')
        torch.save(net.state_dict(), os.path.join(dir_checkpoint,'INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    # else:
    #     eval_net(net, data, n_classes, criterion, device)