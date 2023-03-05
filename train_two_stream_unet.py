import argparse
import logging
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import ipdb
import torch.nn.functional as F
from math import pi
import random
from tensorboardX import SummaryWriter

# from eval import eval_net
from two_stream_unet import TwoStreamUNet, ContrastiveLoss
from rishabh import iou, dice_score, precision_recall

# torch.manual_seed(10)
torch.manual_seed(42)
# torch.manual_seed(101)

# from torch.utils.tensorboard import SummaryWriter
# from utils.dataset import BasicDataset # write your own
from utils_two_stream_unet import find_flow_history, get_dict_vals, sigmoid_focal_loss, \
                                    get_data_dict_history, get_data_dict_kalman, get_weights, \
                                get_list_train_test_data, get_data_dict_kalman_vec


from torch.utils.data import DataLoader, random_split

# ROOT_FOLDER = '/data/raghavvg/NeedleMasks/' #COMMENT THIS IF IF SYSTEM CHANGES

# IMAGE_LOC  = 'JPEGImages'
# MASK_LOC = 'SegmentationClass'

# PARENT_FOLDER_TRAIN = 'new_dataset' #os.path.join(ROOT_FOLDER,'data')
# PARENT_FOLDER_TEST = 'new_dataset/test' #os.path.join(ROOT_FOLDER, 'data/test')

#! command for transfering data:  rsync -a --ignore-existing data/PigLabData  luyuan@luyuan.wifi.cmu.edu:~/thomaswe/TwoStreamUnet/new_dataset

#! command to run for training: (conda activate vision_stuff)
#! python train_two_stream_unet.py --use_saved_data --saved_data_file 601_dict_normalized_all_positive_nflow=1_gauss_both_needle_label --epochs 150 --batch-size 10 --iter gauss_vanilla_L2_reg_1e-4

# ! command to validate (will be on the video in LIST_OF_DATASETS_TEST)

#! use --samples_per_plugin images=100 to get images visible in tensorboard at every 100 steps

def log_tensorboard(writer, global_step, type, true_masks, masks_pred, true_overlayed_imgs, pred_overlayed_imgs, true_pred_overlayed_imgs, imgs, sigma, flows,
                    rand_inds, kalman_flag=False, flow_flag=False):
    '''
    logger for only validation
    '''
    # choose random index to check | all 
    sigma_= None
    if kalman_flag:
        true_masks_ = true_masks
        masks_pred_ = masks_pred
        true_overlayed_imgs_ = true_overlayed_imgs
        pred_overlayed_imgs_ = pred_overlayed_imgs
        imgs_ = imgs
        sigma_ = sigma
    else:
        true_masks_ = true_masks[rand_inds]
        masks_pred_ = masks_pred[rand_inds]
        true_overlayed_imgs_ = true_overlayed_imgs[rand_inds]
        pred_overlayed_imgs_ = pred_overlayed_imgs[rand_inds]
        imgs_ = imgs[rand_inds]
        true_pred_overlayed_imgs = true_pred_overlayed_imgs[rand_inds]
        # flows_rand_inds = flows[rand_inds]
        # writer.add_images(type+'/flows', flows_rand_inds[:,0:1,:,:], global_step)
    
    writer.add_images(type+'/images', imgs_, global_step)
    # writer.add_images(type+'/mask_pred_sigmoid', masks_pred_, global_step)
    writer.add_images(type+'/true_overlayed', true_overlayed_imgs_, global_step)
    writer.add_images(type+'/pred_overlayed', pred_overlayed_imgs_, global_step)
    if true_pred_overlayed_imgs is not None:
        writer.add_images(type+'/combined_overlayed', true_pred_overlayed_imgs, global_step)
    if sigma_ is not None:
        writer.add_images(type+'/uncertainty', sigma, global_step)


def eval_net(writer, global_step, net, test_data, n_classes, criterion, device, args):
    '''
    #! EVALUATE
    '''
    # images, flows, needle_masks, masks, flow_concats  = get_dict_vals(data)    
    kalman_flag = args.kalman_flag 
    gauss_flag = args.gauss_flag
    flow_flag = args.flow_flag
    n_flow = args.n_flow
    vector_flag = args.vector_flag

    n_eval = len(test_data)    
    epoch_loss = 0
    true_mask_list, pred_mask_list, loss_list = [], [], []
    true_overlayed_imgs_list, pred_overlayed_imgs_list, true_pred_overlayed_imgs_list = [], [], []
    sigma_list, mean_list = [], []
    if vector_flag:
        x_start_error_list, y_start_error_list, angle_error_list, length_error_list = [], [], [], []
    spatial_features_list, temporal_features_list = [], []
    images_list = []
    flows_list = []
    iou_list = []
    tp_by_all_positive_list = []
    precision_list, recall_list, dice_score_list = [], [], []
    with torch.no_grad():
        for i, data in enumerate(test_data):                   
            mask_type = torch.float32 #if n_classes == 1 else torch.long
            imgs = data['images'] 
            masks_true = data['needle_masks'] 
            flows = None
            imgs_prev = None
            if kalman_flag and vector_flag:
                true_mask_new = data['needle_masks_new']
                needle_params = data['needle_params'] 
                needle_params = needle_params.to(device=device, dtype=mask_type)
            # elif flow_flag:
            #     imgs_prev = data['images_prev']
            #     flows = data['flow_concats'] 
            #     imgs_prev = imgs_prev.to(device=device) 
            #     flows = flows.to(device=device, dtype=mask_type) / 255
            else:
                if n_flow>1:
                    imgs_prev = data['images_prev']
                    imgs_prev = imgs_prev.to(device=device, dtype=mask_type) 

            imgs = imgs.to(device=device, dtype=mask_type)
            masks_true = masks_true.to(device=device, dtype=mask_type)
            masks_pred, mean, sigma, flow = net(imgs,flows,imgs_prev) 
            # masks_pred, x_spatial, x_temporal = net(imgs,flows_) # flow in form of x, y matrices

            if gauss_flag:
                sigma = sigma.to('cpu')                
                sigma = sigma/torch.max(sigma) if torch.max(sigma) > 1 else sigma                
                sigma_list.append(sigma)
                # use mean as prediction during evaluation                 
                masks_pred = mean

            if vector_flag:
                # scale_needle_param = torch.tensor([1/256, 1/256, 1/(2*pi), 1/(256*(2**0.5))]).to(device)
                scale_needle_param = torch.tensor([256, 256, (2*pi), 256]).to(device)
                masks_pred  = masks_pred[:,:,:4]*scale_needle_param
                masks_true = needle_params[:,:,:4] #*scale_needle_param 
                x_start_error = torch.mean(torch.abs(needle_params[:,:,0] - masks_pred[:,:,0]))
                y_start_error = torch.mean(torch.abs(needle_params[:,:,1] - masks_pred[:,:,1]))
                angle_error = torch.mean(torch.abs(needle_params[:,:,2] - masks_pred[:,:,2])) 
                length_error = torch.mean(torch.abs(needle_params[:,:,3] - masks_pred[:,:,3]))
            else:
                masks_pred = torch.sigmoid(masks_pred)
                masks_pred_threshold = (masks_pred > 0.5).float() # additional filtering ? 
                #* compute IOU and keep a store            
                iou_batch, tp_by_all_positive, _ = iou(masks_pred_threshold, masks_true, kalman_flag=kalman_flag)
                iou_batch, tp_by_all_positive = iou_batch.item(), tp_by_all_positive.item()
                iou_list.append(iou_batch)
                tp_by_all_positive_list.append(tp_by_all_positive)
                #* compute precision, recall and DSC
                dice_score_value = dice_score(masks_pred_threshold, masks_true, kalman_flag=kalman_flag)
                precision, recall = precision_recall(masks_pred_threshold, masks_true, kalman_flag=kalman_flag)
                dice_score_value, precision, recall = dice_score_value.item(), precision.item(), recall.item()
                precision_list.append(precision); recall_list.append(recall); dice_score_list.append(dice_score_value) 

                masks_pred_threshold = masks_pred_threshold.to( 'cpu')
            
            # FIND LOSS
            loss = criterion(masks_pred, masks_true)
            loss_list.append(loss.item())

            # SEND TENSORS TO CPU
            imgs = imgs.to('cpu') 
            masks_true = masks_true.to('cpu') 
            masks_pred = masks_pred.to( 'cpu')
            if flow_flag:
                flows = flows.to('cpu') 
                flows_list.append(flows)

            if len(images_list) < 20:
                images_list.append(imgs)

                if vector_flag:
                    x_start_error_list.append(x_start_error)
                    y_start_error_list.append(y_start_error)
                    angle_error_list.append(angle_error)
                    length_error_list.append(length_error)
                else:
                    true_mask_list.append(masks_true)
                    pred_mask_list.append(masks_pred)
                    true_overlayed_img = torch.concat([imgs, 0.6*imgs + 0.4*masks_true, imgs], dim = -3) 
                    pred_overlayed_img = torch.concat([imgs, 0.6*imgs + 0.4*masks_pred_threshold, imgs], dim = -3) 
                    true_pred_overlayed_img = torch.concat([masks_pred_threshold, torch.zeros_like(imgs), masks_true], dim = -3) 

                    true_overlayed_imgs_list.append(true_overlayed_img)
                    pred_overlayed_imgs_list.append(pred_overlayed_img)
                    true_pred_overlayed_imgs_list.append(true_pred_overlayed_img)

        epoch_loss = np.mean(loss_list)
        images_list = torch.concat(images_list, dim=0)
        if not vector_flag:
            true_mask_list = torch.concat(true_mask_list, dim=0)        
            pred_mask_list = torch.concat(pred_mask_list, dim=0)
            true_overlayed_imgs_list = torch.concat(true_overlayed_imgs_list, dim=0)
            pred_overlayed_imgs_list = torch.concat(pred_overlayed_imgs_list, dim=0)
            true_pred_overlayed_imgs_list = torch.concat(true_pred_overlayed_imgs_list, dim=0)
            if flow_flag:
                flows_list = torch.concat(flows_list, dim=0)
            if gauss_flag:            
                sigma_list = torch.concat(sigma_list, dim=0)
    
    if not vector_flag:
        avg_iou = np.mean(iou_list)
        avg_tp_by_all_positive = np.mean(tp_by_all_positive_list)
        avg_dice_score = np.mean(dice_score_list)
        avg_recall = np.mean(recall_list)
        avg_precision = np.mean(precision)
        print("AVERAGE iou={}, dice_score={}, recall={}, precision={}".format(avg_iou, avg_dice_score, avg_recall, avg_precision))

    # write random predictions and ground truths 
    if kalman_flag:
        rand_inds = range(imgs.shape[1]) #random.sample(range(0, n_eval), min(n_eval,10)) # choose any random image at each logging 
        batch_idx = random.sample(range(0, len(images_list)-1),1)[0]                
        if vector_flag:
            masks = torch.zeros_like(masks_true[0], dtype=torch.uint8).permute(0, 2, 3, 1).numpy()
            masks_pred = masks_pred[0].numpy()
            masks_list = []
            for kk in range(masks.shape[0]):
                x_tip = masks_pred[kk,0] + np.cos(masks_pred[kk,2])*masks_pred[kk,3]
                y_tip = masks_pred[kk,1] + np.sin(masks_pred[kk,2])*masks_pred[kk,3]
                # masks_tmp = cv2.circle(masks[kk], (int(round(masks_pred[kk,1])), int(round(masks_pred[kk,0]))), 3, (0, 0, 255), 2)
                # masks_tmp = cv2.circle(masks_tmp, (int(round(y_tip)), int(round(x_tip))), 3, (0, 255, 0), 2)
                masks_tmp = cv2.line(np.concatenate([masks[kk]]*3, axis=-1), (int(round(masks_pred[kk,1])), int(round(masks_pred[kk,0]))), (int(round(y_tip)), int(round(x_tip))), (0,0,255), 4)
                masks_list.append(masks_tmp)
            masks_list = torch.from_numpy(np.stack(masks_list)).permute(0, 3, 1, 2)
            #? just logging error of last data
            writer.add_images('test/mask_pred', masks_list, global_step)
            writer.add_images('test/mask_pred_overlayed', masks_list + true_mask_new[0], global_step)
            writer.add_images('test/true_pred', masks_true[0], global_step)
            writer.add_images('test/true_pred_new', true_mask_new[0], global_step)
            writer.add_scalar('test/x_start_error', torch.mean(torch.stack(x_start_error_list)), global_step)
            writer.add_scalar('test/y_start_error', torch.mean(torch.stack(y_start_error_list)), global_step)
            writer.add_scalar('test/length_error', torch.mean(torch.stack(length_error_list)), global_step)
            writer.add_scalar('test/angle_error', torch.mean(torch.stack(angle_error_list)), global_step)
        elif gauss_flag:
            log_tensorboard(writer, global_step, 'test', true_mask_list[batch_idx], pred_mask_list[batch_idx], true_overlayed_imgs_list[batch_idx], pred_overlayed_imgs_list[batch_idx], true_pred_overlayed_imgs_list[batch_idx], images_list[batch_idx], sigma_list[batch_idx] ,flows_list,
                            rand_inds, kalman_flag=kalman_flag, flow_flag=args.flow_flag)          
        else:            
            log_tensorboard(writer, global_step, 'test', true_mask_list[batch_idx], pred_mask_list[batch_idx], true_overlayed_imgs_list[batch_idx], pred_overlayed_imgs_list[batch_idx], true_pred_overlayed_imgs_list[batch_idx], images_list[batch_idx], None ,flows_list,
                            rand_inds, kalman_flag=kalman_flag, flow_flag=args.flow_flag) 


        # iou_0, tp_by_all_positive_0, _ = iou(pred_mask_list[batch_idx:batch_idx+1], true_mask_list[batch_idx:batch_idx+1], kalman_flag=kalman_flag)
        iou_0, tp_by_all_positive_0 = None, None #iou_0.item(), tp_by_all_positive_0.item()
        
    else:
        # rand_inds = range(imgs.shape[0])
        # import ipdb; ipdb.set_trace()
        rand_inds= random.sample(range(len(images_list)), 20)
        log_tensorboard(writer, global_step, 'test', true_mask_list, pred_mask_list, true_overlayed_imgs_list, pred_overlayed_imgs_list, true_pred_overlayed_imgs_list, images_list, None ,flows_list,
                        rand_inds, kalman_flag, flow_flag=args.flow_flag)        
        iou_0, tp_by_all_positive_0 = None, None    
    
    if vector_flag:
        return epoch_loss, None, None, None, None, None, None, None

    return epoch_loss, avg_iou, iou_0, avg_tp_by_all_positive, tp_by_all_positive_0, avg_dice_score, avg_precision, avg_recall

def train_net(net, args, **kwargs):
    # import ipdb; ipdb.set_trace()
    device = kwargs['device']
    batch_size = args.batch_size
    lr = args.lr
    saved_data_file = args.saved_data_file
    n_flow = args.n_flow
    flow_history_flag = args.flow_history_flag
    flow_flag = args.flow_flag
    kalman_flag = args.kalman_flag
    vector_flag = args.vector_flag
    
    PARENT_FOLDER_TRAIN, LIST_OF_DATASETS_TRAIN, PARENT_FOLDER_TEST, LIST_OF_DATASETS_TEST, IMAGE_LOC, MASK_LOC = get_list_train_test_data(args.data_type)

    if not args.use_saved_data:
        # images_tensor, flows_tensor, needle_masks_tensor, masks_tensor, flow_concats_tensor = get_data(net.temporal_in_channel)
        if kalman_flag:
            if vector_flag:
                print("vector kalman filter")
                train_data = get_data_dict_kalman_vec(PARENT_FOLDER_TRAIN, LIST_OF_DATASETS_TRAIN, IMAGE_LOC, MASK_LOC, saved_data_file, type='train', traj_len=args.traj_len, data_type=args.data_type)
                test_data = get_data_dict_kalman_vec(PARENT_FOLDER_TEST, LIST_OF_DATASETS_TEST, IMAGE_LOC, MASK_LOC, saved_data_file, type='test', traj_len=args.traj_len, data_type=args.data_type)
            else:
                train_data = get_data_dict_kalman(PARENT_FOLDER_TRAIN, LIST_OF_DATASETS_TRAIN, IMAGE_LOC, MASK_LOC, saved_data_file, type='train', traj_len=args.traj_len, data_type=args.data_type)
                test_data = get_data_dict_kalman(PARENT_FOLDER_TEST, LIST_OF_DATASETS_TEST, IMAGE_LOC, MASK_LOC, saved_data_file, type='test', traj_len=args.traj_len, data_type=args.data_type)
        else:
            train_data, max_flow, min_flow = get_data_dict_history(n_flow, PARENT_FOLDER_TRAIN, LIST_OF_DATASETS_TRAIN, IMAGE_LOC, MASK_LOC, saved_data_file,'train', args.data_type, flow_history_flag, flow_flag)
            test_data, _, _ = get_data_dict_history(n_flow, PARENT_FOLDER_TEST, LIST_OF_DATASETS_TEST, IMAGE_LOC, MASK_LOC, saved_data_file,'test', args.data_type, flow_history_flag, flow_flag, max_flow, min_flow)
        # _data, max_flow, min_flow = get_data_dict_history(n_flow, LIST_OF_DATASETS_TRAIN, PARENT_FOLDER_TRAIN, saved_data_file,'all',flow_history_flag)
        
    else:
        train_data = torch.load(os.path.join('saved_data', saved_data_file, 'train.pt')) #torch.load('saved_data/3/train.pt')
        test_data = torch.load(os.path.join('saved_data', saved_data_file, 'test.pt')) #torch.load('saved_data/3/test.pt')

    # import ipdb; ipdb.set_trace()
    #! Splitting train data into train and test | add flag for this 
    # _len_data  = len(train_data)
    # train_data, test_data = torch.utils.data.random_split(train_data, [int(0.75*_len_data), _len_data-int(0.75*_len_data)], generator=torch.Generator().manual_seed(42))
    
    #* wrap data in dataloader 
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data = DataLoader(test_data, batch_size=2*batch_size, shuffle=False, drop_last=True)
    

    conv_layers = args.conv_layers
    iter = kwargs['iter'] #args.iter
    if args.task == 'train':
        dir_checkpoint = os.path.join('checkpoints/exp', str(iter) + '_conv_layers_{}_nflow_{}'.format(conv_layers,n_flow))
        if not os.path.exists(dir_checkpoint):
            os.makedirs(dir_checkpoint) 
    else:
        dir_checkpoint = os.path.join('checkpoints/exp', str(iter) + '_conv_layers_{}_nflow_{}'.format(conv_layers,n_flow))

        
    n_train = len(train_data)*batch_size 
    n_val = len(test_data)*batch_size 
    # train, val = random_split(dataset, [n_train, n_val])

    #!PARAMS for LOGGING in TENSORBOARD
    patience = 5
    tensorboard_flag = args.tensorboard
    if tensorboard_flag:
        if args.task == 'val':
            writer = SummaryWriter(comment='_VAL_LR_{}_BS_{}_patience_{}_nflow_{}_conv_layers_{}_iter_{}'.format(lr, batch_size, patience, n_flow, conv_layers, iter))
        else:
            writer = SummaryWriter(comment='LR_{}_BS_{}_patience_{}_nflow_{}_conv_layers_{}_iter_{}'.format(lr, batch_size, patience, n_flow, conv_layers, iter))

    global_step = 0
    eval_step = args.eval_steps #! eval after 100 optimization steps 
    if tensorboard_flag:
        logging.info(f'''Starting training:
            Epochs:          {args.epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {args.store_weights}
            Device:          {device.type}
            '''
            )

    #! initialize optimizer
    learned_flow = args.learned_flow
    if learned_flow:         
        # put flow network params on different lr
        optimizer = optim.AdamW([{'params':get_weights(net.UNet.named_parameters()) + get_weights(net.spatial_conv.named_parameters()) +\
                                     get_weights(net.temporal_conv.named_parameters())}, 
                                    {'params':get_weights(net.flownet.named_parameters()), 'lr':lr*1e-1}], lr=lr, weight_decay=1e-5) #momentum=0.9
    else:
        # optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-5)
        # ipdb.set_trace()
        if kwargs['encoder_type'] in ['resnet18', 'resnet34', 'hybrid']:
            if kalman_flag:
                optimizer = optim.AdamW([{'params': get_weights(net.UNet.kalman_model.named_parameters()) + get_weights(net.UNet.up2.named_parameters())
                                        + get_weights(net.UNet.up3.named_parameters()) + get_weights(net.UNet.up4.named_parameters()) + get_weights(net.UNet.outc.named_parameters())
                                        , 'lr':lr}, 
                                        {'params':get_weights(net.UNet.encoder.named_parameters()), 'lr':1e-5}], weight_decay=5e-5)
            else:
                optimizer = optim.AdamW([{'params': get_weights(net.UNet.up2.named_parameters())
                                        + get_weights(net.UNet.up3.named_parameters()) + get_weights(net.UNet.up4.named_parameters()) + get_weights(net.UNet.outc.named_parameters())
                                        , 'lr':lr}, 
                                        {'params':get_weights(net.UNet.encoder.named_parameters()), 'lr':1e-5}], weight_decay=5e-5)            
        else:
            optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-5)
        
        # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)
    
    #! initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.7)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # set criteria based on number of classes 
    n_classes = kwargs['n_classes']
    if vector_flag:
        criterion = nn.MSELoss()
    else:
        if n_classes > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(100)) #pos_weight=150
            criterion = nn.BCEWithLogitsLoss() 
            constrastive_criterion = ContrastiveLoss()

    #! TRAINING STARTS HERE
    min_val_score = float('inf')
    max_eval_dice = 0.
    max_eval_tp_by_all_positive = 0
    for epoch in range(args.epochs):
        net.train()

        epoch_loss = []        
        # epoch_classification_loss = []        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for i,data in enumerate(train_data):                
                mask_type = torch.float32 #if n_classes == 1 else torch.long
                imgs = data['images'] 
                true_masks = data['needle_masks']
                if kalman_flag:
                    if vector_flag:
                        needle_params = data['needle_params'] #x_start, y_start, needle_angle, needle_length, x_tip, y_tip
                        needle_params = needle_params.to(device=device, dtype=mask_type)
                        true_masks_new = data['needle_masks_new']
                    flows = None 
                    imgs_prev = None
                elif flow_flag:
                    imgs_prev = data['images_prev']
                    flows = data['flow_concats'] 
                    imgs_prev = imgs_prev.to(device=device)
                    flows = flows.to(device=device, dtype=mask_type)/255 
                else:
                    flows = None 
                    if args.n_flow > 1:
                        imgs_prev = data['images_prev']
                        imgs_prev = imgs_prev.to(device=device, dtype=mask_type)
                    else:
                        imgs_prev = None
                # import ipdb; ipdb.set_trace()
                # print("true masks max = " , torch.max(true_masks))
                imgs = imgs.to(device=device, dtype=mask_type)
                true_masks = true_masks.to(device=device, dtype=mask_type)                
                #! compute predictions
                masks_pred, mean, sigma, flow = net(imgs,flows,imgs_prev) # flow in form of image 
                
                #! compute loss
                if vector_flag:
                    # scale_needle_param = torch.tensor([1/256, 1/256, 1/(2*pi), 1/(256*(2**0.5))]).to(device)
                    scale_needle_param = torch.tensor([256, 256, (2*pi), 256]).to(device)
                    masks_pred  = masks_pred[:,:,:4]*scale_needle_param # 
                    masks_true = needle_params[:,:,:4] # *scale_needle_param 
                    loss = criterion(masks_pred, masks_true)
                    x_start_error = torch.mean(torch.abs(needle_params[:,:,0] - masks_pred[:,:,0]))
                    y_start_error = torch.mean(torch.abs(needle_params[:,:,1] - masks_pred[:,:,1]))
                    angle_error = torch.mean(torch.abs(needle_params[:,:,2] - masks_pred[:,:,2]))
                    length_error = torch.mean(torch.abs(needle_params[:,:,3] - masks_pred[:,:,3]))
                else:
                    loss =  criterion(masks_pred, true_masks)
                # loss = sigmoid_focal_loss(masks_pred, true_masks, device)
                
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1) ## increasead from 0.1 to 0.5 BE MIndFUL OF this 
                optimizer.step()

                epoch_loss.append(loss.item())

                if tensorboard_flag:
                    writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss(batch)': loss.item(), 'lr':optimizer.param_groups[0]['lr']})
                # push data back to cpu
                imgs.to('cpu')
                true_masks.to('cpu')

                pbar.update(imgs.shape[0])
                global_step += 1           
                #! EVALUATE
                if global_step % eval_step == 0:                    
                    # for tag, value in net.named_parameters():
                    #     tag = tag.replace('.', '/')                                                
                    #     if value == None or value.grad == None:                            
                    #         pass
                    #     else:
                    #         writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    #         writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)                    
                    
                    net.eval()
                    val_score, avg_iou_eval, iou_0, avg_tp_by_all_positive, tp_by_all_positive_0, avg_dice_score, avg_precision, avg_recall = \
                                                eval_net(writer, global_step, 
                                                  net, test_data, n_classes, criterion, device, args )

                    if avg_dice_score is not None and avg_dice_score > max_eval_dice:
                        max_eval_dice = avg_dice_score
                        if args.store_weights:
                            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_best_dice.pth'))
                    # if avg_tp_by_all_positive > max_eval_tp_by_all_positive:
                    #     max_eval_tp_by_all_positive = avg_tp_by_all_positive
                    #     if args.store_weights:
                    #         torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_best_tp_by_all_positive.pth'))
                    if val_score < min_val_score:
                        min_val_score = val_score
                        if args.store_weights:
                            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_best_val_score.pth'))
                    
                    #* save weight 
                    # if args.store_weights:
                    #     torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_{}.pth'.format(global_step)))
                   
                    scheduler.step(val_score)                        
                    # scheduler.step()                        
                    
                    net.train()
                    logging.info('Validation Loss: {}'.format(val_score))
                    # log test scalars | #? NOTE THAT _avg implies over all data
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('Loss/test', val_score, global_step)      
                    if not vector_flag:              
                        writer.add_scalar('Scalars/test_tp_by_all_positive_avg', avg_tp_by_all_positive, global_step)
                        writer.add_scalar('Scalars/test_IOUAvg', avg_iou_eval, global_step)
                        writer.add_scalar('Scalars/test_precision_avg', avg_precision, global_step)
                        writer.add_scalar('Scalars/test_recall_avg', avg_recall, global_step)
                        writer.add_scalar('Scalars/test_dice_score_avg', avg_dice_score, global_step)
                        if iou_0 is not None:
                            writer.add_scalar('IOU/test', iou_0, global_step)
                            writer.add_scalar('tp_by_all_positive/test', tp_by_all_positive_0, global_step)
                    
                    if kalman_flag:
                        # only log from first batch as data in video form
                        imgs = imgs[0]
                        masks_pred = masks_pred[0]
                        true_masks = true_masks[0]
                        if vector_flag:
                            needle_param = needle_params[0]
                            true_masks_new = true_masks_new[0]

                    writer.add_images('train/images', imgs, global_step)

                    if flow_flag:
                        writer.add_images('train/flows', flows[:,0:1,:,:], global_step)

                    if not vector_flag:
                        masks_pred_sigmoid = torch.sigmoid(masks_pred) 
                        masks_pred_threshold = (masks_pred_sigmoid > 0.5).float()                
                        # find IOU of training data at batch 0, data size is SxCxHxW                    
                        train_iou, train_tp_by_all_positive, _ = iou(masks_pred_threshold.unsqueeze(0), true_masks.unsqueeze(0), kalman_flag=kalman_flag)
                        train_precision, train_recall = precision_recall(masks_pred_threshold.unsqueeze(0), true_masks.unsqueeze(0), kalman_flag=kalman_flag)
                        train_dice_score = dice_score(masks_pred_threshold.unsqueeze(0), true_masks.unsqueeze(0), kalman_flag=kalman_flag)
                        writer.add_scalar('Scalars/train_tp_by_all_positive_avg', train_tp_by_all_positive, global_step)
                        writer.add_scalar('Scalars/train_IOUAvg', train_iou, global_step)
                        writer.add_scalar('Scalars/train_precision_avg', train_precision, global_step)
                        writer.add_scalar('Scalars/train_recall_avg', train_recall, global_step)
                        writer.add_scalar('Scalars/train_dice_score_avg', train_dice_score, global_step)                    
                        # writer.add_images('train/mask_pred_sigmoid', masks_pred_sigmoid, global_step)
                        writer.add_images('train/true_overlayed' ,torch.concat([imgs, 0.6*imgs + 0.4*true_masks, imgs], dim = 1), global_step)
                        writer.add_images('train/pred_overlayed' ,torch.concat([imgs, 0.6*imgs + 0.4*masks_pred_threshold, imgs], dim = 1), global_step)
                        writer.add_images('train/combined_overlayed' ,torch.concat([masks_pred_threshold, torch.zeros_like(imgs) , true_masks], dim = 1), global_step)
                        if sigma is not None:
                            mean, sigma = mean[0], sigma[0] 
                            mean_sigmoid = torch.sigmoid(mean)
                            mean_threshold = (mean_sigmoid > 0.5).float()
                            # for plotting colored 
                            # im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)

                            sigma = sigma/torch.max(sigma) if torch.max(sigma) > 1 else sigma
                            writer.add_images('train/mean', mean_threshold, global_step)
                            writer.add_images('train/uncertainty', sigma, global_step)
                    else:
                        #? 0: x_start, 1: y_start, 2: needle_angle, 3: needle_length
                        #draw circles based on x_start, y_start
                        masks = torch.zeros_like(true_masks, dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                        masks_pred = masks_pred.detach().cpu().numpy()
                        masks_list = []                        
                        for kk in range(masks.shape[0]): 
                            x_tip = masks_pred[kk,0] + np.cos(masks_pred[kk,2])*masks_pred[kk,3]
                            y_tip = masks_pred[kk,1] + np.sin(masks_pred[kk,2])*masks_pred[kk,3]
                            # masks_ = cv2.circle(masks[kk], (int(round(masks_pred[kk,1])), int(round(masks_pred[kk,0]))), 3, (0, 0, 255), 2)
                            # masks_ = cv2.circle(masks_, (int(round(y_tip)), int(round(x_tip))), 3, (0, 255, 0), 2)
                            masks_ = cv2.line(np.concatenate([masks[kk]]*3, axis=-1), (int(round(masks_pred[kk,1])), int(round(masks_pred[kk,0]))), (int(round(y_tip)), int(round(x_tip))), (0,0,255), 5)
                            masks_list.append(masks_)
                        
                        masks_list = torch.from_numpy(np.stack(masks_list, axis=0)).permute(0,3,1,2)
                        writer.add_images('train/mask_pred_overlayed', masks_list + true_masks_new, global_step)
                        writer.add_images('train/mask_pred', masks_list, global_step)
                        writer.add_images('train/mask_true', true_masks, global_step)
                        writer.add_images('train/mask_true_new', true_masks_new, global_step)
                        writer.add_scalar('train/x_start_error', x_start_error, global_step)
                        writer.add_scalar('train/y_start_error', y_start_error, global_step)
                        writer.add_scalar('train/length_error', length_error, global_step)
                        writer.add_scalar('train/angle_error', angle_error, global_step)


            # log avg epoch_loss
            print("epoch loss : ", np.mean(epoch_loss))
            writer.add_scalar('Loss/Epoch_train', np.mean(epoch_loss), epoch)
        if args.store_weights:
            try:
                # os.mkdir(dir_checkpoint)
                os.makedirs(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            # save last and best epoch based on eval loss             
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_last.pth'))

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--eval_steps', metavar='ES', type=float, nargs='?', default=100,
                        help='Evaluation Steps', dest='eval_steps')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--use_saved_data', action='store_true' ,default=False,
                        help='using saved data')
    parser.add_argument('--task', type=str, default = 'train')
    parser.add_argument('--saved_data_file', type=str, default='4')
    parser.add_argument('--iter', type=str, default='0', help='sets the tensorbaord and weight name as well')
    parser.add_argument('--val_weights', type=str, default='0')
    parser.add_argument('--pure_images', action='store_true', default=False, help='to use only images and no flow')
    parser.add_argument('--n_flow', type=int, default=1)
    parser.add_argument('--conv_layers', type=int, default=1, choices=[1,2], 
                        help='number of conv layers in each level of encoder decoder')
    parser.add_argument('--multi_attn', type=int, default=0)
    parser.add_argument('--flow_history_flag', action='store_true', default=False, help='using past flows')
    parser.add_argument('--learned_flow', action='store_true', default=False, help='using learning based flow')
    parser.add_argument('--late_fusion', action='store_true', default=False, help='fusing flow and image feature maps at bottleneck layer')
    parser.add_argument('--classification_flag', action='store_true', default=False, help='adding classification loss: needle present or not')
    parser.add_argument('--attention_flag', action='store_true', default=False, help='using attention based UNet')
    parser.add_argument('--store_weights', action='store_true', default = False, help='make true when weights need to be saved')
    parser.add_argument('--kalman_flag', action='store_true', default = False, help='make true when using kalman filtering')
    parser.add_argument('--tensorboard', action='store_true', default = False, help='logging on tensorboard')    
    parser.add_argument('--freeze_weights' , action='store_true', default=False, help='freezing unet weights and training motion model')
    parser.add_argument('--traj_len', type=int, default=50, help='length of video sequence for kalman filter')
    parser.add_argument('--eval' , action='store_true', default=False, help='Run Validation for video')
    parser.add_argument('--flow_flag' , action='store_true', default=False, help='flag whether to use flow or not')
    parser.add_argument('--gauss_flag', action='store_true', default=False, help='flag for using gaussian distribution inside Kalman')
    parser.add_argument('--transformer_flag', action='store_true', default=False, help='flag for using tranformer for kalman gain')
    parser.add_argument('--vector_flag', action='store_true', default=False, help='vector kalman filter')
    parser.add_argument('--process_model_flag', action='store_true', default=False, help='generate next state using just process model')
    parser.add_argument('--high_res_flag', action='store_true', default=False, help='generate next state using just process model')
    parser.add_argument('--data_type', type=str, default='DARPA', help='one of the following: [DARPA, UPMC, BlueGel]')
    # parser.add_argument('--recurrence_type', type=str, default='lstm', help='choose one of the following when kalman_flag is chosen: [lstm, conv_lstm, conv_kalman, conv_kalman_gauss]')

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

    batch_size = args.batch_size 
    encoder_type_list = ['vanilla','resnet18','hybrid'] #['resnet18', 'hybrid','vanilla']
    for encoder in encoder_type_list:
        print("encoder = ", encoder)
        if not args.late_fusion: #args.late_fusion is not used, can remove this if statement
            scale = 1

            kwargs = {}
            kwargs['spatial_in_channel'] = 1
            kwargs['out_channels'] = 16
            kwargs['n_classes'] = 1
            kwargs['n_depth'] = 4
            kwargs['bilinear'] = False
            kwargs['unet_channel_start'] = 64  
            kwargs['kf_channels'] = 32
            kwargs['encoder_type'] = encoder # choose from ['vanilla', 'resnet18', 'resnet34', 'hybrid']
            kwargs['recurrence_type'] = 'conv_kalman' # choose from ['lstm', 'conv_lstm', 'conv_kalman', 'conv_kalman_gauss']
            kwargs['iter'] = encoder + '_' + args.iter 

            net = TwoStreamUNet(args,device=device, **kwargs)
    
            # freeze flownet parameters
            # ipdb.set_trace()
            # for tag, value in net.named_parameters():
            #     print("tag = " , tag , "  " , value.requires_grad)
        net.to(device=device)
        # faster convolutions, but more memory
        # cudnn.benchmark = True

        if args.load:
            # to check weights : net.UNet.inc.weight
            weight_loc = os.path.join('checkpoints/exp', args.load,'CP_best_val_score.pth')
            # if not args.kalman_flag:
            #     net.load_state_dict(torch.load(args.load, map_location=device))
            # else:
            #     UNet_weights = torch.load(args.load, map_location=device)
            #     for name, params in net.named_parameters():
            #         if 'kalman' not in name:
            #             params = UNet_weights[name]
                        # params.requires_grad = False 
            net.load_state_dict(torch.load(weight_loc, map_location=device))        
            # print(net.UNet.inc.double_conv[0].weight[0])
            logging.info(f'Model loaded from {args.load}')


        try:
            train_net(net= net, args= args, device=device, **kwargs)
            del net
        except KeyboardInterrupt:        
            pass
            # torch.save(net.state_dict(), os.path.join(dir_checkpoint,'INTERRUPTED.pth'))
            # logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
    # else:
    #     eval_net(net, data, n_classes, criterion, device)