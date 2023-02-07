import argparse
from asyncore import write
from cmath import e
from email.policy import default
from glob import glob
import logging
import os
from re import L
import sys
from types import new_class

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
from utils_two_stream_unet import *
from rishabh import *

from torch.utils.data import DataLoader, random_split


def log_tensorboard(writer, global_step, type, true_masks, masks_pred, true_overlayed_imgs, pred_overlayed_imgs, imgs, last_encoded_feature, flows,
                    rand_inds, kalman_flag=False, true_pred_overlayed_imgs=None):
    '''
    logger for only validation
    '''
    # choose random index to check | all   
    if not kalman_flag:
        true_masks_ = true_masks
        masks_pred_ = masks_pred
        true_overlayed_imgs_ = true_overlayed_imgs
        pred_overlayed_imgs_ = pred_overlayed_imgs
        true_pred_overlayed_imgs_ = true_pred_overlayed_imgs
        imgs_ = imgs
    else:
        true_masks_ = true_masks[rand_inds]
        masks_pred_ = masks_pred[rand_inds]
        true_overlayed_imgs_ = true_overlayed_imgs[rand_inds]
        pred_overlayed_imgs_ = pred_overlayed_imgs[rand_inds]
        true_pred_overlayed_imgs_ = true_pred_overlayed_imgs[rand_inds]
        imgs_ = imgs[rand_inds]
    if not kalman_flag:
        pass
        # flows_rand_inds = flows[rand_inds]
        # writer.add_images(type+'/flows', flows_rand_inds[:,0:1,:,:], global_step)
    
    writer.add_images(type+'/images', imgs_, global_step)
    writer.add_images(type+'/mask_pred_sigmoid', masks_pred_, global_step)
    writer.add_images(type+'/true_overlayed', true_overlayed_imgs_, global_step)
    writer.add_images(type+'/pred_overlayed', pred_overlayed_imgs_, global_step)
    if last_encoded_feature is not None:
        writer.add_images(type+'_features'+'/last_encoded_feature', last_encoded_feature.unsqueeze(1), global_step)
    if true_pred_overlayed_imgs is not None:
        writer.add_images(type + '/true_pred_combined_overlay', true_pred_overlayed_imgs_, global_step)


def eval_net_stacked(writer, global_step, net, test_data, n_classes, criterion, device, args):
    '''
    #! EVALUATE
    '''
    # images, flows, needle_masks, masks, flow_concats  = get_dict_vals(data)
    kalman_flag = False
    n_eval = len(test_data)    
    epoch_loss = 0
    true_mask_list, pred_mask_list, loss_list = [], [], []
    true_overlayed_imgs_list, pred_overlayed_imgs_list, true_pred_overlayed_imgs_list = [], [], []
    images_list = []
    flows_list = []
    iou_list, dsc_list, precision_list, recall_list = [], [], [], []        
    with torch.no_grad():
        # i = 0
        for j, data in enumerate(test_data):
            # import ipdb; ipdb.set_trace()
            print("j=",j)
            new_video_flag = True                 
            mask_type = torch.float32 #if n_classes == 1 else torch.long                
            imgs = data['images'] 
            true_masks = data['needle_masks']
            if args.flow_flag:
                imgs_prev = data['images_prev']
                flows = data['flow_concats']                     
                imgs_prev = imgs_prev.to(device=device)
                flows = flows.to(device=device, dtype=mask_type).unsqueeze(0) / 255                    
                imgs = imgs.to(device=device, dtype=mask_type).unsqueeze(0).unsqueeze(0)
                true_masks = true_masks.to(device=device, dtype=mask_type).unsqueeze(0).unsqueeze(0)
            else:
                flows = None #data['flow_concats']                      
                if args.n_flow > 1:
                    imgs_prev = data['images_prev']
                    imgs_prev = imgs_prev.to(device=device, dtype=mask_type).unsqueeze(0)
                else:
                    imgs_prev = None #data['images_prev']
                
                imgs = imgs.to(device=device, dtype=mask_type).unsqueeze(0)
                true_masks = true_masks.to(device=device, dtype=mask_type).unsqueeze(0)
            
            # ipdb.set_trace()
            masks_pred, _, _, _ = net(imgs, flow=flows, image_prev=imgs_prev, new_video_flag=new_video_flag)
            new_video_flag = False            
            
            # masks_pred, x_spatial, x_temporal = net(imgs,flows_) # flow in form of x, y matrices

            loss = criterion(masks_pred, true_masks) # bce with logits already has sigmoid 
            # loss = sigmoid_focal_loss(masks_pred, true_masks, device)
            masks_pred = torch.sigmoid(masks_pred)
            masks_pred_threshold = (masks_pred > 0.5).float() # additional filtering ? 
            
            #* compute IOU and keep a store            
            iou_val, _, _ = iou(masks_pred_threshold, true_masks, kalman_flag=kalman_flag, eval=True)#.item()
            if iou_val is not None:
                iou_val = iou_val.item()
                iou_list.append(iou_val)
            
            dice_score_val = dice_score(masks_pred_threshold, true_masks, kalman_flag=kalman_flag, eval=True)
            if dice_score_val is not None:
                dice_score_val = dice_score_val.item()
                dsc_list.append(dice_score_val)       

            precision, recall = precision_recall(masks_pred_threshold, true_masks, kalman_flag=kalman_flag, eval=True)
            if precision is not None:          
                precision, recall = precision.item(), recall.item()
                precision_list.append(precision); recall_list.append(recall); 
                # print("precision=" , precision, " recall=",recall)
            
            imgs = imgs.to('cpu') 
            true_masks = true_masks.to('cpu') 
        
            masks_pred = masks_pred.to( 'cpu')
            masks_pred_threshold = masks_pred_threshold.to( 'cpu')                

            images_list.append(imgs)
            loss_list.append(loss.item())

            true_mask_list.append(true_masks)
            pred_mask_list.append(masks_pred)
            # overlayed_imgs_list.append(torch.concat([imgs.to('cpu'), true_masks, masks_pred_],dim = 1))
            alpha = 0.8
            true_overlayed_img = torch.concat([imgs, alpha*imgs + (1-alpha)*true_masks, imgs], dim = -3) 
            pred_overlayed_img = torch.concat([imgs, alpha*imgs + (1-alpha)*masks_pred_threshold, imgs], dim = -3) 
            true_pred_overlayed_img = torch.concat([masks_pred_threshold, torch.zeros_like(imgs), true_masks], dim = -3) 
                    
            true_overlayed_imgs_list.append(true_overlayed_img)
            pred_overlayed_imgs_list.append(pred_overlayed_img)
            true_pred_overlayed_imgs_list.append(true_pred_overlayed_img)
            rand_inds = None

    epoch_loss = np.mean(loss_list)
    images_list = torch.concat(images_list, dim=0)
    true_mask_list = torch.concat(true_mask_list, dim=0)        
    # if not kalman_flag:
        # flows_list = torch.concat(flows_list, dim=0)
    pred_mask_list = torch.concat(pred_mask_list, dim=0)
    true_overlayed_imgs_list = torch.concat(true_overlayed_imgs_list, dim=0)
    pred_overlayed_imgs_list = torch.concat(pred_overlayed_imgs_list, dim=0)
    true_pred_overlayed_imgs_list = torch.concat(true_pred_overlayed_imgs_list, dim=0)
    
    print("len of precions={}, len of DSC={}".format(len(precision_list), len(dsc_list)))
    avg_iou = np.mean(iou_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_dsc = np.mean(dsc_list)
    print("iou={}, precision={}, recall={}, DSC={}".format(avg_iou, avg_precision, avg_recall, avg_dsc))


def find_needle_params(mask):
    ipdb.set_trace()
    
    idx = np.where(mask == 1)

    if len(idx[0]) == 0:
        return mask, None

    idx_y_min = np.where(idx[1] == np.min(idx[1]))
    idx_y_max = np.where(idx[1] == np.max(idx[1]))

    print("idx_y_min = ", idx_y_min)
    print("idx_y_max = ", idx_y_max)


    x_max = np.max(idx[0][idx_y_max])
    x_min = np.min(idx[0][idx_y_min])

    y_min = np.min(idx[1]) 
    y_max = np.max(idx[1])

    # swap max and min if x_max < x_min 
    if x_min > x_max:
        # reverse the notation
        x_min, x_max = x_max, x_min
        y_min, y_max = y_max, y_min
        
    print("min x = {}, min y = {}".format(x_min, y_min))
    print("max x = {}, max y = {}".format(x_max, y_max))
    # make circle on the min and max : image will be 3 channel     
    mask_new = cv2.circle(mask, (y_min, x_min), 3, (0, 0, 255), 2)
    mask_new = cv2.circle(mask, (y_max, x_max), 3, (0, 255, 0), 2)

    needle_length = ((x_min - x_max)**2 + (y_min - y_max)**2)**(0.5)
    needle_angle = np.arctan2(y_max-y_min, x_max-x_min)

    x_start, y_start = x_min, y_min
    x_tip, y_tip = x_max, y_max

    return mask_new, [x_start, y_start, needle_angle, needle_length, x_tip, y_tip]

def eval_net(writer, global_step, net, test_data, n_classes, criterion, device, args):
    '''
    #! EVALUATE
    '''
    kalman_flag = args.kalman_flag
    # images, flows, needle_masks, masks, flow_concats  = get_dict_vals(data)
    n_eval = len(test_data)    
    epoch_loss = 0
    true_mask_list, pred_mask_list, loss_list = [], [], []
    true_overlayed_imgs_list, pred_overlayed_imgs_list, true_pred_overlayed_imgs_list = [], [], []
    images_list = []
    flows_list = []
    iou_list, dsc_list, precision_list, recall_list = [], [], [], []        

    with torch.no_grad():
        # i = 0
        # ipdb.set_trace()
        for j, data in enumerate(test_data):
            # import ipdb; ipdb.set_trace()
            print("j=",j)
            new_video_flag = True
            # import ipdb; ipdb.set_trace()
            for i in range(len(data['images'])):            
                # print("i= ", i)
                mask_type = torch.float32 #if n_classes == 1 else torch.long                
                imgs = data['images'][i] 
                true_masks = data['needle_masks'][i] 
                if kalman_flag:
                    flows = None 
                    imgs_prev = None 
                    imgs = imgs.to(device=device, dtype=mask_type).unsqueeze(0).unsqueeze(0)
                    true_masks = true_masks.to(device=device, dtype=mask_type).unsqueeze(0).unsqueeze(0)
                    flows, imgs_prev = None , None
                elif args.flow_flag:
                    imgs_prev = data['images_prev']
                    flows = data['flow_concats']                     
                    imgs_prev = imgs_prev.to(device=device)
                    flows = flows.to(device=device, dtype=mask_type).unsqueeze(0) / 255                    
                    imgs = imgs.to(device=device, dtype=mask_type).unsqueeze(0).unsqueeze(0)
                    true_masks = true_masks.to(device=device, dtype=mask_type).unsqueeze(0).unsqueeze(0)
                else:
                    flows = None #data['flow_concats']                      
                    if args.n_flow > 1:
                        imgs_prev = data['images_prev']
                        imgs_prev = imgs_prev.to(device=device, dtype=mask_type)
                    else:
                        imgs_prev = None #data['images_prev']
                    
                    imgs = imgs.to(device=device, dtype=mask_type).unsqueeze(0)
                    true_masks = true_masks.to(device=device, dtype=mask_type).unsqueeze(0)
                
                masks_pred, masks_pred_mean, _, _, = net(imgs, flow=flows, image_prev=imgs_prev, new_video_flag=new_video_flag)
                new_video_flag = False
                # below is false for lstm + unet 
                if (i+1)%args.traj_len == 0:
                    new_video_flag = True
                # masks_pred, x_spatial, x_temporal = net(imgs,flows_) # flow in form of x, y matrices
                if args.kalman_flag and args.gauss_flag:
                    masks_pred = masks_pred_mean
                
                loss = criterion(masks_pred, true_masks) # bce with logits already has sigmoid 
                # loss = sigmoid_focal_loss(masks_pred, true_masks, device)
                masks_pred = torch.sigmoid(masks_pred)
                masks_pred_threshold = (masks_pred > 0.5).float() # additional filtering ? 
                
                # if args.needle_params:
                #     true_masks_ = true_masks[0][0][0].cpu().numpy()
                #     masks_pred_threshold_ = masks_pred_threshold[0][0][0].cpu().numpy()
                #     true_mask_new, needle_params_true = find_needle_params(true_masks_)
                #     masks_pred_new, needle_params_pred = find_needle_params(masks_pred_threshold_)
                #     if needle_params_true == None:
                #         print("no needle")
                #     else:
                #         # [x_start, y_start, needle_angle, needle_length, x_tip, y_tip]
                #         # write images here 
                #         print("needle params")
                #         print(needle_params_true)
                #         print(needle_params_pred)


                #* compute IOU and keep a store            
                iou_val, _, _ = iou(masks_pred_threshold, true_masks, kalman_flag=kalman_flag, eval=True)#.item()
                if iou_val is not None:
                    iou_val = iou_val.item()
                    iou_list.append(iou_val)
                
                dice_score_val = dice_score(masks_pred_threshold, true_masks, kalman_flag=kalman_flag, eval=True)
                if dice_score_val is not None:
                    dice_score_val = dice_score_val.item()
                    dsc_list.append(dice_score_val)       

                precision, recall = precision_recall(masks_pred_threshold, true_masks, kalman_flag=kalman_flag, eval=True)
                if precision is not None:          
                    precision, recall = precision.item(), recall.item()
                    precision_list.append(precision); recall_list.append(recall); 
                    # print("precision=" , precision, " recall=",recall)
                
                imgs = imgs.to('cpu') 
                true_masks = true_masks.to('cpu') 
                # if not kalman_flag:
                #     flows = flows.to('cpu') 
                #     flows_list.append(flows)
                # else:
                #     pass

                masks_pred = masks_pred.to( 'cpu')
                masks_pred_threshold = masks_pred_threshold.to( 'cpu')                

                images_list.append(imgs)
                loss_list.append(loss.item())

                true_mask_list.append(true_masks)
                pred_mask_list.append(masks_pred)
                # overlayed_imgs_list.append(torch.concat([imgs.to('cpu'), true_masks, masks_pred_],dim = 1))
                alpha = 0.8
                true_overlayed_img = torch.concat([imgs, alpha*imgs + (1-alpha)*true_masks, imgs], dim = -3) 
                pred_overlayed_img = torch.concat([imgs, alpha*imgs + (1-alpha)*masks_pred_threshold, imgs], dim = -3) 
                true_pred_overlayed_img = torch.concat([masks_pred_threshold, torch.zeros_like(imgs), true_masks], dim = -3) 
                        
                true_overlayed_imgs_list.append(true_overlayed_img)
                pred_overlayed_imgs_list.append(pred_overlayed_img)
                true_pred_overlayed_imgs_list.append(true_pred_overlayed_img)
                rand_inds = None
                if kalman_flag:
                    rand_inds = 0 #range(imgs.shape[1]) # all images across sequences
                    # import ipdb; ipdb.set_trace()
                if writer is not None:
                    log_tensorboard(writer, i + j*len(data['images']), 'test', true_masks, masks_pred, true_overlayed_img, pred_overlayed_img, imgs, None ,None,
                                    rand_inds, kalman_flag, true_pred_overlayed_img)              

                # writer.add_scalar('test/dsc_score', dice_score , i)
                # iou_0, intersection_0 = iou(masks_pred_threshold[0:1], true_masks[0:1], kalman_flag=kalman_flag) #.item()
                # writer.add_scalar('test/IOU_data_plotted' , iou_0)
                # writer.add_scalar('test/intersection_data_plotted' , intersection_0)
                    
            # if j >= 0:
            #     break
        epoch_loss = np.mean(loss_list)
        images_list = torch.concat(images_list, dim=0)
        true_mask_list = torch.concat(true_mask_list, dim=0)        
        # if not kalman_flag:
            # flows_list = torch.concat(flows_list, dim=0)
        pred_mask_list = torch.concat(pred_mask_list, dim=0)
        true_overlayed_imgs_list = torch.concat(true_overlayed_imgs_list, dim=0)
        pred_overlayed_imgs_list = torch.concat(pred_overlayed_imgs_list, dim=0)
        true_pred_overlayed_imgs_list = torch.concat(true_pred_overlayed_imgs_list, dim=0)
    
    print("len of precions={}, len of DSC={}".format(len(precision_list), len(dsc_list)))
    avg_iou = np.mean(iou_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_dsc = np.mean(dsc_list)
    print("iou={}, precision={}, recall={}, DSC={}".format(avg_iou, avg_precision, avg_recall, avg_dsc))

    # write random predictions and ground truths 
    # rand_inds = random.sample(range(0, n_eval), min(n_eval,10)) # choose any random image at each logging 
    # if kalman_flag:
    #     log_tensorboard(writer, global_step, 'test', true_mask_list[0], pred_mask_list[0], true_overlayed_imgs_list[0], pred_overlayed_imgs_list[0], images_list[0], last_encoded_feature[0] ,flows_list,
    #                     rand_inds, kalman_flag)            
    # else:
    #     log_tensorboard(writer, global_step, 'test', true_mask_list, pred_mask_list, true_overlayed_imgs_list, pred_overlayed_imgs_list, images_list, last_encoded_feature[0] ,flows_list,
    #                     rand_inds, kalman_flag)        
        
    # return epoch_loss, avg_iou, 

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=1, help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,  help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--use_saved_data', action='store_true' ,default=False, help='using saved data')
    parser.add_argument('--task', type=str, default = 'train')
    parser.add_argument('--saved_data_file', type=str, default='4')
    parser.add_argument('--iter', type=str, default='0', help='sets the tensorbaord and weight name as well')
    parser.add_argument('--val_weights', type=str, default='0')
    parser.add_argument('--pure_images', action='store_true', default=False, help='to use only images and no flow')
    parser.add_argument('--n_flow', type=int, default=1)
    parser.add_argument('--conv_layers', type=int, default=1)
    parser.add_argument('--multi_attn', type=int, default=0)
    parser.add_argument('--flow_history_flag', action='store_true', default=False, help='using past flows')
    parser.add_argument('--learned_flow', action='store_true', default=False, help='using learning based flow')
    parser.add_argument('--late_fusion', action='store_true', default=False, help='fusing flow and image feature maps at bottleneck layer')
    parser.add_argument('--classification_flag', action='store_true', default=False, help='adding classification loss: needle present or not')
    parser.add_argument('--attention_flag', action='store_true', default=False, help='using attention based UNet')
    parser.add_argument('--store_weights', action='store_true', default = False, help='make true when weights need to be saved')
    parser.add_argument('--kalman_flag', action='store_true', default = False, help='make true when using kalman filtering')
    parser.add_argument('--tensorboard', action='store_true', default = False, help='logging on tensorboard')    
    parser.add_argument('--weight_type', type=str, default='best', choices=['best', 'lasy'], help='choosing either the best weight or the last weight')
    parser.add_argument('--freeze_weights' , action='store_true', default=False, help='freezing unet weights and training motion model')
    parser.add_argument('--eval' , action='store_true', default=True, help='Run Validation for video')
    parser.add_argument('--flow_flag', action='store_true', default = False, help='make true when optical flow')
    parser.add_argument('--traj_len', type=int, default=10, help='trajectory length after which kalman re-initialized')
    parser.add_argument('--data_type', type=str, default='DARPA', help='type of data choose from [DARPA, UPMC, BlueGel]')
    parser.add_argument('--gauss_flag', action='store_true', default=False, help='flag for using gaussian distribution inside Kalman')
    parser.add_argument('--transformer_flag', action='store_true', default=False, help='flag for using tranformer for kalman gain')
    parser.add_argument('--process_model_flag', action='store_true', default=False, help='generate next state using just process model')
    parser.add_argument('--high_res_flag', action='store_true', default=False, help='generate next state using just process model')
    parser.add_argument('--vector_flag', action='store_true', default=False, help='use vector based kalman filter')
    parser.add_argument('--needle_params', action='store_true', default=False, help='needle parameters to compute')

    return parser.parse_args()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #* GENERATE DATA
    _, _, PARENT_FOLDER_TEST, LIST_OF_DATASETS_TEST, IMAGE_LOC, MASK_LOC = get_list_train_test_data(args.data_type)

    if not args.use_saved_data:
        if args.flow_flag or args.n_flow > 1:
            test_data = get_data_dict_history(args.n_flow, PARENT_FOLDER_TEST, LIST_OF_DATASETS_TEST, IMAGE_LOC, MASK_LOC, args.saved_data_file, \
                                                type='test', flow_flag=args.flow_flag, data_type=args.data_type)
        else:
            test_data = get_data_dict_kalman_eval(PARENT_FOLDER_TEST, LIST_OF_DATASETS_TEST, IMAGE_LOC, MASK_LOC, args.saved_data_file, \
                                                    type='test', data_type=args.data_type)

    else:
        test_data = torch.load(os.path.join('saved_data', args.saved_data_file, 'test.pt')) #torch.load('saved_data/3/test.pt')
    
    ipdb.set_trace()
    #! when feeding in the entire seqeunce no need to wrap in DataLoader
    # test_data =  DataLoader(test_data, batch_size=1*batch_size, shuffle=False)

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    # dir_checkpoint = os.path.join('checkpoints/exp', str(args.iter) + '_conv_layers_{}_n_flow_={}'.format(args.conv_layers,args.n_flow))
    batch_size = args.batch_size 
    
    list_of_weights = [
                        # 'VanillaUnetOG_start_channel_64_seed_10_Abl_conv_layers_2_nflow_1',
                        # 'VanillaUnetOG_start_channel_64_seed_101_Abl_conv_layers_1_nflow_1',
                    #    'UNet_cross2_BAMC_unetstart_64_kf_32_seed_42_conv_layers_1_nflow_1'
                    #    'Attention_cross2_BAMC_unetstart_64_kf_32_seed_42_conv_layers_1_nflow_1'
                    ]
    # list_of_weights = ['FlowUnetOG_start_channel_64_seed_10_Abl_conv_layers_1_nflow_1',
    #                    'FlowUnetOG_start_channel_64_seed_42_Abl_conv_layers_1_nflow_1',
    #                    'FlowUnetOG_start_channel_64_seed_101_Abl_conv_layers_1_nflow_1'
    #                   ]
    list_of_weights = ['Convkalman2_cross2_BAMC_unetstart_64_kf_32_seed_42_conv_layers_1_nflow_1']
    # list_of_weights = ['ConvKalman_unetstart_64_kf_32_trajlen_35_seed_42_conv_layers_1_nflow_1']
    # list_of_weights = ['ConvKalman_unetstart_64_kf_64_trajlen_35_seed_101_conv_layers_1_nflow_1']
    # list_of_weights = ['ConvKalmanGauss_UNet_DARPA_unetstart_64_kf_32_seed_42_conv_layers_1_nflow_1']
    # list_of_weights = ['Stacked_cross2_BAMC_unetstart_64_kf_32_seed_42_conv_layers_1_nflow_4']
    # list_of_weights = ['LSTM_cross2_BAMC_unetstart_64_kf_32_seed_42_conv_layers_1_nflow_1']
    # list_of_weights = ['ConvKalman_UNet_UPMC_unetstart_64_kf_32_seed_42_trajlen=15_conv_layers_1_nflow_1'] # ConvKalman_BlueGel_unetstart_64_kf_32_seed_101_trajlen=12_conv_layers_1_nflow_1


    # list_of_weights = ['Stacked_UNet_BlueGel_unetstart_64_seed_10_conv_layers_1_nflow_4']


    for weight in list_of_weights:
        dir_checkpoint = os.path.join('checkpoints/exp', weight,'CP_best_val_score.pth')
        # dir_checkpoint = os.path.join('checkpoints/exp', weight,'CP_best_iou.pth')
        
        if not args.late_fusion:
            scale = 1

            kwargs = {}
            kwargs['spatial_in_channel'] = 1
            kwargs['out_channels'] = 16
            kwargs['n_classes'] = 1
            kwargs['n_depth'] = 5
            kwargs['bilinear'] = False
            kwargs['unet_channel_start'] = 64
            kwargs['kf_channels'] = 32

            net = TwoStreamUNet(args, device=device, **kwargs)
        
        else:
            print("in late fusion, no trained model present \n make late_fusion flag false")

        #! LOAD DATA AND MODEL
        net.to(device=device)
        
        weight_file_loc = dir_checkpoint #args.load #os.path.join(dir_checkpoint, 'CP_' + args.weight_type + '.pth')
        net.load_state_dict(torch.load(weight_file_loc, map_location=device))
        
        net.eval()

        if args.tensorboard:
            writer = SummaryWriter(comment='_VAL_iter_{}'.format(args.iter))
        else:
            writer = None     

        try:
            criterion = nn.BCEWithLogitsLoss() 

            eval_net(writer, 0, net, test_data, kwargs['n_classes'],criterion, device, args)
            # eval_net_stacked(writer, 0, net, test_data, kwargs['n_classes'],criterion, device, args)

            print("done evaluation, closing tensorboard writer ")
            if writer is not None:
                writer.close()        

        except KeyboardInterrupt:        
            pass
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

