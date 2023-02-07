import argparse
from asyncore import write
from cmath import asin, e
from email.policy import default
from glob import glob
import logging
import os
from re import L
import sys
from types import new_class
from math import pi
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
# from TwoStreamUnet.eval_two_stream_unet import LIST_OF_DATASETS_TEST

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
    # if len(spatial_features) > 0: 
    #     pass
    #     # spatial_features_ = spatial_features[rand_inds].unsqueeze(2)
    #     # writer.add_images(type+'_features/spatial_features', spatial_features_.reshape(-1,1,x_spatial_shape[-2],x_spatial_shape[-1]), global_step)
    # if len(temporal_features) > 0:
    #     pass
    #     # temporal_features_ = temporal_features[rand_inds].unsqueeze(2)
    #     # writer.add_images(type+'_features/temporal_features', temporal_features_.reshape(-1,1,x_spatial_shape[-2],x_spatial_shape[-1]), global_step)
def check_folder_exists(folder):

    if not os.path.exists(folder):
        os.mkdir(folder)

def eval_net_stacked(writer, global_step, net, test_data, n_classes, criterion, device, args):
    '''
    #! EVALUATE
    '''
    # images, flows, needle_masks, masks, flow_concats  = get_dict_vals(data)
    n_eval = len(test_data)    
    
    kalman_flag = args.kalman_flag

    # dir_name = os.path.join('icra_results','UNET_stacked')
    dir_name = os.path.join('ThesisResults','Stacked_seed42')
    check_folder_exists(dir_name)
    # dir_us_img = os.path.join(dir_name, 'us_img')
    # check_folder_exists(dir_us_img)
    # dir_us_img_pred_overlayed = os.path.join(dir_name, 'us_img_pred_overlayed')
    # check_folder_exists(dir_us_img_pred_overlayed)
    # dir_us_img_true_mask_overlayed = os.path.join(dir_name, 'us_img_pred_true_overlayed')
    # check_folder_exists(dir_us_img_true_mask_overlayed)
    dir_us_img_pred_mask = os.path.join(dir_name, 'us_img_pred_mask')
    check_folder_exists(dir_us_img_pred_mask)

    with torch.no_grad():
        step = 0
        ipdb.set_trace()
        for j, data in enumerate(test_data):
            # import ipdb; ipdb.set_trace()
            print("j=",j)
            mask_type = torch.float32 #if n_classes == 1 else torch.long                
            imgs = data['images'] 
            true_masks = data['needle_masks'] 
            if kalman_flag:
                flows = None 
                imgs_prev = None 
                imgs = imgs.to(device=device, dtype=mask_type).unsqueeze(0).unsqueeze(0)
                true_masks = true_masks.to(device=device, dtype=mask_type).unsqueeze(0).unsqueeze(0)
                flows = None 
                imgs_prev = None 
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
                    imgs_prev = imgs_prev.to(device=device, dtype=mask_type).unsqueeze(0)
                    imgs = imgs.to(device=device, dtype=mask_type).unsqueeze(0)
                    true_masks = true_masks.to(device=device, dtype=mask_type).unsqueeze(0)
                else:
                    imgs_prev = None #data['images_prev']
                    imgs = imgs.to(device=device, dtype=mask_type).unsqueeze(0)
                    true_masks = true_masks.to(device=device, dtype=mask_type).unsqueeze(0)
                
            new_video_flag = False 
            masks_pred, _, _, _ = net(imgs, flow=flows, image_prev=imgs_prev, new_video_flag=new_video_flag)            

            masks_pred = torch.sigmoid(masks_pred)
            masks_pred_threshold = (masks_pred > 0.5).float() 
            step += 1

            imgs = imgs.to('cpu') 
            true_masks = true_masks.to('cpu') 

            masks_pred = masks_pred.to( 'cpu')
            masks_pred_threshold = masks_pred_threshold.to( 'cpu')                

            # ipdb.set_trace()
            # plot and write using cv2.imwrite            
            if kalman_flag:
                us_img, masks_pred_threshold, true_masks = imgs[0][0][0], masks_pred_threshold[0][0][0], true_masks[0][0][0]
            else:
                us_img, masks_pred_threshold, true_masks = imgs[0][0], masks_pred_threshold[0][0], true_masks[0][0]

            us_img = us_img.numpy()
            us_img = np.stack([us_img]*3, axis=-1)*255
            masks_pred_, true_masks_, zero_mask = np.zeros_like(us_img), np.zeros_like(us_img), np.zeros_like(us_img)
            masks_pred_[:,:,2] = masks_pred_threshold.numpy()*255
            true_masks_[:,:,2] = true_masks.numpy()*255                

            alpha = 0.5
            us_img_pred_overlayed = cv2.addWeighted(us_img, 1-alpha, masks_pred_, alpha,0.0)
            us_img_true_overlayed = cv2.addWeighted(us_img, 1-alpha, true_masks_, alpha,0.0)
            us_img_ = cv2.addWeighted(us_img, 1-alpha, zero_mask, alpha, 0.0)
            # alpha = 0.4
            # gt_mask_pred_mask_overlayed = cv2.addWeighted(true_masks_, 1-alpha, masks_pred_, alpha, 0.0)

            # cv2.imwrite(os.path.join(dir_us_img,'frame_' + '000{}'.format(step).zfill(6) + '.png'), us_img)
            # desired_steps = [773,774,776, 777, 778, 779]
            # desired_steps = [73, 767]
            if step: #in desired_steps:
                # print("here")
                # cv2.imwrite(os.path.join(dir_us_img,'us_img_' + '000{}'.format(step).zfill(6) + '.png'), us_img_)
                # cv2.imwrite(os.path.join(dir_us_img_true_mask_overlayed,'true_overlayed_v2_' + '000{}'.format(step).zfill(6) + '.png'), us_img_true_overlayed)
                # cv2.imwrite(os.path.join(dir_us_img_pred_overlayed,'frame_' + '000{}'.format(step).zfill(6) + '.png'), us_img_pred_overlayed)                    
                cv2.imwrite(os.path.join(dir_us_img_pred_mask,'pred_mask_' + '000{}'.format(step).zfill(6) + '.png'), masks_pred_[:,:,2])


def eval_net(writer, global_step, net, test_data, n_classes, criterion, device, args):
    '''
    #! EVALUATE
    '''
    kalman_flag = args.kalman_flag
    # images, flows, needle_masks, masks, flow_concats  = get_dict_vals(data)
    # n_eval = len(test_data)
    
    dir_name = os.path.join('ThesisResults','VanillaUNet_v2_seed101')
    check_folder_exists(dir_name)
    # dir_us_img = os.path.join(dir_name, 'us_img')
    # check_folder_exists(dir_us_img)
    # dir_us_img_pred_overlayed = os.path.join(dir_name, 'us_img_pred_overlayed')
    # check_folder_exists(dir_us_img_pred_overlayed)
    # dir_us_img_true_mask_overlayed = os.path.join(dir_name, 'us_img_pred_true_overlayed')
    # check_folder_exists(dir_us_img_true_mask_overlayed)
    # dir_us_img_true_mask = os.path.join(dir_name, 'us_img_true_mask')
    # check_folder_exists(dir_us_img_true_mask)
    dir_us_img_pred_mask = os.path.join(dir_name, 'us_img_pred_mask')
    check_folder_exists(dir_us_img_pred_mask)

    if args.process_model_flag:
        # make a new dir for process model plots
        dir_us_img_process_model = os.path.join(dir_name, 'process_model_no_threshold')
        check_folder_exists(dir_us_img_process_model)

    with torch.no_grad():
        step = 0
        new_video_flag = True
        for j, data in enumerate(test_data):
            # import ipdb; ipdb.set_trace()
            print("j=",j)
            for i in range(len(data['images'])):
                print("i= ", i)
                mask_type = torch.float32 #if n_classes == 1 else torch.long                
                imgs = data['images'][i]
                true_masks = data['needle_masks'][i]
                if kalman_flag:
                    imgs = imgs.to(device=device, dtype=mask_type).unsqueeze(0).unsqueeze(0)
                    true_masks = true_masks.to(device=device, dtype=mask_type).unsqueeze(0).unsqueeze(0)
                    flows = None 
                    imgs_prev = None
                    if args.vector_flag:
                        needle_params = data['needle_params'][i] #x_start, y_start, needle_angle, needle_length, x_tip, y_tip
                        needle_params = needle_params.to(device=device, dtype=mask_type)
                        true_masks_new = data['needle_masks_new'][i]
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
                        imgs_prev = imgs_prev.to(device=device, dtype=mask_type).unsqueeze(0)
                        imgs = imgs.to(device=device, dtype=mask_type).unsqueeze(0).unsqueeze(0)
                        true_masks = true_masks.to(device=device, dtype=mask_type).unsqueeze(0).unsqueeze(0)
                    else:
                        imgs_prev = None #data['images_prev']
                        imgs = imgs.to(device=device, dtype=mask_type).unsqueeze(0)
                        true_masks = true_masks.to(device=device, dtype=mask_type).unsqueeze(0)

                masks_pred, _, _, _ = net(imgs, flow=flows, image_prev=imgs_prev, new_video_flag=new_video_flag)

                new_video_flag = False
                # below is false for lstm + unet 
                if kalman_flag:
                    if (i+1)%(args.traj_len) == 0:
                        new_video_flag = True
                # masks_pred, x_spatial, x_temporal = net(imgs,flows_) # flow in form of x, y matrices

                # loss = criterion(masks_pred, true_masks) # bce with logits already has sigmoid 
                # loss = sigmoid_focal_loss(masks_pred, true_masks, device)
                if not args.vector_flag:
                    masks_pred = torch.sigmoid(masks_pred)
                    masks_pred_threshold = (masks_pred > 0.5).float() # 
                    masks_pred = masks_pred.to( 'cpu')
                    masks_pred_threshold = masks_pred_threshold.to( 'cpu')                
                
                imgs = imgs.to('cpu') 
                true_masks = true_masks.to('cpu') 

                        
                step += 1

                # plot and write using cv2.imwrite
                if kalman_flag:
                    if args.process_model_flag:
                        masks_pred_threshold_list = masks_pred[:,0,0].numpy()
                        # us_img, masks_pred_threshold, true_masks = imgs[0][0][0], masks_pred_threshold[:,0,0], true_masks[0][0][0]

                    if args.vector_flag:
                        # make gt mask and prediction using needle params and mask_pred
                        us_img = imgs[0][0][0]
                        H, W = us_img.shape[-2], us_img.shape[-1]
                        masks_pred_threshold, true_masks = np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W, 3), dtype=np.uint8)
                        # convert to np and use cv2 line
                        needle_params = needle_params.cpu().numpy()
                        needle_params = needle_params.astype(np.uint8)
                        true_masks = cv2.line(true_masks, (needle_params[1], needle_params[0]), (needle_params[5],needle_params[4]), (255,255,255), 5)
                        
                        scale_needle_param = torch.tensor([256, 256, (2*pi), 256]).to(device)
                        masks_pred = masks_pred[:4]*scale_needle_param
                        masks_pred = masks_pred.cpu().numpy()
                        masks_pred = masks_pred[:,0]
                        x_tip = masks_pred[0] + np.cos(masks_pred[2])*masks_pred[3]
                        y_tip = masks_pred[1] + np.sin(masks_pred[2])*masks_pred[3]
                        masks_pred_threshold = cv2.line(masks_pred_threshold, (int(round(masks_pred[1])), int(round(masks_pred[0]))), (int(round(y_tip)), int(round(x_tip))), (255,255,255), 5)

                        true_masks = torch.from_numpy(true_masks) #.permute(2,0,1)
                        masks_pred_threshold = torch.from_numpy(masks_pred_threshold)#.permute(2,0,1)
                        true_masks, masks_pred_threshold = true_masks[:,:,0]/255, masks_pred_threshold[:,:,0]/255
                    else:
                        us_img, masks_pred_threshold, true_masks = imgs[0][0][0], masks_pred_threshold[0][0][0], true_masks[0][0][0]

                else:
                    us_img, masks_pred_threshold, true_masks = imgs[0][0], masks_pred_threshold[0][0], true_masks[0][0]

                us_img = us_img.numpy()
                us_img = np.stack([us_img]*3, axis=-1)*255
                masks_pred_, true_masks_, zero_mask = np.zeros_like(us_img), np.zeros_like(us_img), np.zeros_like(us_img)
                masks_pred_[:,:,2] = masks_pred_threshold.numpy()*255
                true_masks_[:,:,2] = true_masks.numpy()*255                

                alpha = 0.5
                us_img_pred_overlayed = cv2.addWeighted(us_img, 1-alpha, masks_pred_, alpha,0.0)
                us_img_true_overlayed = cv2.addWeighted(us_img, 1-alpha, true_masks_, alpha,0.0)
                us_img_ = cv2.addWeighted(us_img, 1-alpha, zero_mask, alpha, 0.0)
                # alpha = 0.4
                # gt_mask_pred_mask_overlayed = cv2.addWeighted(true_masks_, 1-alpha, masks_pred_, alpha, 0.0)

                # cv2.imwrite(os.path.join(dir_us_img,'frame_' + '000{}'.format(step).zfill(6) + '.png'), us_img)
                # desired_steps = [773,774,776, 777, 778, 779]
                # desired_steps = [73, 767]
                # ipdb.set_trace()
                if step: #in desired_steps:
                    # print("here")
                    # cv2.imwrite(os.path.join(dir_us_img,'us_img_' + '000{}'.format(step).zfill(6) + '.png'), us_img_)
                    # cv2.imwrite(os.path.join(dir_us_img_true_mask_overlayed,'true_overlayed_' + '000{}'.format(step).zfill(6) + '.png'), us_img_true_overlayed)
                    # cv2.imwrite(os.path.join(dir_us_img_pred_overlayed,'frame_' + '000{}'.format(step).zfill(6) + '.png'), us_img_pred_overlayed)                    
                    # cv2.imwrite(os.path.join(dir_us_img_true_mask,'true_mask_' + '000{}'.format(step).zfill(6) + '.png'), true_masks_[:,:,2])
                    cv2.imwrite(os.path.join(dir_us_img_pred_mask,'pred_mask_' + '000{}'.format(step).zfill(6) + '.png'), masks_pred_[:,:,2])

                    #write process model images 
                    if args.process_model_flag:
                        for kk in range(masks_pred_threshold_list.shape[0]):
                            img_process_model = np.stack([masks_pred_threshold_list[kk]]*3, axis=-1)*255
                            cv2.imwrite(os.path.join(dir_us_img_process_model,'process_model_' + '000{}'.format(step).zfill(6) + '_' + str(kk) + '.png'), img_process_model)
                    
                    # cv2.imwrite(os.path.join(dir_name,'attention_unet_overlayed_' + '000{}'.format(step).zfill(6) + '.png'), us_img_pred_overlayed)
                # if step == 779:
                #     break
                    # ipdb.set_trace()

            # if j >= 0:
            #     break
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
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
    parser.add_argument('--process_model_flag', action='store_true', default=False, help='generate next state purely using process model')
    parser.add_argument('--gauss_flag', action='store_true', default=False, help='flag for using gaussian distribution inside Kalman')
    parser.add_argument('--transformer_flag', action='store_true', default=False, help='flag for using tranformer for kalman gain')
    parser.add_argument('--high_res_flag', action='store_true', default=False, help='generate next state using just process model')
    parser.add_argument('--vector_flag', action='store_true', default=False, help='generate next state using just process model')


    return parser.parse_args()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #* GENERATE DATA

    PARENT_FOLDER_TRAIN, LIST_OF_DATASETS_TRAIN, PARENT_FOLDER_TEST, LIST_OF_DATASETS_TEST, IMAGE_LOC, MASK_LOC = get_list_train_test_data(args.data_type)    
    if not args.use_saved_data:
        # test_data = get_data_dict_kalman(LIST_OF_DATASETS_TEST, PARENT_FOLDER_TEST, args.saved_data_file, 'test', traj_len = 50)
        if args.flow_flag or args.n_flow > 1:
            # test_data = get_data_dict_history(args.n_flow, LIST_OF_DATASETS_TEST, PARENT_FOLDER_TEST, args.saved_data_file, type='test', flow_flag=args.flow_flag, flow_history_flag=args.flow_history_flag)
            test_data = get_data_dict_history(args.n_flow, PARENT_FOLDER_TEST, LIST_OF_DATASETS_TEST, IMAGE_LOC, MASK_LOC, args.saved_data_file,'test', args.data_type, flow_flag=args.flow_flag, flow_history_flag=args.flow_history_flag)
        else:
            if args.vector_flag:
                test_data = get_data_dict_kalman_vec(PARENT_FOLDER_TEST, LIST_OF_DATASETS_TEST, IMAGE_LOC, MASK_LOC, args.saved_data_file, type='test', traj_len=args.traj_len, data_type=args.data_type,repeat_flag=False)
            else:
                test_data = get_data_dict_kalman(PARENT_FOLDER_TEST, LIST_OF_DATASETS_TEST, IMAGE_LOC, MASK_LOC, args.saved_data_file, type='test', traj_len=args.traj_len, data_type=args.data_type)

    else:
        test_data = torch.load(os.path.join('saved_data', args.saved_data_file, 'test.pt')) #torch.load('saved_data/3/test.pt')
    
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
    
    # list_of_weights = [
                        # 'VanillaUnet_start_channel_64_Abl_conv_layers_1_nflow_1',
                        # 'VanillaUnet_start_channel_64_seed_42_Abl_conv_layers_1_nflow_1',
                        # 'VanillaUnet_start_channel_64_seed_101_Abl_conv_layers_1_nflow_1'
                        # ]
    # list_of_weights = ['FlowUnetOG_start_channel_64_seed_10_Abl_conv_layers_1_nflow_1',
    #                    'FlowUnetOG_start_channel_64_seed_42_Abl_conv_layers_1_nflow_1',
    #                    'FlowUnetOG_start_channel_64_seed_101_Abl_conv_layers_1_nflow_1'
    #                   ]
    # list_of_weights = ['LSTMUnet_start_channel_64_Abl_conv_layers_1_nflow_1']
    # list_of_weights = ['ConvkalmanNet_Unet_start_channel_64_kf_32_seed_101_Abl_conv_layers_1_nflow_1']
    # list_of_weights = ['FlowUnetOG_start_channel_64_seed_42_Abl_conv_layers_1_nflow_1']
    list_of_weights = ['VanillaUnet_start_channel_64_seed_101_Abl_conv_layers_1_nflow_1']
    # list_of_weights = ['AttentionUnetOG_start_channel_64_seed_10_Abl_conv_layers_2_nflow_1']
    # list_of_weights = ['StackedUnet_unetstart_64_seed_42_conv_layers_1_nflow_4']
    # list_of_weights = ['ConvKalmanVec_UNet_BlueGel_unetstart_64_kf_32_seed_42_conv_layers_1_nflow_1']

    for weight in list_of_weights:
        # dir_checkpoint = os.path.join('checkpoints/exp', weight,'CP_300.pth')
        # dir_checkpoint = os.path.join('checkpoints/exp', weight,'CP_best_iou.pth')
        dir_checkpoint = os.path.join('checkpoints/exp', weight,'CP_best_val_score.pth')
        # dir_checkpoint = os.path.join('checkpoints/exp', weight,'CP_last.pth')
        
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
            # eval_net(net= net, device=device, args=args, **kwargs)
            eval_net(writer, 0, 
                    net, test_data, kwargs['n_classes'], 
                    criterion, device, args)

            # eval_net_stacked(writer, 0, net, test_data, kwargs['n_classes'], 
            #                 criterion, device, args)

            print("done evaluation, closing tensorboard writer ")
            if writer is not None:
                writer.close()        

        except KeyboardInterrupt:        
            pass
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

