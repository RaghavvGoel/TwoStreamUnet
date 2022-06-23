import argparse
from glob import glob
import logging
import os
from re import L
import sys

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
from two_stream_unet import TwoStreamUNet
from rishabh import iou

torch.manual_seed(10)
# from torch.utils.tensorboard import SummaryWriter
# from utils.dataset import BasicDataset # write your own 
from utils_two_stream_unet import get_data, get_data_all_dataset, get_dict_vals
from torch.utils.data import DataLoader, random_split

# data_folder_path = 'data/task_positives_11-2022_04_12_18_28_14-segmentation mask 1.1/'
# dir_img = os.path.join(data_folder_path, 'JPEGImages') #'data/imgs/'
# dir_mask = os.path.join(data_folder_path, 'SegmentationClass') #'data/masks/'
PARENT_FOLDER_TRAIN = 'data'
PARENT_FOLDER_TEST = 'data/test'
LIST_OF_DATASETS_TRAIN = ['task_positives_11-2022_04_12_18_28_14-segmentation mask 1.1',
                    'task_positives_153-2022_04_12_18_32_22-segmentation mask 1.1',
                    'task_positives_189-2022_04_12_18_32_54-segmentation mask 1.1',
                    'task_positives_193-2022_04_18_19_26_03-segmentation mask 1.1',
                    'task_positives_67-2022_04_18_19_11_58-segmentation mask 1.1',
                    'task_positives_222-2022_04_19_22_31_15-segmentation mask 1.1',
                    'task_positives_230-2022_04_12_18_39_39-segmentation mask 1.1'
                    ]

LIST_OF_DATASETS_TEST = ['task_positives_205-2022_04_21_17_04_12-segmentation mask 1.1',
                    ]
# dir_checkpoint = 'checkpoints/exp'
iter = 16
dir_checkpoint = os.path.join('checkpoints/exp', str(iter))
if not os.path.exists(dir_checkpoint):
    os.makedirs(dir_checkpoint)

# def get_dict_vals(data):

#     images_tensor = data['images']
#     flows_tensor = data['flows']
#     needle_masks_tensor = data['needle_masks']
#     masks_tensor = data['masks']
#     flow_concats_tensor = data['flow_concats']
    
#     return images_tensor, flows_tensor, needle_masks_tensor, masks_tensor, flow_concats_tensor 

# def test_train(net, device):

#     #generate random inputs  
#     # image = torch.rand(1,1,256,256).to(device)
#     # flow = torch.rand(1,5,256,256).to(device)
#     image_list, flow_list, needle_masks_list, mask_list, flow_concat_list = get_data()
#     ipdb.set_trace()
#     # WRAP IN DATALOADER , SHUFFLE FALSE INITIALLY 
#     # DEFINE LOSS 
#     # DEFINE OPTIMIZER 
#     # LOOP OVER DATA 
#     # STORE IN TENSORBOARD THE PREDICTION MASKS AND LOSS 
#     logits = net(image, flow)

    # ipdb.set_trace()

def eval_net(net, data, n_classes, criterion, device):
    '''
    pass data as dictionart and extract here 
    '''
    images, flows, needle_masks, masks, flow_concats  = get_dict_vals(data)
    n_eval = len(images)
    batch_size = 10 #len(images)//4
    epoch_loss = 0
    true_mask_list, pred_mask_list, loss_list, loss_v2_list, true_overlayed_imgs_list, pred_overlayed_imgs_list = [], [], [], [], [], []
    pred_v2_overlayed_imgs_list = []

    with torch.no_grad():
        # for i in range(n_eval//batch_size):
        i = 0
        while i <= (n_eval//batch_size):
            i += 1
            ind_ = range((i-1)*batch_size,i*batch_size) if i*batch_size < n_eval else range((i-1)*batch_size,n_eval)
            imgs = images[ind_] #batch['image']
            true_masks = needle_masks[ind_] #batch['mask']
            flows = flow_concats[ind_]

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            flows = flows.to(device=device, dtype=mask_type)

            masks_pred = net(imgs,flows)

            # loss = criterion(masks_pred, true_masks) # bce with logits already has sigmoid 
            loss = sigmoid_focal_loss(masks_pred, true_masks, device)
            masks_pred = torch.sigmoid(masks_pred)
            masks_pred_ = (masks_pred > 0.5).float() # additional filtering ? 
            # loss_v2 = criterion(masks_pred_, true_masks)
            # epoch_loss += loss.item()
            imgs = imgs.to(device='cpu') #, dtype=torch.float32        
            true_masks = true_masks.to(device='cpu') #, dtype=mask_type)
            flows = flows.to(device='cpu') #, dtype=mask_type)
            masks_pred = masks_pred.to(device= 'cpu')
            masks_pred_ = masks_pred_.to(device= 'cpu')

            loss_list.append(loss.item())
            # loss_v2_list.append(loss_v2.item())
            true_mask_list.append(true_masks)
            pred_mask_list.append(masks_pred)
            # ipdb.set_trace()
            # overlayed_imgs_list.append(torch.concat([imgs.to('cpu'), true_masks, masks_pred_],dim = 1))
            tmp1 = torch.concat([imgs, 0.5*(imgs + true_masks), imgs], dim = 1) 
            # tmp2 = torch.concat([imgs, 0.5*(imgs + masks_pred), imgs], dim = 1) 
            tmp3 = torch.concat([imgs, 0.5*(imgs + masks_pred_), imgs], dim = 1) 
                    
            true_overlayed_imgs_list.append(tmp1)
            # pred_overlayed_imgs_list.append(tmp2)
            pred_v2_overlayed_imgs_list.append(tmp3)


        # ipdb.set_trace()
        epoch_loss = np.mean(loss_list)
        # epoch_loss_v2 = np.mean(loss_v2_list)
        true_mask_list = torch.concat(true_mask_list, dim=0)
        pred_mask_list = torch.concat(pred_mask_list, dim=0)
        true_overlayed_imgs_list = torch.concat(true_overlayed_imgs_list, dim=0)
        # pred_overlayed_imgs_list = torch.concat(pred_overlayed_imgs_list, dim=0)
        pred_v2_overlayed_imgs_list = torch.concat(pred_v2_overlayed_imgs_list, dim=0)

        # print("loss: ", epoch_loss, " loss_v2: " , epoch_loss_v2)
        # print("shapes: ", pred_mask_list.shape)

        # ipdb.set_trace()
        # tmp_ = torch.concat([true_mask_list, torch.sigmoid(pred_mask_list) > 0.5])
        # overlayed_imgs = torch.concat([images[len(true_mask_list),0:1,:,:], tmp_.to('cpu')], dim = 1)
        
    # written or write random predictions and ground truths 
    # will help in debug
    return epoch_loss, true_mask_list, pred_mask_list, true_overlayed_imgs_list, pred_v2_overlayed_imgs_list, images

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
              iter = 0):

    global dir_checkpoint
    # use_saved_data = True
    # dataset = BasicDataset(dir_img, dir_mask, img_scale)
    # iter = 16
    dir_checkpoint = os.path.join('checkpoints/exp', str(iter))
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)    
    
    if not use_saved_data:
        # images_tensor, flows_tensor, needle_masks_tensor, masks_tensor, flow_concats_tensor = get_data(net.temporal_in_channel)
        train_data =  get_data_all_dataset(net.temporal_in_channel, LIST_OF_DATASETS_TRAIN, PARENT_FOLDER_TRAIN, iter,'train')
        test_data =  get_data_all_dataset(net.temporal_in_channel, LIST_OF_DATASETS_TEST, PARENT_FOLDER_TEST, iter, 'test')
    else:
        train_data = torch.load(os.path.join('saved_data', saved_data_file, 'train.pt')) #torch.load('saved_data/3/train.pt')
        test_data = torch.load(os.path.join('saved_data', saved_data_file, 'test.pt')) #torch.load('saved_data/3/test.pt')

    images_test, flows_test, needle_masks_test, masks_test, flow_concats_test  = get_dict_vals(test_data)  
    images_train, flows_train, needle_masks_train, masks_train, flow_concats_train  = get_dict_vals(train_data)    
        
    n_val = len(images_test) #int(len(dataset) * val_percent)
    n_train = len(images_train) #- n_val
    # train, val = random_split(dataset, [n_train, n_val])
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    """
    GENERATE DATA W/O WRAPPING IN DATALOADER, CAN'T USE SHUFFLE WITH FLOW or shuffle in batches
    for one image, flow has 3 previous frames, can create Bx5xHxW for flow and Bx1XHxW for US images
    """
    # ipdb.set_trace()
    # writer = SummaryWriter(comment='LR_{}_BS_{}_PATIENCE_{}'.format(lr, batch_size, 20))
    patience = 5
    writer = SummaryWriter(comment='LR_{}_BS_{}_PATIENCE_{}_train_iter_{}'.format(lr, batch_size, patience, iter))
    global_step = 0
    eval_step = 50

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # also try cosine_annealing
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.9)
    
    if n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(150)) #pos_weight=150

    rand_inds = random.sample(range(0, n_val), 20) # sampling 10 random numbers for plotting 
    ################################ VALIDATION ################################################ 
    ################################ VALIDATION ################################################ 
    ################################ VALIDATION ################################################ 
    if task == 'val':
        # only run eval and return | load weights 
        # ipdb.set_trace()
        dir_checkpoint_ = os.path.join('checkpoints', 'exp', weight_file)
        # dir_checkpoint_ = os.path.join('/home/luyuan/thomaswe/TwoStreamUnet/checkpoints', 'exp', '13', 'CP_best.PTH')
        net.load_state_dict(torch.load(dir_checkpoint_))
        net.eval()
        val_score, true_masks_test, masks_pred_test, true_overlayed_imgs, pred_overlayed_imgs, imgs_test \
                    = eval_net(net, test_data, n_classes, criterion, device)                                    
        if net.n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(val_score))
            writer.add_scalar('Loss/test', val_score, global_step)
        else:
            logging.info('Validation Loss Coeff: {}'.format(val_score))
            writer.add_scalar('Loss/test', val_score, global_step)

        # if net.n_classes == 1:
            # writer.add_images('train/mask_true', true_masks, global_step)
            # writer.add_images('train/mask_pred', torch.sigmoid(masks_pred) > 0.5, global_step)
            # choose random index to check 
        true_masks_test_ = true_masks_test[rand_inds]
        masks_pred_test_ = masks_pred_test[rand_inds]
        true_overlayed_imgs_ = true_overlayed_imgs[rand_inds]
        pred_overlayed_imgs_ = pred_overlayed_imgs[rand_inds]
        imgs_test_ = imgs_test[rand_inds]
        writer.add_images('test/images', imgs_test_, global_step)
        writer.add_images('test/mask_true', true_masks_test_, global_step)
        writer.add_images('test/mask_pred', masks_pred_test_, global_step)
        writer.add_images('test/mask_pred_sigmoid', masks_pred_test_ > 0.5, global_step)
        writer.add_images('test/true_overlayed', true_overlayed_imgs_, global_step)
        writer.add_images('test/pred_overlayed', pred_overlayed_imgs_, global_step)
        
        print("done evaluation ")
        writer.close()
        return 
    ################################  ################################################ 
    ################################  ################################################ 
    ################################  ################################################ 

    min_val_score = 1000 # some large number
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        # print("check why a lot of GPU is being used")
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            # for batch in train_loader:
            for i in range(n_train//batch_size):
                # ipdb.set_trace()
                ind_ = range((i)*batch_size,(i+1)*batch_size)
                imgs = images_train[ind_] #batch['image']
                true_masks = needle_masks_train[ind_] #batch['mask']
                flows = flow_concats_train[ind_]

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                flows = flows.to(device=device, dtype=torch.float32)

                # print("check why no issue in train of different dtype of mask_pred and true_mask")
                # ipdb.set_trace()
                masks_pred = net(imgs,flows)
                # loss = criterion(masks_pred, true_masks)
                loss = sigmoid_focal_loss(masks_pred, true_masks, device) 
                # dice_loss = find_dice_loss(masks_pred, true_masks) # most prob wrong implementation 
                # dice_loss = torch.sum(masks_pred*true_masks, dim=0)/(torch.sum(masks_pred, dim=0) + torch.sum(true_masks, dim=0))
                # dice_loss = torch.mean(dice_loss)
                # print("dice_loss = ", dice_loss)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                # writer.add_scalar('Loss/train_dice_loss', dice_loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item(), 'lr':optimizer.param_groups[0]['lr']})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1) ## increasead from 0.1 to 0.5 BE MIndFUL OF this 
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                # if global_step % (n_train // (10 * batch_size)) == 0:
                if global_step % eval_step == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    
                    net.eval()
                    val_score, true_masks_test, masks_pred_test, true_overlayed_imgs, pred_overlayed_imgs, imgs_test = eval_net(net, test_data, n_classes, criterion, device)
                    if val_score < min_val_score:
                        min_val_score = val_score
                        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_best.pth'))
                    net.train()
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    test_iou = iou(masks_pred_test, true_masks_test)
                    writer.add_scalar('IOU/test', test_iou, global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Loss: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)

                    # ipdb.set_trace()
                    writer.add_images('train/images', imgs, global_step)
                    # writer.add_images('train/flows', torch.concat([flows[:,-1:,:,:]]*3, dim=1), global_step)
                    writer.add_images('train/flows', flows, global_step)
                    if net.n_classes == 1:
                        writer.add_images('train/mask_true', true_masks, global_step)
                        masks_pred = torch.sigmoid(masks_pred)
                        masks_pred_ = (masks_pred > 0.5).float()
                        # writer.add_images('train/mask_pred_sigmoid', masks_pred > 0.5, global_step)
                        writer.add_images('train/mask_pred_sigmoid', masks_pred_, global_step)
                        writer.add_images('train/mask_pred', masks_pred, global_step)
                        writer.add_images('train/true_overlayed' ,torch.concat([imgs, 0.5*(imgs + true_masks), imgs], dim = 1), global_step)
                        writer.add_images('train/pred_overlayed' ,torch.concat([imgs, 0.5*(imgs + masks_pred_), imgs], dim = 1), global_step)
                        # COMPUTE IOU | copy in test section below as well
                        # ipdb.set_trace()
                        train_iou = iou(masks_pred_, true_masks)
                        writer.add_scalar('IOU/train', train_iou, global_step)

                        # choose random index to check 
                        true_masks_test_ = true_masks_test[rand_inds]
                        masks_pred_test_ = masks_pred_test[rand_inds]
                        true_overlayed_imgs_ = true_overlayed_imgs[rand_inds]
                        pred_overlayed_imgs_ = pred_overlayed_imgs[rand_inds]
                        imgs_test_ = imgs_test[rand_inds]
                        writer.add_images('test/images', imgs_test_, global_step)
                        writer.add_images('test/mask_true', true_masks_test_, global_step)
                        writer.add_images('test/mask_pred', masks_pred_test_, global_step)
                        writer.add_images('test/mask_pred_sigmoid', masks_pred_test_ > 0.5, global_step)
                        writer.add_images('test/true_overlayed', true_overlayed_imgs_, global_step)
                        writer.add_images('test/pred_overlayed', pred_overlayed_imgs_, global_step)
        if save_cp:
            try:
                # os.mkdir(dir_checkpoint)
                os.makedirs(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            # torch.save(net.state_dict(),
            #            dir_checkpoint + 'CP_epoch{}.pth'.format(epoch+1))
            # logging.info(f'Checkpoint {epoch + 1} saved !')

            # save last and best epoch based on eval loss 
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_last.pth'))

    writer.close()


def sigmoid_focal_loss(inputs, targets, device, alpha = 0.25, gamma = 2, reduction = "none"):
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
    net = TwoStreamUNet(spatial_in_channel=1, temporal_in_channel=1, out_channel=16, n_classes=1)
    
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
    print("val_weights: ", args.val_weights)
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
                iter = args.iter)
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