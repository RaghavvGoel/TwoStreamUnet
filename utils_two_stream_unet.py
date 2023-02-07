import os
from tkinter import N
import numpy as np
import torch
from PIL import Image
import ipdb
import cv2 
from torchvision import transforms
import torch.nn.functional as F

torch.manual_seed(42)

#! add augmentation for training 
transform_rot_flips = transforms.Compose([transforms.RandomHorizontalFlip(p=0.3),
                                          transforms.RandomRotation(15),        
                                        ])
transform_horflip = transforms.RandomHorizontalFlip(p=0.3)

transform_blur = transforms.Compose([transforms.GaussianBlur(kernel_size=5, sigma=(0.01,2)),])
# only for images and not for flows
transform_jitter = transforms.Compose([lambda x:x/255,
                                       transforms.ColorJitter([0.75, 1.25])])

#! add augmentation for testing                                
transform_blur_test = transforms.Compose([lambda x:x/255,
                                        transforms.GaussianBlur(kernel_size=5, sigma=(0.01,2)),
                                        ])                        
#! LOCATIONS 
IMAGE_LOC  = 'JPEGImages'
MASK_LOC = 'SegmentationClass'


def convert_to_uint_and_transpose(img):
    # muliply by 255 and conver to numpy and uint8
    if type(img) != np.ndarray:
        img = img.numpy()
    
    img *= 255
    img = img.transpose(1,2,0)
    img = img.astype(np.uint8)

    return img

def get_weights(named_parameters):
    '''
    return list of params 
    '''
    weights_list = []
    for name, param in named_parameters.named_parameters():
        weights_list.append(param)
    
    return weights_list


def find_contours(img, mask, threshold_flag=False):
    if threshold_flag:
        ret, mask = cv2.threshold(mask, 127, 255, 0)
    contours_pred, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2BGR)
    img = np.concatenate([img]*3, axis=-1)
    for i,c in enumerate(contours_pred):
        # mask = np.zeros(mask.shape, np.uint8)
        # cv2.drawContours(mask, [c], -1, 255, -1)
        ## Get appropriate colour for this label
        # label = 2 if mean > 1.0 else 1 # not needed as only 1 label
        colour = (0,0,255) #RGBforLabel.get(label)  
        cv2.drawContours(img,[c],-1,colour,1)
    return img


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


def concatenate_m_flow_frames(flow_list, n_flow, img_H, img_W):
    if len(flow_list) >= n_flow:
        tmp = torch.cat(flow_list[-n_flow:], dim=-1)              
    else:
        tt1 = torch.cat([torch.zeros(img_H, img_W, 1)]*(n_flow-len(flow_list)), dim=-1)
        tt2 = torch.cat(flow_list, dim=-1)
        tmp = torch.cat([tt2,tt1], dim=-1)

    return tmp  

def concatenate_m_flow_frames_np(flow_list, n_flow, img_H, img_W, gamma=0.5):
    if len(flow_list) >= n_flow:
        tmp = np.concatenate(flow_list[-n_flow:], axis=-1)              
    else:
        tt1 = np.concatenate([np.zeros((img_H, img_W, 1))]*(n_flow-len(flow_list)), axis=-1)
        tt2 = np.concatenate(flow_list, axis=-1)
        tmp = np.concatenate([tt2,tt1], axis=-1)

    # multiply tmp past with weights to have forgetting factor
    tmp = np.concatenate([tmp[:,:,i:i+1]*(gamma**i) for i in range(n_flow)],axis=-1)

    return tmp        

def concatenate_m_flow_frames_raw(flow_xy_list, n_flow, img_H, img_W):
    # two channels per flow
    scale = 2 
    ipdb.set_trace()
    if len(flow_xy_list) >= scale*n_flow:
        tmp = torch.cat(flow_xy_list[-scale*n_flow:], dim=-1)              
    else:
        tt1 = torch.cat([torch.zeros(img_H, img_W, 1)]*(scale*n_flow-len(flow_xy_list)), dim=-1)
        tt2 = torch.cat(flow_xy_list, dim=-1)
        tmp = torch.cat([tt2,tt1], dim=-1)

    return tmp

def get_dict_vals(data):

    images_tensor = data['images']
    flows_tensor = data['flows']
    needle_masks_tensor = data['needle_masks']
    masks_tensor = data['masks']
    flow_concats_tensor = data['flow_concats']
    
    return images_tensor, flows_tensor, needle_masks_tensor, masks_tensor, flow_concats_tensor 

def get_image(i, dataset, PARENT_FOLDER):
    image_path = os.path.join(PARENT_FOLDER, dataset, IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')
    # if not os.path.exists(image_path):
    #     return None
    # img = cv2.imread(image_path)
    # img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
    try: 
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        return img 
    except:
        return None
        

def get_mask(i, dataset, PARENT_FOLDER,type='train'):
    extension = '.PNG'
    mask_path = os.path.join(PARENT_FOLDER, dataset, MASK_LOC, 'frame_' + '000{}'.format(i).zfill(6) + extension)
    try:
        mask = cv2.imread(mask_path)
        mask_resized = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        return mask, mask_resized
    except:
        return None, None
    # if not os.path.exists(mask_path):
    #     return None, None    
    # mask = cv2.imread(mask_path) 
    # mask_resized = cv2.resize(mask, dsize=(256,256), interpolation=cv2.INTER_NEAREST)    
    # return mask, mask_resized #! check if resized used or not 


def find_needle_mask(mask, type='train'):
    mask_needle = 255*(np.array([255,255,255]) - np.where(mask == 51,221,255)).astype(np.uint8)
    mask_needle_resized = cv2.resize(mask_needle, dsize=(256,256), interpolation=cv2.INTER_AREA)#.astype('uint8')
    
    mask_needle_resized_gray = cv2.cvtColor(mask_needle_resized, cv2.COLOR_BGR2GRAY)
    (thresh, mask_needle_binary) = cv2.threshold(mask_needle_resized_gray, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #! no need to fill for test case? 
    if type == 'train':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31, 31))
        mask_needle_binary = cv2.morphologyEx(mask_needle_binary, cv2.MORPH_CLOSE, kernel)    

    # max_mask = np.max(mask_needle_resized)
    # ind_ = np.where(mask_needle_resized == max_mask)[-1][0]
    # tmp_ = mask_needle_resized[:,:,ind_] # BGR #np.array(mask_needle_resized)[:,:,0]
    # # PUTTING CHECK HERE TO NOT USE IMAGES W/O NEEDLE MASK
    # if np.max(tmp_) == 0:
    #     return None
    # mask_needle_arr = tmp_ // np.max(tmp_)
    # mask_needle_arr = mask_needle_arr.astype(np.int16)

    return mask_needle_binary



def find_flow_history(image_list, img_H, img_W, n_history=1, abs_flag = False):
    # find flow b/w current image and past n_history of images 
    flow_list = []
    flowX_list = []
    flowY_list = []
    cartesian = np.zeros((img_H,img_W,3), dtype=np.uint8)
    for k in range(n_history):    
        if len(image_list) > n_history:        
            prvs = np.array(image_list[-1-1-k])
            next = np.array(image_list[-1]) # always take the last image

            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)                        
            mean_flow = np.array([np.mean(flow[..., 0]), np.mean(flow[..., 1])]) #extra 
            if abs_flag:
                flow = np.abs(flow) 
            cartesian[..., [0,2]] = cv2.normalize(flow,None, 0, 255, cv2.NORM_MINMAX)
            cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)            
            flow_list.append(cartesian_bgr[:,:,0:1])
            flowX_list.append(flow[..., 0:1])
            flowY_list.append(flow[..., 1:2])
        else:
            flow_list.append(np.zeros((img_H, img_W, 1), dtype=np.uint8))
            flowX_list.append(np.zeros((img_H, img_W, 1), dtype=np.uint8))
            flowY_list.append(np.zeros((img_H, img_W, 1), dtype=np.uint8))

    return np.concatenate(flow_list, axis=-1), np.concatenate(flowX_list, axis=-1), np.concatenate(flowY_list, axis=-1)


def find_image_history(image_list, n_history):
    '''
        concatenate t-n-histroy to t-1 images if they dont exist then add blanks
    '''
    if len(image_list)-1 >= n_history:
        # return np.stack(image_list[-n_history-1:-1], axis=0) # need extra dim to apply transformations        
        return np.concatenate(image_list[-n_history-1:-1], axis=-1) # need extra dim to apply transformations
    else:
        extras = n_history - (len(image_list) - 1)
        img_blank = np.concatenate([np.zeros_like(image_list[-1])]*extras, axis=-1)        
        return np.concatenate([img_blank, np.concatenate(image_list[:-1], axis=-1)], axis=-1)


def transform_all_in_same_way(inp, type, n_history, flow_flag=False):
    # inp is a list of all things which go in dict
    out_trans = torch.cat(inp, dim=0)
    assert torch.max(out_trans[-1]) <= 1.0, 'max of needle mask needs to be <= 0'    
    if type == 'train':
        # gaussian blur for img
        img = torch.cat([out_trans[0:1].unsqueeze(1)]*3,dim=1)
        img_prev = torch.cat([out_trans[1:1+n_history].unsqueeze(1)]*3, dim = 1)
        img = img.type(torch.uint8)
        img_prev = img_prev.type(torch.uint8)
        # flow_concat = torch.cat([out_trans[n_history+6:n_history+9].unsqueeze(1)]*3, dim = 1)
        img_img_prev = torch.cat([img, img_prev], dim = 0)
        
        # extract out current image and previous images
        # transformed_img_img_prev = transformed_img_img_prev_flow[0:1+n_history]
        
        # apply jitter to images
        transformed_img_img_prev = transform_jitter(img_img_prev)

        assert torch.max(transformed_img_img_prev) <= 1.0, 'max of image data should be 1.0'

        # update in out_trans
        out_trans[0:1] = transformed_img_img_prev[0,0:1] # 1x256x256
        out_trans[1:1+n_history] = transformed_img_img_prev[1:1+n_history,0]
        # out_trans[n_history+6:n_history+9] = transformed_img_img_prev_flow[1+n_history:,0]
        # out_trans = transform_rot_flips(out_trans)

    if type == 'test':
        img = torch.cat([out_trans[0:1].unsqueeze(1)]*3,dim=1)
        img_prev = torch.cat([out_trans[1:1+n_history].unsqueeze(1)]*3, dim = 1)
        img = img.type(torch.uint8)
        img_prev = img_prev.type(torch.uint8)
        # flow_concat = torch.cat([out_trans[n_history+6:n_history+9].unsqueeze(1)]*3, dim = 1)
        img_img_prev = torch.cat([img, img_prev], dim = 0)
        
        transformed_img_img_prev = img_img_prev/255.0 #transform_blur_test(img_img_prev_flow)
        
        # extract out current image and previous images
        transformed_img_img_prev = transformed_img_img_prev[0:1+n_history]        

        # update in out_trans
        out_trans[0:1] = transformed_img_img_prev[0,0:1]
        out_trans[1:1+n_history] = transformed_img_img_prev[1:1+n_history,0]
        # out_trans[n_history+6:n_history+9] = transformed_img_img_prev[1+n_history:,0]

        # out_trans = transform_horflip(out_trans)

    return out_trans


def transform_videos(imgs, masks, type):
    batch_size = imgs.shape[0]    
    assert imgs.shape[0] == masks.shape[0], 'image batch size NOT same as mask batch size'
    if type == 'train':
        imgs_trans = transform_jitter(imgs)
        imgs_masks = torch.cat([imgs_trans, masks], dim=0)
        imgs_masks = transform_rot_flips(imgs_masks)
    else:
        imgs = imgs/255.0 #transform_jitter(imgs) #imgs/255.0
        masks = masks #/255.0
        imgs_masks = torch.cat([imgs, masks], dim=0)
        # imgs_masks = transform_horflip(imgs_masks)
        # imgs_masks = transform_horflip(imgs_masks)

    assert torch.max(imgs_masks[:batch_size]) <= 1.0 , 'augmented image is NOT normalized'
    assert torch.max(imgs_masks[batch_size:]) <= 1.0 , 'augmented mask is NOT normalized'
    
    return imgs_masks[:batch_size], imgs_masks[batch_size:]        


def update_max_min_flows(flow, max_flow, min_flow):
    # update x flow
    if np.max(flow[..., 0]) > max_flow[0]:
        max_flow[0] = np.max(flow[..., 0])    
    if np.min(flow[..., 0] < min_flow[0]):
        min_flow[0] = np.min(flow[..., 0])

    # update y flow
    if np.max(flow[..., 1]) > max_flow[1]:
        max_flow[1] = np.max(flow[..., 1])    
    if np.min(flow[..., 1] < min_flow[1]):
        min_flow[1] = np.min(flow[..., 1])

    return max_flow, min_flow



def get_image_mask_v2(image_path, mask_path, img_name, type='train', data_type=None):
    img = cv2.imread(os.path.join(image_path, img_name))
    img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_NEAREST)

    try:
        mask = cv2.imread(os.path.join(mask_path, img_name))
        # print("mask max = ", np.max(mask))
        if data_type == 'DARPA':
            mask = find_needle_mask(mask, type=type)   
        else:
            mask = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_AREA)
            # print("mask resized max = ", np.max(mask))
            mask = mask[:,:,0]
    except:
        mask = np.zeros_like(img)
        mask = mask[:,:,0]

    return img, mask/255.0



def get_data_dict_kalman_eval(PARENT_FOLDER, LIST_OF_DATASETS, IMAGE_LOC, MASK_LOC, saved_data_file, type='test', data_type=None):

    assert data_type in ['DARPA', 'UPMC', 'BlueGel']
    all_data_dict = []
    for j, dataset_name in enumerate(LIST_OF_DATASETS):
        print("dataset # = {}, name = {}".format(j, dataset_name) )        
        image_list, needle_mask_list = [], []        
        image_path = os.path.join(PARENT_FOLDER, dataset_name, IMAGE_LOC)
        mask_path = os.path.join(PARENT_FOLDER, dataset_name, MASK_LOC)
        img_names_list = sorted(os.listdir(image_path))

        for i in range(len(img_names_list)):
            img, mask = get_image_mask_v2(image_path, mask_path, img_names_list[i], type=type, data_type=data_type) #cv2.imread(os.path.join(image_path, img_list[i]))
            image_list.append(torch.from_numpy(img[:,:,0:1]))
            needle_mask_list.append(torch.from_numpy(mask)) #mask_needle_arr_resized        

        print(" number of images in dataset {} = {} with image_list_len = {} ".format(j, i+1, len(image_list)))

        img_traj = torch.stack(image_list)
        img_traj = img_traj.permute(0,3,1,2)
        mask_traj = torch.stack(needle_mask_list).unsqueeze(3)
        mask_traj = mask_traj.permute(0,3,1,2)

        img_traj_transformed, mask_traj_transformed = transform_videos(img_traj, mask_traj, type)            

        all_data_dict.append({'images':img_traj_transformed, 'needle_masks':mask_traj_transformed})
    
    data_dir = os.path.join('saved_data', saved_data_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    torch.save(all_data_dict, os.path.join(data_dir, type+'.pt'))
    print("finished {} data generation ".format(type))

    return all_data_dict

def get_data_dict_kalman(PARENT_FOLDER, LIST_OF_DATASETS, IMAGE_LOC, MASK_LOC, saved_data_file, type='train', traj_len = 50, data_type=None):
    
    # initialise list to append stacked data from each dataset
    assert data_type in ['DARPA', 'UPMC', 'BlueGel']
    # image_lists, flow_lists, needle_mask_lists, mask_lists, flow_concat_lists = [], [], [], [], []
    all_data_dict = []
    good_pixel_count_lists = []
    n_history = 3
    count_needles = 0
    count_no_needles = 0
    # import ipdb; ipdb.set_trace()
    for j, dataset_name in enumerate(LIST_OF_DATASETS):
        print("dataset # = {}, name = {}".format(j, dataset_name) )
        image_list, needle_mask_list = [], []
        image_path = os.path.join(PARENT_FOLDER, dataset_name, IMAGE_LOC) #os.path.join(PARENT_FOLDER, dataset_name, IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')                
        mask_path = os.path.join(PARENT_FOLDER, dataset_name, MASK_LOC)
        img_names_list = sorted(os.listdir(image_path))
        # img = get_image(i, dataset_name, PARENT_FOLDER)
        img_prev = None
        for i in range(len(img_names_list)):
            img ,mask = get_image_mask_v2(image_path, mask_path, img_names_list[i], type=type, data_type=data_type)
            img_H = img.shape[0]
            img_W = img.shape[1]
            image_list.append(torch.from_numpy(img[:,:,0:1]))
            needle_mask_list.append(torch.from_numpy(mask))

        print(" number of images in dataset {} = {} with image_list_len = {} ".format(j, i+1, len(image_list)))
        # convert into video sequence and append in dictionary
        if len(image_list) < traj_len:
            print("# of frames < {} ".format(traj_len))
        else:
            step_gap = traj_len
            # for i in range(0,len(image_list),traj_len):
            for i in range(0,len(image_list),step_gap):
                idx = min(i+traj_len, len(image_list))
                img_traj = torch.stack(image_list[idx-traj_len:idx])
                # shift channels to 2nd dim
                img_traj = img_traj.permute(0,3,1,2)
                # add channel dim to masks
                mask_traj = torch.stack(needle_mask_list[idx-traj_len:idx]).unsqueeze(3)
                mask_traj = mask_traj.permute(0,3,1,2)
                # transform images and masks
                # transform_hor_flip(inp)            
                img_traj_transformed, mask_traj_transformed = transform_videos(img_traj, mask_traj, type)            

                all_data_dict.append({'images':img_traj_transformed, 'needle_masks':mask_traj_transformed})
            
    # ipdb.set_trace()
    data_dir = os.path.join('saved_data', saved_data_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    torch.save(all_data_dict, os.path.join(data_dir, type+'.pt'))
    print("finished {} data generation ".format(type))

    return all_data_dict

def find_needle_params(mask):
    # input is resized mask 
    if type(mask) == torch.Tensor:
        mask = mask.permute(1,2,0) # channel at last dim 
        mask = mask.numpy()*255
        mask = mask.astype(np.uint8)

    idx = np.where(mask[:,:,0] == 255)
    idx_y_min = np.where(idx[1] == np.min(idx[1]))
    idx_y_max = np.where(idx[1] == np.max(idx[1]))
    # print("idx_y_min = ", idx_y_min)
    # print("idx_y_max = ", idx_y_max)
    x_max = np.max(idx[0][idx_y_max])
    x_min = np.min(idx[0][idx_y_min])

    y_min = np.min(idx[1]) 
    y_max = np.max(idx[1])


    if x_min > x_max:
        # reverse the notation
        x_min, x_max = x_max, x_min
        y_min, y_max = y_max, y_min
    # print("min x = {}, min y = {}".format(x_min, y_min))
    # print("max x = {}, max y = {}".format(x_max, y_max))
    # make circle on the min and max : image will be 3 channel    
    mask = np.concatenate([mask]*3, axis=-1) 
    # mask_new = cv2.circle(mask, (y_min, x_min), 3, (0, 0, 255), 2)
    # mask_new = cv2.circle(mask, (y_max, x_max), 3, (0, 255, 0), 2)
    mask_new = cv2.line(mask, (y_min, x_min), (y_max, x_max), (0, 255, 0), 3)
    needle_length = ((x_min - x_max)**2 + (y_min - y_max)**2)**(0.5)
    needle_angle = np.arctan2(y_max-y_min, x_max-x_min)

    x_start, y_start = x_min, y_min
    x_tip, y_tip = x_max, y_max

    return mask_new, [x_start, y_start, needle_angle, needle_length, x_tip, y_tip]

def get_data_dict_kalman_vec(PARENT_FOLDER, LIST_OF_DATASETS, IMAGE_LOC, MASK_LOC, saved_data_file, type='train', traj_len=20, data_type=None, repeat_flag=True):
    #! Vector DATA 
    all_data_dict = []
    for j, dataset_name in enumerate(LIST_OF_DATASETS):
        print("dataset # " , j)
        # find image, masks
        # apply transformation and then find needle tip, length, theta
        image_list, needle_mask_list = [], []    
        image_path = os.path.join(PARENT_FOLDER, dataset_name, IMAGE_LOC)
        mask_path = os.path.join(PARENT_FOLDER, dataset_name, MASK_LOC)
        img_names_list = os.listdir(image_path)
        mask_names_list = os.listdir(mask_path)

        for i in range(len(img_names_list)):
            img,mask = get_image_mask_v2(image_path, mask_path, img_names_list[i], type=type, data_type=data_type)
            img_H = img.shape[0]
            img_W = img.shape[1]
            image_list.append(torch.from_numpy(img[:,:,0:1]))
            needle_mask_list.append(torch.from_numpy(mask))

        print(" number of images in dataset {} = {} with image_list_len = {} ".format(j, i+1, len(image_list)))
        # convert into video sequence and append in dictionary
        if len(image_list) < traj_len:
            print("# of frames < {} ".format(traj_len))
        else:
            step_gap = 3 if repeat_flag else traj_len
            for i in range(0,len(image_list),step_gap):
                idx = min(i+traj_len, len(image_list))
                img_traj = torch.stack(image_list[idx-traj_len:idx])
                # shift channels to 2nd dim
                img_traj = img_traj.permute(0,3,1,2)
                # add channel dim to masks
                mask_traj = torch.stack(needle_mask_list[idx-traj_len:idx]).unsqueeze(3)
                mask_traj = mask_traj.permute(0,3,1,2)
                # transform images and masks
                img_traj_transformed, mask_traj_transformed = transform_videos(img_traj, mask_traj, type)
                # find x_tip, y_tip, l, theta from mask_traj_transformed
                q_list = []
                needle_present_list = []
                mask_traj_new_list = []
                for b in range(traj_len):
                    mask_new, needle_params = find_needle_params(mask_traj_transformed[b])
                    mask_traj_new_list.append(mask_new)
                    q_list.append(torch.tensor(needle_params))
                    # needle_present_list.append(torch.tensor([needle_present]))

                q_list = torch.stack(q_list, dim=0)
                # needle_present_list = torch.stack(needle_present_list, dim=0)
                mask_traj_new_list = np.stack(mask_traj_new_list, axis=0)
                # ipdb.set_trace()
                all_data_dict.append({'images':img_traj_transformed, 
                                      'needle_masks_new':torch.from_numpy(mask_traj_new_list).permute(0,3,1,2), 
                                      'needle_masks':mask_traj_transformed, 
                                      'needle_params':q_list
                                      })

    # import ipdb; ipdb.set_trace()
    data_dir = os.path.join('saved_data', saved_data_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    torch.save(all_data_dict, os.path.join(data_dir, type+'.pt'))
    print("finished {} data generation \n".format(type))

    return all_data_dict


def get_data_dict_history(n_flow, PARENT_FOLDER, LIST_OF_DATASETS, IMAGE_LOC, MASK_LOC, saved_data_file, type='train', data_type='DARPA', flow_history_flag=False, flow_flag=False, max_flow=[0,0], min_flow=[100,100]):
    
    image_lists, flow_lists, needle_mask_lists, mask_lists, flow_concat_lists = [], [], [], [], []
    all_data_dict = []
    good_pixel_count_lists = []
    n_history = n_flow # only valid when flow_history_flag = True
    count_needles = 0
    count_no_needles = 0

    for j, dataset_name in enumerate(LIST_OF_DATASETS):
        print("dataset # " , j)
        image_list, flow_list, needle_mask_list, mask_list, flow_concat_list = [], [], [], [], []
        good_pixel_count_list = []
        flow_x_list, flow_y_list = [], []
        image_path = os.path.join(PARENT_FOLDER, dataset_name, IMAGE_LOC)
        mask_path = os.path.join(PARENT_FOLDER, dataset_name, MASK_LOC)
        image_name_list = sorted(os.listdir(image_path))
        img_prev = None
        for i in range(len(image_name_list)):
            # import ipdb; ipdb.set_trace()
            img, mask = get_image_mask_v2(image_path, mask_path, image_name_list[i], type=type, data_type=data_type) #get_image(i, dataset_name, PARENT_FOLDER)
            img_H = img.shape[0]
            img_W = img.shape[1]
            image_list.append(img[:,:,0:1])
            mask_list.append(mask)
            if img_prev is None:
                cartesian = np.zeros_like(img)
                # # initial flow for image_frame 1 
                flow = np.zeros_like(img[:,:, 0:1])                
                flow_new = np.concatenate([flow]*n_history, axis=-1) 
                flow_x = flow_new 
                flow_y = flow_new 
                flow_list.append(flow)
                flow_x_list.append(flow_x)
                flow_y_list.append(flow_y)
                
                if flow_history_flag or flow_flag:
                    all_data_dict.append({'images':torch.from_numpy(img[:,:,0:1]),
                                        'images_prev': torch.from_numpy(np.concatenate([np.zeros_like(img[:,:,0:1])]*n_history, axis=-1)),
                                        'flows':torch.from_numpy(flow),
                                        'needle_masks':np.expand_dims(mask, axis=-1),
                                        # 'masks':mask_resized,
                                        'flow_concats':flow_new,
                                        'flow_x':flow_x,
                                        'flow_y':flow_y,
                                        }) 
                else:
                    all_data_dict.append({'images':torch.from_numpy(img[:,:,0:1]),
                                          'images_prev':torch.from_numpy(np.concatenate([np.zeros_like(img[:,:,0:1])]*n_history, axis=-1)),
                                          'needle_masks':np.expand_dims(mask, axis=-1),
                                        })
            else:
                if flow_history_flag:
                    # computing flow b/w [T,T-1], [T,T-2], ....  forall T
                    flow, flowX, flowY = find_flow_history(image_list, img_H, img_W, n_history, abs_flag = False)
                    flow_list.append(flow)

                    # include n_history images                     
                    img_prev_history = find_image_history(image_list, n_history)

                    all_data_dict.append({'images':torch.from_numpy(img[:,:,0:1]),
                                        'images_prev':torch.from_numpy(img_prev_history),
                                        'flows':torch.from_numpy(flow[:,:,-1:]),
                                        'needle_masks':np.expand_dims(mask, axis=-1),
                                        # 'masks':mask_resized,
                                        'flow_concats':flow,
                                        'flow_x':flowX,
                                        'flow_y':flowY
                                        })                          

                elif flow_flag:
                    # this case is for flow between [T, T-1] only 
                    prvs = np.array(img_prev)[:,:,0]
                    next = np.array(img)[:,:,0]                    
                    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)                        
                    mean_flow = np.array([np.mean(flow[..., 0]), np.mean(flow[..., 1])]) #extra 
                    flow = np.abs(flow) 
                    cartesian[..., [0,2]] = cv2.normalize(flow,None, 0, 255, cv2.NORM_MINMAX)
                    cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)
                    flow_list.append(cartesian_bgr[:,:, 0:1])

                    flow_x_list.append(flow[..., 0:1]) 
                    flow_y_list.append(flow[..., 1:2])
                    # concatenate m flow frames along channel dim
                    tmp = concatenate_m_flow_frames_np(flow_list, n_flow, img_H, img_W)
                    flowX = concatenate_m_flow_frames_np(flow_x_list, n_flow, img_H, img_W)
                    flowY = concatenate_m_flow_frames_np(flow_y_list, n_flow, img_H, img_W)                                        
                
                    all_data_dict.append({'images':torch.from_numpy(img[:,:,0:1]),
                                        'images_prev':torch.from_numpy(img[:,:,0:1]),
                                        'flows':torch.from_numpy(cartesian_bgr[:,:,0:1]),
                                        'needle_masks':np.expand_dims(mask, axis=-1),
                                        # 'masks':mask_resized,
                                        'flow_concats':torch.from_numpy(tmp),
                                        'flow_x':flowX,
                                        'flow_y':flowY
                                        })   
                else:
                    if n_history > 1:
                        img_prev_history = find_image_history(image_list, n_history)
                    else:
                        img_prev_history = img[:,:,0:1]

                    assert img_prev_history.shape[-1] == n_history, 'img histroy not being created properly, check'
                    all_data_dict.append({'images':torch.from_numpy(img[:,:,0:1]),
                                          'images_prev':torch.from_numpy(img_prev_history),
                                          'needle_masks':np.expand_dims(mask, axis=-1),
                                        })
                ## compute max and min flow in x and y directions for train data only
                if type == 'train' and flow_flag:
                    max_flow, min_flow = update_max_min_flows(np.concatenate([flowX, flowY], axis=-1), max_flow, min_flow)


            img_prev = img

        # import ipdb; ipdb.set_trace()
        print(" number of images in dataset {} = {} with image_list_len = {} ".format(j, i+1, len(image_list)))

    ## update flow in entire data, normalize by max and min 
    # print("number of needle labels = " , count_needles, "  number of no needle labels = " , count_no_needles)
    if flow_flag:
        print("max_flow = " , max_flow)
        print("min_flow = " , min_flow)
        cartesian = np.zeros_like(img)
        max_flow = np.max(max_flow)
        min_flow = np.min(min_flow)
    

    for k, data in enumerate(all_data_dict):
        # NORMALIZING
        if flow_history_flag or flow_flag:
            n_flow = n_history
            updated_flow_x = (data['flow_x'] - min_flow)/max_flow 
            updated_flow_y = (data['flow_y'] - min_flow)/max_flow 
            
            flow_bgr = []
            for j in range(n_history):
                updated_flow = np.concatenate([updated_flow_x[:,:,j:j+1], updated_flow_y[:,:,j:j+1]], axis=-1)
                cartesian[..., [0,2]] = cv2.normalize(updated_flow,None, 0, 255, cv2.NORM_MINMAX)
                cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)
                flow_bgr.append(cartesian_bgr[:,:,0:1])
            
            flow_bgr = np.concatenate(flow_bgr, axis = -1)    
            
            data['flow_x'] = torch.from_numpy(updated_flow_x)
            data['flow_y'] = torch.from_numpy(updated_flow_y)
            data['flows'] = torch.from_numpy(cartesian_bgr[:,:,0:1]) #cartesian_bgr[:,:,0:1]
            data['flow_concats'] = torch.from_numpy(flow_bgr)


            out_transform = transform_all_in_same_way([data['images'].permute(2,0,1),data['images_prev'].permute(2,0,1),data['flows'].permute(2,0,1),
                                    torch.from_numpy(data['needle_masks']).permute(2,0,1), 
                                    data['flow_concats'].permute(2,0,1)], type, n_flow)

            if flow_history_flag:
                n_flow = n_history
            
            #every input has 3 channels 
            key_list = ['images', 'images_prev','flows','needle_masks','flow_concats']
            _history = 1 + n_history
            ind_list = [[0,1],[1,_history],[_history,_history+1],[_history+1,_history+2],\
                        [_history+2,_history+2+n_flow]]
                        
        else:
            out_transform = transform_all_in_same_way([data['images'].permute(2,0,1), data['images_prev'].permute(2,0,1),
                                    torch.from_numpy(data['needle_masks']).permute(2,0,1)],type, n_flow)    
            assert out_transform.shape[0] == 1+1+(n_flow-1)+1, 'check extra stuff being appended'        
            key_list = ['images', 'images_prev','needle_masks']
            ind_list = [[0,1], [1,1+n_flow], [1+n_flow,1+n_flow+1]]

        for key,ind in zip(key_list, ind_list):
            data[key] = out_transform[ind[0]:ind[1]]            

    # create dir to store data 
    data_dir = os.path.join('saved_data', saved_data_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    torch.save(all_data_dict, os.path.join(data_dir, type+'.pt'))    

    if type == 'train':
        # if training data then return max, min flow
        return all_data_dict, max_flow, min_flow 
    else:
        return all_data_dict, None, None


# def get_data_dict(n_flow, LIST_OF_DATASETS, PARENT_FOLDER, saved_data_file, type='train', flow_history_flag=False ,max_flow=[0,0], min_flow=[100,100]):
    
#     # initialise list to append stacked data from each dataset
#     image_lists, flow_lists, needle_mask_lists, mask_lists, flow_concat_lists = [], [], [], [], []
#     all_data_dict = []
#     good_pixel_count_lists = []
#     # max flow and min flow in training data only and use same in test data             
#     for j, dataset_name in enumerate(LIST_OF_DATASETS):
#         print("dataset # " , j)
#         i = 0
#         image_list, flow_list, needle_mask_list, mask_list, flow_concat_list = [], [], [], [], []
#         good_pixel_count_list = []
#         flow_x_list, flow_y_list = [], []
#         image_path = os.path.join(PARENT_FOLDER, dataset_name, IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')
#         # mask_path = os.path.join(PARENT_FOLDER, LIST_OF_DATASETS[0], MASK_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.png')
        
#         # as flow for 1st frame is zero | first frame outisde while 
#         img = get_image(i, dataset_name, PARENT_FOLDER)
#         mask, mask_resized = get_mask(i, dataset_name, PARENT_FOLDER) #Image.open(mask_path)    
#         # ipdb.set_trace()
#         if mask is not None:
#             mask_needle_arr_resized = find_needle_mask(mask)
#             cartesian = np.zeros_like(img)

#             # # initial flow for frame 1 
#             # flow_ = torch.zeros_like(img)[:,:, 0:1]
#             flow_ = np.zeros_like(img)[:,:, 0:1]
#             # flow_ = torch.zeros_like(img)[:,:, :]
#             flow_new = np.concatenate([flow_]*n_flow, axis=-1) #torch.cat([flow_]*n_flow, dim = -1)
#             flow_x = flow_new #torch.cat([flow_]*n_flow, dim = -1)
#             flow_y = flow_new #torch.cat([flow_]*n_flow, dim = -1)
#             flow_list.append(flow_)
#             flow_x_list.append(flow_x)
#             flow_y_list.append(flow_y)
            
#             all_data_dict.append({'images':img[:,:,0:1],
#                                   'images_prev':np.zeros_like(img[:,:,0:1]),
#                                 'flows':flow_, 
#                                 'needle_masks':mask_needle_arr_resized,
#                                 'masks':mask_resized,
#                                 'flow_concats':flow_new,
#                                 'flow_x':flow_x,
#                                 'flow_y':flow_y
#                                 })

#             # finding number of good pixels
#             _ , count_ = np.unique(mask_needle_arr_resized, return_counts=True) 
#             if count_[1] != 0:
#                 good_pixel_count_list.append(count_[0]/count_[1])

#         img_H = img.shape[0]
#         img_W = img.shape[1]
#         while os.path.exists(image_path):            
#             # print("i = " , i)
#             i = i + 1            
#             img_next = get_image(i, dataset_name, PARENT_FOLDER) #Image.open(image_path)        
#             if img_next is None:
#                 print("done")
#                 break
#             mask, mask_resized = get_mask(i, dataset_name, PARENT_FOLDER) #Image.open(mask_path)            
#             # if mask is none then don't append images
#             if mask is not None:                
#                 mask_needle_arr_resized = find_needle_mask(mask)
#                 # if needle mask is none don't append
#                 if mask_needle_arr_resized is not None:
#                     # append image, mask, mask_needle                    
#                     image_list.append(img_next[:,:,0:1])   
#                     needle_mask_list.append(mask_needle_arr_resized)
#                     mask_list.append(mask)                                               
#                     _ , count_ = np.unique(mask_needle_arr_resized, return_counts=True) 
#                     if count_[1] != 0:
#                         good_pixel_count_list.append(count_[0]/count_[1])                        
                    
#                     #compute optical flow and stack in groups of n
#                     prvs = np.array(img)[:,:,0]
#                     next = np.array(img_next)[:,:,0]


#                     flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)                        
#                     mean_flow = np.array([np.mean(flow[..., 0]), np.mean(flow[..., 1])]) #extra 
#                     flow = np.abs(flow) 
#                     cartesian[..., [0,2]] = cv2.normalize(flow,None, 0, 255, cv2.NORM_MINMAX)
#                     cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)
#                     flow_list.append(cartesian_bgr[:,:, 0:1]) #torch.from_numpy(cartesian_bgr)[:,:, 0:1]

#                     # # to get a colored flow
#                     # # hsv[..., [0,2]] = cv2.normalize(flow-mean_flow,None, 0, 255, cv2.NORM_MINMAX)
#                     # # cartesian_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    
#                     flow_x_list.append(flow[..., 0:1]) #torch.from_numpy(flow[..., 0:1])
#                     flow_y_list.append(flow[..., 1:2]) # torch.from_numpy(flow[..., 1:2])
#                     # concatenate m flow frames along channel dim
#                     tmp = concatenate_m_flow_frames_np(flow_list, n_flow, img_H, img_W)
#                     ## INPUTTING NOISE                     
#                     # tmp = torch.randint(0, 255, (img_W, img_H, 1))
#                     tmpX = concatenate_m_flow_frames_np(flow_x_list, n_flow, img_H, img_W)
#                     tmpY = concatenate_m_flow_frames_np(flow_y_list, n_flow, img_H, img_W)                                        
                
#                     all_data_dict.append({'images':img_next[:,:,0:1],
#                                          'images_prev':img[:,:,0:1],
#                                         'flows':cartesian_bgr[:,:,0:1], 
#                                         'needle_masks':mask_needle_arr_resized,
#                                         'masks':mask_resized,
#                                         'flow_concats':tmp,
#                                         'flow_x':tmpX,
#                                         'flow_y':tmpY
#                                         })   

#                     ## find max and min flow in x  and y directions for train data only
#                     if type == 'train':
#                         max_flow, min_flow = update_max_min_flows(flow, max_flow, min_flow)
#                 # update previous image
#                 img = img_next

#     # ipdb.set_trace()
#     ## update flow in all entire data, normalize by global max and min 
#     print("max_flow = " , max_flow)
#     print("min_flow = " , min_flow)
#     cartesian = np.zeros_like(img)
#     max_flow_ = np.max(max_flow)

#     for k, data in enumerate(all_data_dict):
#         # NORMALIZING
#         updated_flow_x = (data['flow_x'] - min_flow[0])/max_flow_ #(max_flow[0] - min_flow[0])
#         updated_flow_y = (data['flow_y'] - min_flow[1])/max_flow_ #(max_flow[1] - min_flow[1])     
#         # print("flow x shape : " , updated_flow_x.shape)   
#         # print("flow y shape : " , updated_flow_y.shape) 
#         range_ = [k-n_flow+1,k+1] if k > n_flow else [0,k+1]  
#         flow_bgr = []
#         for j in range(n_flow):
#             updated_flow = np.concatenate([updated_flow_x[:,:,j:j+1], updated_flow_y[:,:,j:j+1]], axis=-1)
#             cartesian[..., [0,2]] = cv2.normalize(updated_flow,None, 0, 255, cv2.NORM_MINMAX)
#             cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)
#             flow_bgr.append(cartesian_bgr[:,:,0:1])
        
#         flow_bgr = np.concatenate(flow_bgr, axis = -1)    
        
#         data['flow_x'] = torch.from_numpy(updated_flow_x)
#         data['flow_y'] = torch.from_numpy(updated_flow_y)
#         data['flows'] = torch.from_numpy(cartesian_bgr[:,:,0:1]) #cartesian_bgr[:,:,0:1]
#         data['flow_concats'] = torch.from_numpy(flow_bgr)

#         out_transform = transform_all_in_same_way([data['images'].permute(2,0,1), data['images_prev'].permute(2,0,1),data['flows'].permute(2,0,1),
#                                 torch.from_numpy(data['needle_masks']).unsqueeze(-1).permute(2,0,1),torch.from_numpy(data['masks']).permute(2,0,1), 
#                                 data['flow_concats'].permute(2,0,1), data['flow_x'].permute(2,0,1), data['flow_y'].permute(2,0,1)], type)
#         data['images'] = out_transform[0:1,:,:]
#         data['immages_prev'] = out_transform[1:2,:,:]
#         data['flows'] = out_transform[2:3,:,:]
#         data['needle_masks'] = out_transform[3:4,:,:]
#         data['masks'] = out_transform[4:7,:,:]
#         data['flow_concats'] = out_transform[7:7+n_flow,:,:]
#         data['flow_x'] = out_transform[7+n_flow:7+2*n_flow,:,:]
#         data['flow_y'] = out_transform[7+2*n_flow:7+3*n_flow,:,:]

#     data_dir = os.path.join('saved_data', saved_data_file)
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)

#     torch.save(all_data_dict, os.path.join(data_dir, type+'.pt'))
#     # ipdb.set_trace()

#     if type == 'train':
#         return all_data_dict, max_flow, min_flow 
#     else:
#         return all_data_dict, None, None

def get_list_train_test_data(data_type):

    assert data_type in ['DARPA' , 'UPMC', 'BlueGel'], 'check datatype'
    if data_type == 'DARPA':
        PARENT_FOLDER_TRAIN = 'new_dataset' 
        PARENT_FOLDER_TEST = 'new_dataset/test' 

        IMAGE_LOC  = 'JPEGImages'
        MASK_LOC = 'SegmentationClass'

        # LIST_OF_DATASETS_TRAIN = [
        #                         'task_negatives_136-2022_04_21_17_19_17-segmentation mask 1.1',
        #                         'task_positives_11-2022_04_12_18_28_14-segmentation mask 1.1',
        #                         'task_positives_153-2022_04_12_18_32_22-segmentation mask 1.1',
        #                         'task_positives_189-2022_04_12_18_32_54-segmentation mask 1.1',
        #                         'task_positives_193-2022_04_18_19_26_03-segmentation mask 1.1',
        #                         'task_positives_222-2022_04_19_22_31_15-segmentation mask 1.1',
        #                         'task_positives_94-2022_04_19_22_08_47-segmentation mask 1.1',
        #                         'task_positives_134-2022_04_12_18_31_26-segmentation mask 1.1',
        #                         'task_negatives_191-2022_04_12_18_49_18-segmentation mask 1.1',
        #                         'task_positives_240-2022_04_22_19_59_15-segmentation mask 1.1',
        #                         'task_negatives_189-2022_04_12_18_48_43-segmentation mask 1.1',
        #                         'task_negatives_179-2022_04_12_18_47_47-segmentation mask 1.1',
        #                         'task_positives_205-2022_04_21_17_04_12-segmentation mask 1.1'
        #                         ]

        # LIST_OF_DATASETS_TEST = [
        #                         'task_positives_93-2022_04_12_18_28_39-segmentation mask 1.1',                        
        #                         'task_negatives_204-2022_04_08_16_58_31-segmentation mask 1.1',
        #                         'task_positives_67-2022_04_18_19_11_58-segmentation mask 1.1'
        #                         ]
        
        #* CROSS 1
        LIST_OF_DATASETS_TRAIN = [
                                'task_negatives_136-2022_04_21_17_19_17-segmentation mask 1.1',
                                'task_positives_11-2022_04_12_18_28_14-segmentation mask 1.1',
                                'task_positives_153-2022_04_12_18_32_22-segmentation mask 1.1',
                                'task_positives_189-2022_04_12_18_32_54-segmentation mask 1.1',
                                'task_positives_193-2022_04_18_19_26_03-segmentation mask 1.1',
                                'task_positives_222-2022_04_19_22_31_15-segmentation mask 1.1',
                                'task_positives_94-2022_04_19_22_08_47-segmentation mask 1.1',
                                'task_positives_134-2022_04_12_18_31_26-segmentation mask 1.1',
                                'task_negatives_191-2022_04_12_18_49_18-segmentation mask 1.1',
                                # 'task_positives_240-2022_04_22_19_59_15-segmentation mask 1.1', # moved to test
                                'task_negatives_189-2022_04_12_18_48_43-segmentation mask 1.1',
                                'task_negatives_179-2022_04_12_18_47_47-segmentation mask 1.1',
                                'task_positives_205-2022_04_21_17_04_12-segmentation mask 1.1', #! this is test dataset
                                'task_negatives_204-2022_04_08_16_58_31-segmentation mask 1.1' ,
                                'task_positives_93-2022_04_12_18_28_39-segmentation mask 1.1',
                                ]

        LIST_OF_DATASETS_TEST = [
                                'task_positives_240-2022_04_22_19_59_15-segmentation mask 1.1',
                                'task_negatives_179-2022_04_12_18_47_47-segmentation mask 1.1',
                                'task_positives_67-2022_04_18_19_11_58-segmentation mask 1.1'
                                ]




    elif data_type == 'BlueGel':
        PARENT_FOLDER_TRAIN = 'new_dataset/BlueGelData' 
        PARENT_FOLDER_TEST = 'new_dataset/BlueGelData' 

        IMAGE_LOC  = 'image'
        MASK_LOC = 'image_masks'

        LIST_OF_DATASETS_TRAIN = ['01-straight-mar-05-trial-1',
                                '02-straight-mar-05-trial-4',
                                '03-small-mar-05-trial-2',
                                '04-small-mar-05-trial-1',
                                '06-medium-mar-05-trial-6',
                                '07-small-mar-06-trial-3',
                                '08- medium-mar-06-trial-1',
                                '10-straight-mar-09-trial-1',
                                '11-straight-mar-09-trial-4',
                                '12-straight-mar-09-trial-7',
                                '14-small-mar-09-trial-3',
                                '15-small-mar-09-trial-6',
                                '16-medium-mar-09-trial-2',
                                # '18-medium-mar-09-trial-5', 
                                '19-medium-mar-09-trial-9', 
                                '20-small-mar-09-trial-8',
                                '21-large-mar-11-trial-2', 
                                '22-large-mar-11-trial-7', 
                                '23-large-mar-11-trial-9',
                                '24-large-mar-11-trial-12', 
                                # '25-large-mar-11-trial-16', # bad labels
                                '26-large-mar-11-trial-17',
                                '27-large-mar-11-trial-18'
                                ]


        LIST_OF_DATASETS_TEST = [
                                '05-small-mar-05-trial-4',                                 
                                '09-medium-mar-06-trial-2', 
                                '13-straight-mar-09-trial-9',
                                '17-medium-mar-09-trial-4',  
                                '18-medium-mar-09-trial-5'                                 
                                ]

    elif data_type == 'UPMC':
        PARENT_FOLDER_TRAIN = 'new_dataset/PigLabData' 
        PARENT_FOLDER_TEST = 'new_dataset/PigLabData' 

        IMAGE_LOC  = 'image'
        MASK_LOC = 'image_masks'

        LIST_OF_DATASETS_TRAIN = ['01','02','03','04','05']
        LIST_OF_DATASETS_TEST = ['00']
    
    
    return PARENT_FOLDER_TRAIN,LIST_OF_DATASETS_TRAIN, PARENT_FOLDER_TEST, LIST_OF_DATASETS_TEST, IMAGE_LOC, MASK_LOC



if __name__ == '__main__':

    #! below is not used and was just for debugging
    n_flow = 3
    i = 1
    image_list, flow_list, needle_masks_list, mask_list, flow_concat_list = [], [], [], [], []
    image_path = os.path.join(IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')

    # as flow needs to be computed first frame outisde while 
    img = Image.open(image_path)
    img_tensor = convert_tensor(img)
    image_list.append(img_tensor)
    hsv = np.zeros_like(img)
    cartesian = np.zeros_like(img)
    hsv[..., 1] = 255

    flow_ = torch.zeros_like(img_tensor)[0:2,:,:]
    flow_ = torch.cat([flow_]*3, dim = 0)
    flow_concat_list.append(flow_)

    img_H = img_tensor.shape[1]
    img_W = img_tensor.shape[2]

    # ipdb.set_trace()
    while os.path.exists(image_path):
        # save images
        # image_path = os.path.join(IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')

        # next image frame
        image_path = os.path.join(IMAGE_LOC, 'frame_' + '000{}'.format(i+1).zfill(6) + '.PNG')
        img2 = Image.open(image_path)    
        
        # compute optical flow and stack in groups of n
        
        prvs = np.array(img)[:,:,0]
        next = np.array(img2)[:,:,0]

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        print("mean of flows: " , np.mean(flow[..., 0]), " " , np.mean(flow[..., 1]))
        mean_flow = np.array([np.mean(flow[..., 0]), np.mean(flow[..., 1])])
        cartesian[..., 0:2] = cv2.normalize(flow-mean_flow,None, 0, 255, cv2.NORM_MINMAX)
        # cartesian[..., 0] = cv2.normalize(flow[..., 0],None, 0, 255, cv2.NORM_MINMAX)
        # cartesian[..., 2] = cv2.normalize(flow[..., 1],None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite('flow_cartesian/flow_cartesian_{}.png'.format(i), cartesian)
        flow_list.append(convert_tensor(cartesian)[0:2,:,:])

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('flow_polar/flow_polar_{}.png'.format(i), bgr)
        # read segmentation masks
        mask_path = os.path.join(MASK_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.png')
        mask = Image.open(mask_path)
        # mask.save('segmentation_mask.png')

        # get just needle masks
        mask_arr = np.array(mask)
        mask_tensor = convert_tensor(mask)
        mask_list.append(mask_tensor)
        # print("shape mask array", mask_arr.shape) 
        
        # finding just needle mask
        mask_needle = 255*(np.array([255,255,255]) - np.where(mask_arr == 51,221,255))
        mask_needle = np.concatenate([np.expand_dims(mask_needle[:,:,0], axis=-1)]*3, axis=-1)
        mask_needle_ = Image.fromarray(mask_needle.astype('uint8'))
        # print("needle mask shape: ", mask_needle_.size)
        mask_needle_.save('needle_mask/pure_needle_{}.png'.format(i))
        needle_masks_list.append(convert_tensor(mask_needle_)[0:1,:,:])
        
        # save combined needle_mask and optical flow
        # ipdb.set_trace()
        flow_mask = cv2.vconcat([mask_needle.astype('uint8'), cartesian])
        cv2.imwrite('flow_and_mask/{}.png'.format(i), flow_mask)

        i += 1
        image_path = os.path.join(IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')


        # update previous image
        img = img2
        # store image 
        img_tensor = convert_tensor(img)
        image_list.append(img_tensor)

        # ipdb.set_trace()
        # concatenate m flow frames along channel dim
        if len(flow_list) >= 3:
            tmp = torch.cat(flow_list[-3:], dim=0)
            ipdb.set_trace()  
        else:
            tt1 = torch.cat([torch.zeros(2, img_H, img_W)]*(n_flow-len(flow_list)), dim=0)
            tt2 = torch.cat(flow_list, dim=0)
            tmp = torch.cat([tt2,tt1], dim=0)
        flow_concat_list.append(tmp)
        if i > 10:
            print("stopping")
            ipdb.set_trace()
            break