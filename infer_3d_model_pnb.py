
from __future__ import division, print_function
import os, re, math, cmath
from glob import glob
import numpy as np
import cv2
import tensorflow as tf
from bayesian_unet_3d_kg import BayesUNet3DKG
import imutils
import matplotlib.patches as patches
from imutils import perspective
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pandas as pd
import ipdb

SLIDING_WINDOW = False

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def isleft(p1, p2, p3): #check if p3 is to the left of the line defined by p1 and p2
    p1, p2 = p2, p1
    position = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) < 0)
    return position

def update_map(img, p1, p2, p2_1, p2_2): #rescale the map and add 0.3 to the areas inside needle cone

    x1, y1 = p1
    x2, y2 = p2

    if p2_1[0] < p2_2[0]:
        p2_1, p2_2 = p2_2, p2_1 #p2_1 is the left line (downwards)

    mid_x, mid_y = int((x2 + x1)/2), int((y2 + y1)/2)

    img = img.astype(float) / 256.
    img *= 0.7 #scale pixel values
    updated_map = img

    for i in range(updated_map.shape[0]):
        for j in range(updated_map.shape[1]):   
            if (isleft((mid_x, mid_y), p2_2, (i, j)) and not isleft((mid_x, mid_y), p2_1, (i, j))) or (not isleft((mid_x, mid_y), p2_2, (i, j)) and isleft((mid_x, mid_y), p2_1, (i, j))):
                updated_map[j,i] += 0.3
    return updated_map


def get_h5_file(parent_dir):
    dir_contents = os.listdir(parent_dir)
    assert len(dir_contents) == 1, "There is more than 1 file in this model folder path. Please reduce it down to just the h5 file."
    assert dir_contents[0].endswith('.h5'), "There is no h5 file stored in this model folder path. Please check again."
    return os.path.join(parent_dir, dir_contents[0])

def create_model(model_name, num_classes, depth, dropout=0.50, batch_norm=True):
    print("Creating model now")

    if model_name == '_Bayes3DUNetKG_':
        model = BayesUNet3DKG(depth=depth,
                              num_classes=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout, 
                              dropout_type='block',
                              regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.001))
    else:
        raise NotImplementedError(model_name + " Not implemented")

    print("Created model")
    return model

def preprocess_img(img, img_start = [0,0], img_end = [774,512]):
    if len(img.shape) > 2:
        img_out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_out = img
    img_out = cv2.resize(img_out, dsize=(256, 256), interpolation = cv2.INTER_NEAREST)
    return img_out[:,:, np.newaxis].astype(np.float32)

def calculate_distance(needle, nerve, vessel, shape, type = 'centroid'):
    needle_thresh, nerve_thresh, vessel_thresh = 0, 0, 0 #minimum contour area 
    edged = (needle + nerve).astype(np.uint8)
    needle = needle.astype(np.uint8)
    nerve = nerve.astype(np.uint8)
    vessel = vessel.astype(np.uint8)
    # edged = np.expand_dims(edged, axis = 0)
    # edged = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
    cnts_edged, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts_needle, _ = cv2.findContours(needle.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts_nerve, _ = cv2.findContours(nerve.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts_vessel, _ = cv2.findContours(vessel.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    # cnts_edged = imutils.grab_contours(cnts_edged)
    img = cv2.drawContours(edged, cnts_edged, -1, (0, 255, 0), 3)
    cv2.imwrite('/home/nthumbav/Downloads/temp.png', img)

    # for c in cnts:
	# # if the contour is not sufficiently large, ignore it
    #     if cv2.contourArea(c) < 100:
    #         continue
    min_dist = np.Inf

    if type == 'centroid':
        needle_centroids, nerve_centroids, vessel_centroids = [], [], []
        for c in cnts_needle:
            if cv2.contourArea(c) < needle_thresh:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            needle_centroids.append((cX, cY))

        for c in cnts_nerve:
            if cv2.contourArea(c) < nerve_thresh:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            nerve_centroids.append((cX, cY))
        
        for c in cnts_vessel:
            if cv2.contourArea(c) < vessel_thresh:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            vessel_centroids.append((cX, cY))
        
        for cent in needle_centroids:
            cent = np.array([cent])
            cent = np.repeat(cent, len(nerve_centroids), axis = 0)
            try:
                distances = np.linalg.norm(cent - nerve_centroids, axis = 1)
                if np.amin(distances) == 0 or np.amin(distances) == 1.:
                    continue
                min_dist = min(np.amin(distances), min_dist)
            except:
                continue
        
        if min_dist == np.Inf:
            for cent in needle_centroids:
                cent = np.array([cent])
                try:
                    cent = np.repeat(cent, len(vessel), axis = 0)
                    distances = np.linalg.norm(cent - vessel_centroids, axis = 1)
                    min_dist = min(np.amin(distances), min_dist)
                except:
                    continue
    
    elif type == 'pointwise':
        for cnt_needle in cnts_needle:
            for cnt_nerve in cnts_nerve:
                cnt_needle, cnt_nerve = np.squeeze(cnt_needle), np.squeeze(cnt_nerve)
                for pt in range(cnt_needle.shape[0]):
                    needle_pt = np.expand_dims(cnt_needle[pt], axis = 0)
                    needle_pt = np.repeat(needle_pt, cnts_nerve.shape[0], axis = 0)
                    distances = np.linalg.norm(needle_pt - cnt_nerve, axis = 0)
                    min_dist = min(np.amin(distances), min_dist)

        if min_dist == np.Inf:
            for cnt_needle in cnts_needle:
                for cnt_nerve in cnts_vessel:
                    cnt_needle, cnt_vessel = np.squeeze(cnt_needle), np.squeeze(cnts_vessel)
                    for pt in range(cnt_needle.shape[0]):
                        needle_pt = np.expand_dims(cnt_needle[pt], axis = 0)
                        needle_pt = np.repeat(needle_pt, cnt_vessel.shape[0], axis = 0)
                        distances = np.linalg.norm(needle_pt - cnt_vessel, axis = 0)
                        min_dist = min(np.amin(distances), min_dist)


    return min_dist

def postprocess_img(img, minLineLength = 0, maxLineGap = 100):
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    lines = cv2.HoughLinesP(gray, 1, np.pi/180, 10, minLineLength, maxLineGap)
    updated_map = img
    if lines is not None:
        for x1,y1,x2,y2 in lines[0]:
            mid_x, mid_y = int((x2 + x1)/2), int((y2 + y1)/2)
            angle = -int(math.atan((mid_y-y2)/(x2-mid_x))*180/math.pi)
            
            length = 10000 #some large number
            angle_1, angle_2 = math.radians(angle - 20), math.radians(angle + 20)
            # x2_2 = int(mid_x + length * math.cos(math.radians(angle_2)))
            # y2_2 = int(mid_y + length * math.sin(math.radians(angle_2)))
            # x2_1 = int(mid_x + length * math.cos(math.radians(angle_1)))
            # y2_1 = int(mid_y + length * math.sin(math.radians(angle_1)))
            pt1 = cmath.rect(length, angle_1)  
            x2_1 = int(pt1.real + mid_x  )
            y2_1 = int(pt1.imag + mid_y)
            
            pt2 = cmath.rect(length, angle_2)  
            x2_2 = int(pt2.real + mid_x)  
            y2_2 = int(pt2.imag + mid_y)

            updated_map = update_map(img, (x1, y1), (x2, y2), (x2_1, y2_1),(x2_2, y2_2))
        
    return updated_map


def load_model(input_image_shape, n_input_frames, n_classes, depth, model_name, model_epoch, filepath, dropout=0.50, batch_norm=True):
    print("Loading Model")
    # model_path = os.path.join(os.path.dirname(os.path.abspath(filepath)), "model_weights")
    model_path = filepath
    model = create_model(model_name, n_classes, depth, dropout, batch_norm)
    if n_input_frames == 1:
        input_shape = (None, input_image_shape[0], input_image_shape[1], input_image_shape[2])
    else:
        input_shape = (None, n_input_frames, input_image_shape[0], input_image_shape[1], input_image_shape[2])
    model.build(input_shape) # need to call first before loading weights (for H5 format)
    model.load_weights(get_h5_file(os.path.join(model_path, model_epoch)), by_name=False) # must manually set names
    return model

def perform_inference(input_list, model, model_name, n_mc_samples, is_training=True):
    """Can set to is_training to false if batch statistics have converged properly, otherwise true is better"""
    print("Predicting Output")
    epi_mean, epi_var = None, None
    if model_name == '_Bayes3DUNetKG_':
        sequence_input = np.expand_dims(np.stack(input_list, axis=0), axis=0)
        # epistemic uncertainty
        all_means = None
        for i in range(n_mc_samples):
            logits_mean, _, logits_logvar, _ = model(sequence_input, is_training=is_training, dropout_training=True)
            logits_mean = tf.expand_dims(logits_mean, 0)
            if all_means is None:
                all_means = logits_mean
            else:
                all_means = tf.concat([all_means, logits_mean], axis=0)
        epi_mean, epi_var = tf.nn.moments(all_means, axes=[0])
    else:
        raise NotImplementedError
    
    return epi_mean, epi_var

def main(filepath, model_folder, image_output_loc, ground_truth_path):
    
    model = create_model(num_classes= 4, depth = 4, model_name = '_Bayes3DUNetKG_')
    model = load_model(input_image_shape = (256,256,1), n_input_frames = 8, n_classes = 4, depth = 4, model_name = '_Bayes3DUNetKG_', model_epoch = '2_best_bn_t', filepath = model_folder)
    images, count, total_count, image_numpy = [], 0, 0, np.zeros((8, 256, 256, 1))
    frames, needle_maps, nerve_maps, vessel_maps = [], [], [], []

    # if video then parse out images else load images from filepath : syntax  'frame_' + '000{}'.format(1).zfill(6) + '.PNG'
    if 'mp4' in filepath:
        video_input = True
        video_file = filepath
        vid_file = cv2.VideoCapture(video_file)
        success, image = vid_file.read()
    else:
        video_input = False
        
    # ipdb.set_trace()
    # while success:
    for i in range(1,596): # total 595 images
        image_name = 'frame_' + '000{}'.format(i).zfill(6) + '.PNG'
        image_path = os.path.join(filepath, image_name)
        image = cv2.imread(image_path)
        if not SLIDING_WINDOW:
            if count == 8:
                # ipdb.set_trace()
                epi_mean, epi_var = perform_inference(image_numpy, model, '_Bayes3DUNetKG_', 1)
                # model.save('models')
                # exit()
                model_output = tf.nn.softmax(epi_mean).numpy()
                needle_softmax = model_output[0,:,:,:,1]*255.
                needle_softmax = needle_softmax.astype(np.uint8)
                nerve_softmax = model_output[0,:,:,:,2]*255.
                nerve_softmax = nerve_softmax.astype(np.uint8)
                vessel_softmax = model_output[0,:,:,:,3]*255.
                vessel_softmax = vessel_softmax.astype(np.uint8)
                for i in range(8):
                    needle_frame = needle_softmax[i,:,:]
                    updated_map = postprocess_img(needle_frame)
                    # cv2.imwrite(image_output_loc + '{}.png'.format(total_count), updated_map)
                    # needle_maps.append(updated_map) # comment out for raw outputs
                    needle_maps.append(needle_frame)
                    nerve_maps.append(nerve_softmax[i,:,:])
                    vessel_maps.append(vessel_softmax[i,:,:])
                    frames.append(image_numpy[i,:,:,:])

                count = 0
                # flow_frames = dense_optical_flow(frames = image_numpy)
                # pass
        # print("image shape before preprocessing: ", image.shape)
        image_numpy[count, :, :, :] = preprocess_img(image)        
        # ipdb.set_trace()
        # print("image shape AFTER preprocessing: ", image_numpy[0].shape)
        if video_input:
            success, image = vid_file.read()
        

        count += 1      
        total_count += 1

    # SAVE OG IMAGES and Their NEEDLE OVERLAP 
    for i in range(len(frames)):
        print("i = ", i)
        # cv2.imwrite(image_output_loc + 'og_{}.png'.format(i), frames[i])
        # write needle_maps on frames and save
        frames_ = np.concatenate([frames[i]]*3,axis=-1)
        frames_[:,:,0] = needle_maps[i]
        # cv2.imwrite(image_output_loc + 'og_needle_overlayed_{}.png'.format(i), frames_)
        frame_tmp = cv2.vconcat([np.concatenate([frames[i]]*3,axis=-1), frames_])
        # cv2.imwrite(image_output_loc + 'og_needle_{}.png'.format(i), frame_tmp)

        # PLOT GROUND TRUTH FRAMES AS WELL 
        gt_image_name = 'frame_' + '000{}'.format(i+1).zfill(6) + '.png'
        gt_image_path = os.path.join(ground_truth_path, gt_image_name)
        gt_image = cv2.imread(gt_image_path).astype(np.float32)
        gt_image = cv2.resize(gt_image, dsize=(256, 256), interpolation = cv2.INTER_NEAREST)
        # ipdb.set_trace()
        frame_tmp_ = cv2.vconcat([frame_tmp.astype(np.float32), gt_image])
        cv2.imwrite(image_output_loc + 'og_needle_{}.png'.format(i), frame_tmp_)


    return frames, needle_maps, nerve_maps, vessel_maps
    
if __name__ == '__main__':

    start = time.time()
    data_folder_path = 'data/task_positives_11-2022_04_12_18_28_14-segmentation mask 1.1/'
    # filepath = 'test_video.mp4'
    filepath = os.path.join(data_folder_path, 'JPEGImages')
    ground_truth_path = os.path.join(data_folder_path, 'SegmentationClass')

    model_folder = 'bayes_3d_kg_pnb_e_5_l_bayesian_combined_kg_f_8_model_bayes_3d_unet_kg_4-26-2022-14-36'

    image_output_loc = 'exp/3/'
    # location to save output images     
    if not os.path.exists(image_output_loc):
        os.makedirs(image_output_loc)
    
    frames, needle_maps, nerve_maps, vessel_maps =  main(filepath, model_folder, image_output_loc, ground_truth_path)
    
    maps_time = time.time()

    print('Time Taken to compute maps : ', maps_time - start)


    min_dist = np.Inf
    for i in range(len(needle_maps)):
        curr_dist = calculate_distance(needle_maps[i], nerve_maps[i], vessel_maps[i], (256,256), 'centroid')
        min_dist = min(curr_dist, min_dist)
    
    end = time.time()

    print('Time taken to compute minimum distance : ', end - maps_time)

    print(min_dist)

    '''
    Run model for all videos - comment out
    '''
    # run = 'positive'

    # if run == 'positive':
    #     root_dir = '../Positives/'
    #     dest_csv = '../pnb_distances_positive.csv'
 
    # else:
    #     root_dir = '../Negatives/'
    #     dest_csv = '../pnb_distances_negative.csv'

    # folders, distances = [], []

    # df = pd.DataFrame(columns=['Folder', 'Distance'])

    # for filepath in tqdm(os.listdir(root_dir)):
    #     folder_path = os.path.join(root_dir, filepath)
    #     vid_file = glob(os.path.join(folder_path, "*.mp4"))

    #     assert len(vid_file) == 1

    #     frames, needle_maps, nerve_maps, vessel_maps =  main(filepath = vid_file[0])

    #     min_dist = np.Inf
    #     for i in range(len(needle_maps)):
    #         curr_dist = calculate_distance(needle_maps[i], nerve_maps[i], vessel_maps[i], (256,256), 'centroid')
    #         min_dist = min(curr_dist, min_dist)

    #     print(min_dist)

    #     folders.append(filepath)
    #     distances.append(min_dist)

        
    # df['Folder'] = folders
    # df['Distance'] = distances
    
    # df.to_csv(dest_csv)



