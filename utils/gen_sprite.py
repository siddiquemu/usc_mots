import os
import os.path as path
import argparse
import subprocess
from joblib import Parallel, delayed
import multiprocessing
import math
import numpy as np
import torch
import cv2
import sys
sys.path.append("modules")
#
import modules.utils as utils
import matplotlib.pyplot as plt
import torchvision as torchvision
import cv2
import matplotlib.cm as cm
from PIL import Image, ImageDraw
from mask_resize import masks_resize

eval_type = 'test'
metric = 1
N = 1 if metric == 1 else 64
if eval_type=='train':
    T = 500
    H = 256
    W = 256
    D = 3
    h = 21
    w = 21
    O = 5
    frame_num = 10000 if metric == 1 else 2e6 #1e4
    train_ratio = 0 if metric == 1 else 0.96
    birth_prob = 0.5
    appear_interval = 5
    scale_var = 0.1
    ratio_var = 0.2
    velocity = 5.3
    task = 'sprite'
    m = h // 2
    eps = 1e-5

if eval_type == 'test':
    T = 20
    H = 128
    W = 128
    D = 3
    h = 21
    w = 21
    O = 3
    frame_num = 1e4 if metric == 1 else 2e6
    train_ratio = 0 if metric == 1 else 0.96
    birth_prob = 0.5
    appear_interval = 5
    scale_var = 0.1
    ratio_var = 0.2
    velocity = 5.3
    task = 'sprite'
    m = h // 2
    eps = 1e-5

txt_name = task + 'gt.txt'
metric_dir = 'metric' if metric == 1 else ''
output_dir = path.join('data', task, 'pt', metric_dir)
output_input_dir = path.join(output_dir, 'input')
utils.rmdir(output_input_dir); utils.mkdir(output_input_dir)
output_gt_dir = path.join(output_dir, 'gt')

# color template
color_num = 6
color_temp = torch.ByteTensor(color_num, 1, 1, D).zero_()
for i in range(0, color_num):
    R = math.floor((i+1)/4) * 255
    G = math.floor(((i+1)%4)/2) * 255
    B = (i+1)%2 * 255
    color_temp[i, :, :, 2].fill_(R)
    color_temp[i, :, :, 1].fill_(G)
    color_temp[i, :, :, 0].fill_(B)

# shape template
shape_num = 4
shape_temp = torch.ByteTensor(shape_num, h, w).zero_()
# circle
circle = shape_temp[0]
center = (h - 1) / 2
radius = h / 2
for i in range(0, h):
    for j in range(0, w):
        if math.pow(i - center, 2) + math.pow(j - center, 2) <= radius * radius:
            circle[i, j] = 255
# rectangle
rectangle = shape_temp[1]
rectangle.fill_(255)
# triangle
triangle = shape_temp[2]
for i in range(0, h):
    for j in range(0, w):
        if j <= w/2 - 1:
            if (h - i) / (j + 1) <= h / (w / 2): 
                triangle[i, j] = 255
        else:
            if (h - i) / (w - j) <= h / (w / 2): 
                triangle[i, j] = 255
# diamond
diamond = shape_temp[3]
for i in range(0, h):
    for j in range(0, w):
        if math.fabs(i - center) + math.fabs(j - center) <= radius:
            diamond[i, j] = 255

# generate data from trackers
train_frame_num = frame_num * train_ratio
test_frame_num = frame_num * (1 - train_ratio)
print('train frame number: ' + str(train_frame_num))
print('test frame number: ' + str(test_frame_num))
batch_nums = {
    'train': math.floor(train_frame_num / (N * T)),
    'test': math.floor(test_frame_num / (N * T))
}


core_num = 1 if metric == 1 else multiprocessing.cpu_count()
oid = 0 # object id
print("Running with " + str(core_num) + " cores.")
if metric == 1:
    utils.mkdir(output_gt_dir)
    file = open(path.join(output_dir, txt_name), "w")
def process_batch(states, batch_id):
    global oid
    masks_f = []
    boxs_f = []

    buffer_big = torch.ByteTensor(2, H + 2 * h, W + 2 * w, D).zero_()
    org_seq = torch.ByteTensor(T, H, W, D).zero_()
    # sample all the random variables
    unif = torch.rand(T, O)
    color_id = torch.rand(T, O).mul_(color_num).floor_().long()
    shape_id = torch.rand(T, O).mul_(shape_num).floor_().long()
    direction_id = torch.rand(T, O).mul_(4).floor_().long() # [0, 3]
    position_id = torch.rand(T, O, 2).mul_(H-2*m).add_(m).floor_().long() # [m, H-m-1]
    scales = torch.rand(T, O).mul_(2).add_(-1).mul_(scale_var).add_(1) # [1 - var, 1 + var]
    ratios = torch.rand(T, O).mul_(2).add_(-1).mul_(ratio_var).add_(1).sqrt_() # [sqrt(1 - var), sqrt(1 + var)]
    for t in range(0, T):
        for o in range(0, O):
            if states[o][0] < appear_interval: # wait for interval frames 
                states[o][0] = states[o][0] + 1
            elif states[o][0] == appear_interval: # allow birth
                if unif[t][o] < birth_prob: # birth
                    # shape and appearance
                    color = color_id[t][o]
                    shape = shape_id[t][o]
                    scale = scales[t][o]
                    ratio = ratios[t][o]
                    h_, w_ = torch.round(h * scale * ratio), torch.round(w * scale / ratio)
                    #color_patch = torch.ByteTensor(h_, w_, D).fill_(1) * color_temp[color]
                    shape_patch = utils.imresize(shape_temp[shape], h_, w_)
                    # resiz data patch to predefined shape
                    #data_patch =  utils.resize_data_patch(data_patch, 56, 56)
                    # pose
                    direction = direction_id[t][o]
                    position = position_id[t][o]
                    x1, y1, x2, y2 = None, None, None, None
                    if direction == 0:
                        x1 = position[0]
                        y1 = torch.tensor(m)
                        x2 = position[1]
                        y2 = torch.tensor(H - 1 - m)
                    elif direction == 1:
                        x1 = position[0]
                        y1 = torch.tensor(H - 1 - m)
                        x2 = position[1]
                        y2 = torch.tensor(m)
                    elif direction == 2:
                        x1 = torch.tensor(m)
                        y1 = position[0]
                        x2 = torch.tensor(W - 1 - m)
                        y2 = position[1]
                    else:
                        x1 = torch.tensor(W - 1 - m)
                        y1 = position[0]
                        x2 = torch.tensor(m)
                        y2 = position[1]
                    theta = math.atan2(y2 - y1, x2 - x1)
                    vx = velocity * math.cos(theta)
                    vy = velocity * math.sin(theta)
                    # initial states
                    states[o] = [appear_interval + 1,shape_patch, shape_patch, x1, y1, vx, vy, 0, oid] #color_patch,
                    oid += 1
            else:  # exists
                #color_patch = states[o][1]
                shape_patch = states[o][2]
                x1, y1, vx, vy = states[o][3], states[o][4], states[o][5], states[o][6]
                step = states[o][7]
                x = torch.round(x1 + step * vx)
                y = torch.round(y1 + step * vy)
                if x < m-eps or x > W-1-m+eps or y < m-eps or y > H-1-m+eps: # the object disappears
                    states[o][0] = 0
                else:
                    #h_, w_ = color_patch.size(0), color_patch.size(1)
                    h_, w_ = shape_patch.size(0), shape_patch.size(1)
                    # center and start position for the big image
                    center_x = x + w
                    center_y = y + h
                    top = math.floor(center_y - (h_ - 1) / 2)
                    left = math.floor(center_x - (w_ - 1) / 2)
                    # put the color patch on image (not used in DHAE)
                    '''
                    color_img = buffer_big[0].zero_()
                    color_img.narrow(0, top, h_).narrow(1, left, w_).copy_(color_patch)
                    color_img = color_img.narrow(0, h, H).narrow(1, w, W) # H * W * D
                    '''

                    # put the shape patch on image (used in DHAE)------------------------------------
                    shape_img = buffer_big[1, :, :, 0].zero_()
                    shape_img.narrow(0, top, h_).narrow(1, left, w_).copy_(shape_patch)
                    shape_img = shape_img.narrow(0, h, H).narrow(1, w, W).unsqueeze(2) # H * W * 1
                    ### Save box and shape mask for DHAE evaluation

                    mask = shape_patch.numpy()
                    # get digit image of frame size
                    digit_256_img = shape_img.numpy()[:,:,0]
                    # Binarize image
                    digits_frame = np.array(digit_256_img > 0, dtype=np.uint8)
                    # obtaing bbox in image frame
                    box = np.array(cv2.boundingRect(digits_frame))
                    mask_28 = masks_resize(box, mask,dim_min = 28,dim_max = 28, vis=False)
                    #--------------------------------------------------------------------------------
                    # convert to float
                    #color_img_f = color_img.float()
                    shape_img_f = shape_img.float()
                    # synthesize a frame
                    org_img_f = org_seq[t].float() # H * W * D
                    syn_image = org_img_f + shape_img_f#/255 * (color_img_f - org_img_f)
                    org_seq[t].copy_(syn_image.round().byte())
                    # update the position
                    states[o][7] = states[o][7] + 1
                    # save for metric evaluation
                    if metric == 1:
                        file.write("%d,%d,%.3f,%.3f,%.3f,%.3f,1,-1,-1,-1\n" % 
                            (batch_id*T+t+1, states[o][8]+1, left-w+1, top-h+1, w_, h_))
                    # save box, shape mask for DHAE clustering and MOT metric evaluation
                    masks_f.append(mask_28)
                    boxs_f.append([batch_id*T+t+1,states[o][8]+1, box[0], box[1], box[2], box[3]])
    print('Total object in batch {} is {}'.format(batch_id,states[o][8]+1))
    boxs_final = np.array(boxs_f)
    masks_final = np.array(masks_f)
    return org_seq, states, masks_final,boxs_final

# combine all batches of box and shape
states_batch = []
data_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking-by-animation/'
dest = data_dir + 'DHAE_feature/SPRITE-MOT/{}Object_{}_org'.format(O,eval_type)
if not os.path.exists(dest):
    os.makedirs(dest)

img_path = dest + '/imgs'
if not os.path.exists(img_path):
    os.makedirs(img_path)
v=1
boxs_all = []
masks_all = []
for n in range(0, N):
    states_batch.append([])
    for o in range(0, O):
        states_batch[n].append([0]) # the states of the o-th object in the n-th sample
with Parallel(n_jobs=core_num, backend="threading") as parallel:
    for split in ['test']:
        S = batch_nums[split]
        ind = 0
        for s in range(0, S): # for each batch of sequences
            out_batch = parallel(delayed(process_batch)(states_batch[n], s) for n in range(0, N)) # N * 2 * T * H * W * D
            out_batch = list(zip(*out_batch)) # 2 * N * T * H * W * D

            print('Boxes: ', out_batch[3][0].shape)
            print('Masks: ', out_batch[2][0].shape)
            #np.save(dest + '/boxs_4digits'+'{:02d}'.format(s), out_batch[3][0], allow_pickle=True, fix_imports=True)
            #np.save(dest + '/masks_4digits'+'{:02d}'.format(s),out_batch[2][0], allow_pickle=True, fix_imports=True)
            masks_all.append(out_batch[2][0])
            boxs_all.append(out_batch[3][0])


            org_seq_batch = torch.stack(out_batch[0], dim=0) # N * T * H * W * D
            states_batch = out_batch[1] # N * []
            if v == 1:
                for t in range(0, T):
                    img_128 = org_seq_batch[0, t].numpy()[:,:,0]
                    dpi = 200
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(img_128.shape[1] / dpi, img_128.shape[0] / dpi)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.axis('off')
                    fig.add_axes(ax)
                    ax.imshow(img_128)
                    #fig.savefig(dest + '/imgs/{:04d}.png'.format(ind+1),cmap = cm.gray,dpi=dpi)
                    im_pil = Image.fromarray(img_128.astype(np.uint8))
                    im_pil.convert('RGBA')
                    im_pil.save(img_path+'/{:04d}.png'.format(ind+1))
                    ind+=1
                    #for t in range(0, T):
                        #utils.imshow(org_seq_batch[0, t], 400, 400, 'img', 50)
            else:
                org_seq_batch = org_seq_batch.permute(0, 1, 4, 2, 3) # N * T * D * H * W
                #filename = split + '_' + str(s) + '.pt'
                #torch.save(org_seq_batch, path.join(output_input_dir, filename))
            print(split + ': ' + str(s+1) + ' / ' + str(S))

        # save all bathces of location and shape features
        boxs = np.array(boxs_all)
        boxs_list = [b for b in boxs if len(b) > 0]
        boxs_train = np.concatenate(boxs_list)

        masks = np.array(masks_all)
        masks_list = [b for b in masks if len(b) > 0]
        mask_train = np.concatenate(masks_list)

        print('Boxes: ', boxs_train.shape)
        print('Masks: ', mask_train.shape)
        np.save(dest + '/boxs_{}digits'.format(O), boxs_train, allow_pickle=True, fix_imports=True)
        np.save(dest + '/masks_{}digits'.format(O), mask_train, allow_pickle=True, fix_imports=True)

if metric == 1:
    file.close()

# save the data configuration
data_config = {
    'task': task,
    'train_batch_num': batch_nums['train'], 
    'test_batch_num': batch_nums['test'],
    'N': N,
    'T': T,
    'D': D,
    'H': H,
    'W': W,
    'h': h,
    'w': w,
    'zeta_s': scale_var,
    'zeta_r': [1, ratio_var]
}
utils.save_json(data_config, path.join(output_dir, 'data_config.json'))