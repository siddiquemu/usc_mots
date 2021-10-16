from __future__ import division
import numpy as np
import glob
from scipy.spatial import distance
import random
import cv2

def extract_patch(name,box_ref,im_h,im_w):
    im = cv2.imread(name)
    im = cv2.cvtCOlor(im,cv2.COLOR_BGR2RGB)
    y0 = int(max(box_ref[1],0))
    x0 = int(max(box_ref[1],0))
    y1 = int(min(box_ref[1]+box_ref[3],im_h))
    x1 = int(min(box_ref[0]+box_ref[2],im_w))
    patch = im[y0:y1,x0:x1]
    return patch

def std_normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    meanX = np.mean(x)
    stdX = np.std(x)
    x = (x-meanX)/stdX
    return x
def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def prepare_data_mask(mask,channel):
    #offset = int(mask.shape[0])
    print('Training Data Preparation Start...')
    #x_t = np.zeros((mask.shape[0],30,30),dtype='float')
    #x_t[0:x_t.shape[0],1:29,1:29] = mask
    #x_t = std_normalize(x_t)
    x_t = mask

    '''
    print('Reading Mask Instances...')
    x_t=np.load('/media/siddique/Data/CLASP2018/CAE_train_mask/10A/mask_exp10acam9_270.npy',encoding='bytes')
    #x_t = np.load('/media/siddique/Data/CLASP2018/CAE_train_mask/10A/mask_exp10acam11_270.npy', encoding='bytes')
    '''

    #cam9: 13000, 13790, cam11: 10000, 11209
    #x_train = x_t[0:12000,:,:].astype('float32')# / 255. #normalize pixel values between 0 and 1 [coarse mask has already in 0-1 scale]
    #x_test = x_t[12000:13790,:,:].astype('float32')# / 255.
    x_train = np.reshape(x_t, (mask.shape[0],mask.shape[1] ,mask.shape[2] ,channel))  # adapt this if using `channels_first` image data format
    #x_test = np.reshape(x_test, (len(x_test), img_y,img_x, 1))  # adapt this if using `channels_first` image data format
    print('Training Data Preparation Done...')
    return x_train

def prepare_data_box(x_t,im_w,im_h):
    print('Training Data Preparation Start...')
    #offset = int(x_t.shape[0])

    x_0 = x_t[:, 2]/ im_w #/ 1920#np.max(x_t[:, 2])
    y_0 = x_t[:, 3]/ im_h #/ 1080#np.max(x_t[:, 3])
    w = x_t[:, 4]/ np.max(x_t[:, 4]) #/ 1920#np.max(x_t[:, 4])
    h = x_t[:, 5]/ np.max(x_t[:, 5])# / 1080#np.max(x_t[:, 5])
    Cx =(x_t[:, 2] + x_t[:, 4] / 2)/ im_w #/ 1920# np.max((x_t[:, 2] + x_t[:, 4] / .2))
    Cy = (x_t[:, 3] + x_t[:, 5] / 2)/ im_h #/ 1080# np.max((x_t[:, 2] + x_t[:, 4] / .2))
    area = (x_t[:, 4] * x_t[:, 5])/ (im_w*im_h)#np.max((x_t[:, 4] * x_t[:, 5]))
    diag = np.sqrt(x_t[:, 4]**2 + x_t[:, 5]**2)/np.sqrt(im_w**2+im_h**2)
    score = x_t[:, 6]
    # prepare dim = 8:[Cx,Cy,x,y,w,h,wh,class]
    x_t = np.array([Cx, Cy, w, h])
    #x_t = np.array([Cx, Cy, w, h,x_0,y_0,area,diag])#, for mot x_0,y_0 instead of w,h
    x_t = np.transpose(x_t)
    #x_t = normalize(x_t)

    print('Training Data Preparation Done...')
    return x_t

def prepare_data_box_kitti(x_t,im_w,im_h):
    print('Training Data Preparation Start...')
    #offset = int(x_t.shape[0])
    im_w, im_h = float(im_w), float(im_h)
    x_0 = x_t[:, 2]/ im_w #/ 1920#np.max(x_t[:, 2])
    y_0 = x_t[:, 3]/ im_h #/ 1080#np.max(x_t[:, 3])
    w = x_t[:, 4]/ np.max(x_t[:, 4])
    h = x_t[:, 5]/ np.max(x_t[:, 5])
    Cx =(x_t[:, 2] + x_t[:, 4] / 2.)/ im_w #/ 1920# np.max((x_t[:, 2] + x_t[:, 4] / .2))
    Cy = (x_t[:, 3] + x_t[:, 5] / 2.)/ im_h #/ 1080# np.max((x_t[:, 2] + x_t[:, 4] / .2))
    area = (x_t[:, 4] * x_t[:, 5])/ (im_w*im_h)#np.max((x_t[:, 4] * x_t[:, 5]))
    diag = np.sqrt(x_t[:, 4]**2 + x_t[:, 5]**2)/np.sqrt(im_w**2+im_h**2)
    score = x_t[:, 6]
    # prepare dim = 8:[Cx,Cy,x,y,w,h,wh,class]
    x_t = np.array([Cx, Cy, w, h])
    #x_t = np.array([Cx, Cy, w, h,x_0,y_0,area,diag])#, for mot x_0,y_0 instead of w,h
    x_t = np.transpose(x_t)
    #x_t = normalize(x_t)

    print('Training Data Preparation Done...')
    return x_t

def prepare_data_patch(patch):
    print('Patch Training Data Preparation Start...')
    offset = int(patch.shape[0])
    x_t = patch[0:offset, :, :,:]
    #x_train = np.reshape(x_t, (len(x_t), patch.shape[1], patch.shape[2], -))
    print('Patch Training Data Preparation Done...')
    return x_t

#pathp = '/media/siddique/Data/CLASP2018/train_data_all/mot_reshape/128/patch0*'
dataset = 'kitti_rgb'
if dataset=='clasp2':
    pathm = '/media/RemoteServer/LabFiles/CLASP2/2019_04_16/exp2/train_data_all/mask0*'
    pathb = '/media/RemoteServer/LabFiles/CLASP2/2019_04_16/exp2/train_data_all/box0*'
if dataset=='kitti':
    # include MOTS-KITTI: Training set, MOT-17: 4, 10, 13 (detector results since we do not need to use labeled data)
    pathm = '/media/RemoteServer/LabFiles/MOTS/train_data_all/128/mask0*'
    pathb = '/media/RemoteServer/LabFiles/MOTS/train_data_all/128/box0*'
    pathp = '/media/siddique/Data/CLASP2018/train_data_all/mot_reshape/128/patch0*'
if dataset=='kitti_rgb':
    pathb = '/media/siddique/RemoteServer/LabFiles/MOTS/128_rgb/box0*'
    pathp = '/media/siddique/RemoteServer/LabFiles/MOTS/128_rgb/patch0*'
    channel = 3
#filesp = glob.glob(pathp)
filesm = glob.glob(pathp)
filesb = glob.glob(pathb)
#filesp.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
filesm.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
filesb.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
patch_all = []
masks_all = []
boxs_all = []
mot_seq=[1,4,6,8]
for j in range(len(filesb)):
    print(filesb[j])
    print(filesm[j])
    #print(filesp[j])
    #x_train_box = prepare_data_box()
    x_train_b = np.load(filesb[j],encoding='bytes')
    x_train_m = np.load(filesm[j],encoding='bytes')
    #x_train_p = np.load(filesp[j],encoding='bytes')
    random_sample = np.random.randint(len(x_train_b)//2, size=int(len(x_train_b)//2))
    x_train_mask = prepare_data_mask(x_train_m[random_sample],channel)
    if j in mot_seq:
        img = cv2.imread('/media/siddique/RemoteServer/LabFiles/MOTS/MOT17/imgs/'+filesm[j][-11:-7]+'/000001.jpg')
        im_h,im_w,_ = img.shape
    else:
        img = cv2.imread(
            '/media/siddique/RemoteServer/LabFiles/MOTS/KITTI/training/image_02/' + filesm[j][-8:-4] + '/000000.png')
        im_h, im_w, _ = img.shape

    x_train_box = prepare_data_box_kitti(x_train_b[random_sample],im_w,im_h)
    #x_train_patch = prepare_data_patch(x_train_p[random_sample])
    #patch_all.append(x_train_patch)
    masks_all.append(x_train_mask)
    boxs_all.append(x_train_box)

#patch_all = np.array(patch_all)
#patch_list = [b for b in patch_all]
#patch_all = np.concatenate(patch_list)

masks_all = np.array(masks_all)
mask_list = [b for b in masks_all]
masks_all = np.concatenate(mask_list)

boxs_all = np.array(boxs_all)
boxs_list = [b for b in boxs_all]
boxs_all = np.concatenate(boxs_list)
if dataset == 'clasp2':
    np.save('/media/RemoteServer/LabFiles/CLASP2/2019_04_16/exp2/train_data_all/train_mask_clasp2_reshape128',
            masks_all, allow_pickle=True, fix_imports=True)
    np.save('/media/RemoteServer/LabFiles/CLASP2/2019_04_16/exp2/train_data_all/train_box_clasp2_reshape128',
            boxs_all, allow_pickle=True, fix_imports=True)
else:
    np.save('/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Mask_Instance_Clustering/train_data_all/rgb256/train_mask_mot_all0_reshape256_kitti_mot17',masks_all,allow_pickle=True,fix_imports=True)
    np.save('/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Mask_Instance_Clustering/train_data_all/rgb256/train_box_mot_all0_reshape256_kitti_mot17',boxs_all,allow_pickle=True,fix_imports=True)
#np.save('/media/siddique/Data/CLASP2018/train_data_all/train_patch_mot_all0_reshape',patch_all,allow_pickle=True,fix_imports=True)
assert masks_all.shape[0]==boxs_all.shape[0], "Number of examples for location and shape features are not equal"
print(masks_all.shape)
print(boxs_all.shape)
#print(patch_all.shape)
print('box feature: (max, min) ', boxs_all.max(),boxs_all.min())