from __future__ import division
from keras.models import Model
from keras.utils import plot_model
from keras.models import model_from_json
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image, ImageColor

import glob
import numpy as np
from sklearn.cluster import k_means
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import os
import cv2
import sys
import glob
import math
import pdb
import time
from collections import Counter
from statistics import mode
import random
from numpy.random import seed

pathname = os.path.dirname(sys.argv[0])
sys.path.insert(0, pathname)

from src.MTL_CAE_Reshape import DHAE_Model_reshape, test_net
from src.DHAE_appearance import DHAE_pRGB
from src.data_association_clustering import cluster_association
from src.con_kmeans import constraints_kmeans
from pycocotools import mask as maskUtils
from utils.mask_resize import masks_resize
from utils.t_SNE_plot import *
from utils.merge_tracklet_offline import *
from utils.tct_utils import *

seed(1)
import tensorflow as tf
tf.compat.v1.set_random_seed(2)


 #python3 mots_eval/eval.py KITTI/tracking_results KITTI/gt_folder KITTI/val.seqmap.txt
def init_model(codeBasePath=None, model_type='poseApp', MTL=None):
    img_y, img_x = 128, 128
    box_dim = 4
    filters = [16, 32, 32, 64]
    embed_dim = 128
    if model_type == 'poseApp':
        channel_num = 3
        if MTL:
            checkpoint_path = codeBasePath + 'model/KITTI_MOT17/rgb/MTL/cp-0493.ckpt'
        if not MTL:
            checkpoint_path = codeBasePath + 'model/KITTI_MOT17/rgb/arbit_weight/cp-0284.ckpt'

        final_model, bottleneck_model, mask_encoder_model, \
        box_encoder_model = DHAE_pRGB(img_y, img_x, box_dim, filters, embed_dim, checkpoint_path, n_G=1,
                                      MTL=MTL, optimizer='adadelta', channel=channel_num).test_net()
        # load saved weights to the DHAE models
        print('load checkpoint from {}'.format(checkpoint_path))
        final_model.load_weights(checkpoint_path)

        names = [weight.name for layer in final_model.layers for weight in layer.weights]
        weights = final_model.get_weights()
        i = 0
        for name, weight in zip(names, weights):
            print(i, name, weight.shape)
            i += 1
        bottleneck_model.set_weights(weights=final_model.get_weights()[:28])
        # mask_encoder_model.set_weights(weights=final_model.get_weights()[:24])
        # box_encoder_model.set_weights(weights=final_model.get_weights()[24:26])
        # plot_model(final_model, to_file=codeBasePath + 'final_model/checkpoints_MOTS_128_kitti_appearance64/model.png')

    if model_type == 'poseShape':
        channel_num = 1
        if MTL:
            checkpoint_path = codeBasePath + 'final_model/model_exp/shape-model/MTL/kitti_128/bmvc/var_weight/cp-0164.ckpt'
        if not MTL:
            checkpoint_path = codeBasePath + 'final_model/model_exp/shape-model/arbit_weight/bmvc/no_box_factor/cp-0***.ckpt'

        final_model, bottleneck_model, mask_encoder_model, \
        box_encoder_model = DHAE_pRGB(img_y, img_x, box_dim, filters, embed_dim, checkpoint_path, n_G=1,
                                      MTL=MTL, optimizer='adadelta', channel=channel_num).test_net()
        # load saved weights to the DHAE models
        print('load checkpoint from {}'.format(checkpoint_path))
        final_model.load_weights(checkpoint_path)

        names = [weight.name for layer in final_model.layers for weight in layer.weights]
        weights = final_model.get_weights()
        i = 0
        for name, weight in zip(names, weights):
            print(i, name, weight.shape)
            i += 1

        bottleneck_model.set_weights(weights=final_model.get_weights()[:28])
    return bottleneck_model

def load_dataset(codeBasePath, seq, classID, det_thr, data_source = 'public'):

    if data_source == 'private':
        if classID == 1:
            motsID = 2  # ped
            area_th = 100
        else:
            motsID = 1  # car
            area_th = 150
    else:
        if classID == 2:
            motsID = 2
            area_th = 100
            # from GT: min(w*h)=
        else:
            motsID = 1  # car
            area_th = 150
            # from GT: min(w*h)= 6*3=18

    if dataset == 'KITTI':
        data_path = codeBasePath + 'data/KITTI/128_rgb'
        boxs = np.load(data_path + '/box0_128_' + seq.split('/')[-1] + '.npy', encoding='bytes')
        masks = np.load(data_path + '/mask0_128_' + seq.split('/')[-1] + '.npy', encoding='bytes')
        masks_28 = np.load(data_path + '/mask0_' + seq.split('/')[-1] + '.npy', encoding='bytes')  # mask rles
        patch_rgb = np.load(data_path + '/patch0_128_' + seq.split('/')[-1] + '.npy', encoding='bytes')

    if dataset == 'MOT17':
        data_path = storage + 'data/MOT17/128_rgb'
        boxs = np.load(data_path + '/box0_128_' + seq.split('/')[-1] + '.npy', encoding='bytes')
        masks = np.load(data_path + '/mask0_128_' + seq.split('/')[-1] + '.npy', encoding='bytes')
        masks_28 = np.load(data_path + '/mask0_' + seq.split('/')[-1] + '.npy', encoding='bytes')  # mask rles
        patch_rgb = np.load(data_path + '/patch0_128_' + seq.split('/')[-1] + '.npy', encoding='bytes')

    ins_128, img_y, img_x = masks.shape
    # To visualize use predicted mask size
    mask_x = mask_y = 28
    ins_4, _ = boxs.shape
    assert masks_28.shape[0] == ins_128 == ins_4, 'number of instances for boxs and masks are unequal'

    # all detections are uninitialized
    if boxs[0, 8] != 0:
        boxs[:, 8] = 0
    # select detections for a specific detection threshold
    patch_rgb = patch_rgb[boxs[:, 6] >= det_thr]
    masks_28 = masks_28[boxs[:, 6] >= det_thr]
    masks = masks[np.where(boxs[:, 6] >= det_thr), :, :][0]
    boxs = boxs[np.where(boxs[:, 6] >= det_thr), :][0]
    # select the required classes based on benchmark
    if tct_multicalss:
        pax_patch_rgb = patch_rgb
        pax_boxs = boxs
        pax_mask = masks
        mask_rles = masks_28
    else:
        pax_patch_rgb = patch_rgb[boxs[:, 7] == classID]
        pax_boxs = boxs[np.where(boxs[:, 7] == classID), :][0]
        pax_mask = masks[np.where(boxs[:, 7] == classID), :, :][0]
        mask_rles = masks_28[boxs[:, 7] == classID]

    if len(pax_boxs) == 0:
        pax_boxs = pax_mask = mask_rles = pax_patch_rgb = None
        return pax_boxs, pax_mask, mask_rles, pax_patch_rgb

    # ignore w<20,h<20
    # TODO: ignore small ped and car
    print('min w {} and min h {} and min area w*h {}'.format(min(pax_boxs[:, 4]), min(pax_boxs[:, 5]),
                                                             min(pax_boxs[:, 4] * pax_boxs[:, 5])))
    areas = pax_boxs[:, 4] * pax_boxs[:, 5]
    # import matplotlib.pyplot as plt1
    # plt1.hist(areas, bins=50, range=(min(areas), 500))
    # plt1.savefig(imgPath + '{}.png'.format(seq.split('/')[-1]),dpi=300)
    # plt1.close()

    pax_boxsw = pax_boxs[pax_boxs[:, 4] * pax_boxs[:, 5] >= area_th]
    pax_maskw = pax_mask[pax_boxs[:, 4] * pax_boxs[:, 5] >= area_th]
    mask_rlesw = mask_rles[pax_boxs[:, 4] * pax_boxs[:, 5] >= area_th]
    pax_patch_rgbw = pax_patch_rgb[pax_boxs[:, 4] * pax_boxs[:, 5] >= area_th]

    pax_boxs = pax_boxsw
    pax_mask = pax_maskw
    mask_rles = mask_rlesw
    pax_patch_rgb = pax_patch_rgbw
    return pax_boxs, pax_mask, mask_rles, pax_patch_rgb, motsID, img_y, img_x

def init_dirs(seq, out_path, imgPath):
    # seq vis path
    out_path_seq = out_path + 'vis/' + seq.split('/')[-1] + '/'
    delete_all(out_path, fmt='png')
    if not os.path.exists(out_path_seq):
        os.makedirs(out_path_seq)
    # directory to visualize alignment
    imgPath_align = imgPath + seq.split('/')[-1] + '/'
    if not os.path.exists(imgPath_align):
        os.makedirs(imgPath_align)
    delete_all(imgPath_align, fmt='png')
    # directory to visualize window tSNE
    tsne_path = imgPath + seq.split('/')[-1] + '/tsne/'
    if not os.path.exists(tsne_path):
        os.makedirs(tsne_path)
    delete_all(tsne_path, fmt='png')

    out_mask_path = out_path + 'vis/maskP/' + seq.split('/')[-1] + '/'
    if not os.path.exists(out_mask_path):
        os.makedirs(out_mask_path)

    return out_path_seq, out_mask_path, imgPath_align, tsne_path

def show_align(t_window_aligned, imgPath_align, names, color):
    posXY = [50, 50]
    img = plot_window(t_window_aligned, names, color, posXY)
    im = Image.fromarray(img)
    im.save(imgPath_align + names[-10:])

## Main Function ####
if __name__ == '__main__':
    np.random.seed(1234)
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61'
    NAS = '/media/siddique/RemoteServer/LabFiles'
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
    model_type = 'poseApp'#'poseShape'#'poseAppUnet', 'poseApp'-best
    dataset = 'KITTI'
    codeBasePath = storage + '/Mask_Instance_Clustering/USC_MOTS_bitbucket/usc_mots/'
    #delete the images in specific paths
    imgPath = storage + '/mots_tools/'+dataset+'/tracking_results/window_exp/'
    mot_evaluation = True
    model_compnt = 'loc+app'

    # parametrers for TCT\\s
    score_th = 0.1 # m=not used
    min_cluster_size = 2#2
    tct_multicalss = 0
    graph_metric = 'mask_iou'
    iou_thr = 0.0
    #parameter for DHAE-SC
    n_G = 1

    time_lag = 3  # 8 #3
    if dataset=='MOT17':
        alpha_t=2 #2:MOT17
        min_track_size = 5  # 5:MOT17
        category_id = 2  #[mot17: 2:ped]
    if dataset=='KITTI':
        alpha_t = 1  # 1:kitti
        min_track_size = 3  #3:kitti
        category_id = 1  # [kitti: 1:car, 2:ped]

    if category_id==1:
        det_thr = 0.6
    if category_id==2:
        det_thr = 0.7

    MTL = 1
    Constraints = 1
    DHAE_clustering = 1

    vis=0
    print_stat = 1
    keep_track_id_start = []

    # use trained DHAE to get embedded feature for clustering
    bottleneck_model = init_model(codeBasePath=codeBasePath,
                                  model_type=model_type,
                                  MTL=MTL)
    if dataset in ['MOT17']:
        img_align = 0
        # read evaluation sequences
        img_format = '.jpg'
        folders = glob.glob(NAS + '/MOTS/MOT17/imgs/' + '*')
        folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        seqs = folders
        motsID = 2

    if dataset=='KITTI':
        img_align = 0
        img_format = '.png'
        folders = glob.glob(NAS + '/MOTS/KITTI/training/image_02/' + '*')
        folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        # Get validation data sequences
        val_set = [folders[-19], folders[-15], folders[-14], folders[-13], folders[-11], folders[-8], folders[-7],
                   folders[-5], folders[-3]]
        seqs = val_set #folders[2::]

    frame_count = 0
    time_count = 0
    # output paths
    out_path = codeBasePath.split('Mask_Instance_Clustering')[0] \
               + '/mots_tools/' + dataset + '/tracking_results/'
    delete_all(out_path, fmt='txt')

    for seq in seqs:
        keep_track_id_start = []
        # read features: PANet or Mask-RCNN, Resnet50 or 101: current: Mask-RCNN X101
        pax_boxs, pax_mask, mask_rles, \
        pax_patch_rgb, motsID, \
        img_y, img_x = load_dataset(codeBasePath, seq, category_id, det_thr, data_source = 'public')
        if pax_boxs is None:
            continue
        out_path_seq, out_mask_path, \
        imgPath_align, tsne_path = init_dirs(seq, out_path, imgPath)

        # read benchmarq images
        path = seq+'/*'+img_format
        files = glob.glob(path)
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        # prepare image path set
        img_set = {}
        img_indx = 1
        for name in files:
            frame_count+=1
            img_set[img_indx] = name
            img_indx+=1

        mots = open(out_path + seq.split('/')[-1] + '.txt','w')
        # initialize the global variables
        det_cluster_id = None
        dets_align_centers = None
        tracklet_box = None
        tracklet_mask = None
        tracklet_mask_rgb = None
        embed = None
        # tracklets array
        trackers = {}
        fr = 1
        combined_mask_per_frame = {}
        ID_ind = 1
        # sequential data window
        fr_start = int(pax_boxs[:, 0][0])
        fr_end = int(pax_boxs[:, 0][-1]) #pax_boxs[:, 0][::]

        vis_window=False
        #get copy of boxes to keep separate the aligned and unaligned boxes
        pax_boxs_align = copy.deepcopy(pax_boxs)
        for names in files:
            print(names)
            im = cv2.imread(names)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_h, im_w, _  = im.shape
            im_shape = im.shape[:2]

            # check that frame has detection:
            # TODO: when current frame has no detection
            #  what about the prvious frame detections??
            # When no detection at current frame work with dummy predictions??
            if fr in pax_boxs[:, 0]:
                #apply camera stabilization on current frame and previous frame
                # when fr>1 and detection is available at current frame
                # align/warped current frame bbox to previous frame bbox
                if img_align:
                    t_bbox = pax_boxs_align[pax_boxs_align[:, 0] == fr]
                    if fr>1:
                        t_bbox = xywh2x1y1x2y2(t_bbox)
                        t_bbox = align_window(fr, t_bbox, img_set)
                        t_bbox = x1y12xywh(t_bbox)
                    pax_boxs_align[pax_boxs_align[:, 0] == fr] = t_bbox

                # Select the temporal window for clustering : use t-t_lag
                if(fr>=fr_start+time_lag-1):
                    start_time = time.time()
                    temp_window_pbox, \
                    temp_window_pmask, \
                    temp_window_pmask_28, \
                    temp_window_patch, \
                    k_value,\
                    t_window_aligned,\
                    keep_key_frame= get_temporal_window(fr,
                                                  time_lag,
                                                  pax_boxs,
                                                  pax_boxs_align,
                                                  pax_mask,
                                                  mask_rles,
                                                  pax_patch_rgb,
                                                  det_cluster_id
                                                  )
                    #visualize the temporal window detections at current frame
                    if vis_window and img_align:
                        show_align(t_window_aligned, imgPath_align, names, color)
                    print('number of instances in window:', k_value)
                    # prepare mask and box features for DHAE
                    _, pax_box_norm = prepare_box_data(t_window_aligned,im_h,im_w)
                    temp_pax_boxs = temp_window_pbox
                    # bottleneck feature and reconstruction for temporal window t-10
                    if model_type=='poseApp':
                        #input_img, input_box, decoded_imgs, decoded_box = final_model.predict([patch_x_test, pax_box_norm])
                        patch_x_test = np.reshape(temp_window_patch,
                                                  [temp_window_patch.shape[0], img_y, img_x, 3]) / 255.
                        if model_compnt == 'loc+app':
                            latent_feature = bottleneck_model.predict([patch_x_test, pax_box_norm])

                    if model_type == 'poseShape':
                        x_test = temp_window_pmask
                        mask_x_test = np.reshape(x_test, [x_test.shape[0], img_y, img_x, 1])
                        if model_compnt == 'loc+shape':
                            latent_feature = bottleneck_model.predict([patch_x_test, pax_box_norm])


                    X_frames = temp_pax_boxs[:,0].astype(np.float32)
                    X_frames = X_frames.copy(order='C')
                    #latent_feature = pax_box_norm
                    if fr%100000==0 and len(latent_feature)<=10:
                        plot_embed_affinity(patch_x_test, latent_feature, decoded_imgs, img_y, img_x, names, out_path, seq.split('/')[-1], color)
                        plot_latent_feature(patch_x_test, latent_feature,decoded_imgs,img_y,img_x,names,out_path,seq.split('/')[-1])

                    # Compute optimum k-values in k-means for target clusters at current frame
                    # new_det init only when max() method use to compute k
                    max_instances_at_t = int(np.max(k_value))
                    init_k = 'new_det'
                    #key_frame = max(np.unique(X_frames)[::-1][k_value==max_instances_at_t])
                    key_frame = keep_key_frame[max_instances_at_t]
                    #max_instances_cumsum = cumsum_diff(k_value) #init method: kmpp
                    print('K value from cluster analysis:',  max_instances_at_t)
                    if Constraints:
                        labels, cluster_centers = constraints_kmeans(fr,
                                                                     X_frames,
                                                                     temp_pax_boxs,
                                                                     latent_feature,
                                                                     temp_window_pmask_28,
                                                                     max_instances_at_t,
                                                                     names,
                                                                     color,
                                                                     time_lag,
                                                                     im_h,im_w,
                                                                     fr_start,
                                                                     iou_thr=iou_thr,
                                                                     k_frame = key_frame,
                                                                     alpha_t=alpha_t,
                                                                     graph_metric=graph_metric,
                                                                     initialization=init_k,
                                                                     verbose=True
                                                                     )
                    else:
                        kmeans = KMeans(n_clusters=max_instances_at_t)
                        kmeans = kmeans.fit(latent_feature)
                        labels = kmeans.predict(latent_feature)
                        cluster_centers = kmeans.cluster_centers_
                    #plot_tSNE(temp_pax_boxs,latent_feature, labels, tsne_path, fr, color)
                    refined_det,\
                    final_mask,\
                    det_cluster_id, \
                    ID_ind,\
                    trackers =  cluster_association(fr,
                                                    X_frames,
                                                    latent_feature,
                                                    temp_pax_boxs,
                                                    temp_window_pmask_28,
                                                    labels,
                                                    ID_ind,
                                                    time_lag,
                                                    score_th,
                                                    min_cluster_size,
                                                    k_value,
                                                    names,
                                                    color,
                                                    img_y, img_x,
                                                    im_shape,
                                                    trackers=trackers,
                                                    vis_pred=vis
                                                    )



                    print('Total tracked ID:',ID_ind-1)
                    print("Execution time {} sec".format(time.time() - start_time))
                    time_count+=time.time() - start_time
                    # plot clustered result using t-SNE
                    # plot_tSNE(latent_feature,labels, tsne_path)
                    #plot_mask_cluster(fr_x_test, labels,max_instances_at_theta, img_x, img_y, names)
                    # show clustered result in lower dimension
                    # tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
                else:
                    print('TODO: reid problem - tracklet is missing for more than t_lag, new ID for re-entry')
            fr += 1
            plt.close()
        #pdb.set_trace()
        # save trackers for mots evaluation
        save_mots(trackers, mots, ID_ind-1, fr_start, fr_end, color, im_h, im_w,
                  out_mask_path, out_path_seq, seq, dataset,
                  img_format, class_label=motsID, scta=0, min_track_size=min_track_size, vis=vis)
        mots.close()

    if time_count>0:
        print('Average Speed: {:.2f} Hz'.format(frame_count/time_count))
        print('total frame: {}, total time {:.2f} sec'.format(frame_count, time_count))

