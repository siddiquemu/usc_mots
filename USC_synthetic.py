from __future__ import division
import cv2
import glob
import os
import math
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import argparse
from keras.models import Model
from keras.models import model_from_json
from keras.models import load_model
from keras import backend as K

from sklearn.cluster import k_means
from sklearn.cluster import SpectralClustering
from pycocotools import mask as maskUtils
from munkres import Munkres
from sklearn import metrics
from sklearn.cluster import KMeans
from collections import Counter
from statistics import mode

#sys.path.insert(0, '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/Mask_Instance_Clustering/COP-Kmeans')
codebase = os.path.dirname(sys.argv[0])
sys.path.insert(0, codebase)

from matplotlib.patches import Polygon
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.manifold import TSNE

from utils.mask_resize import masks_resize
from utils.t_SNE_plot import *
from utils.utils_synthetic import *
from src.data_association_clustering import cluster_association_synthetic
from src.con_kmeans import con_kmeans_synthetic, con_kmeans_clustering
from src.DHAE_synthetic import DHAE

from numpy.random import seed
seed(1)
def print_layers_info(final_model):
    names = [weight.name for layer in final_model.layers for weight in layer.weights]
    weights = final_model.get_weights()
    i = 0
    for name, weight in zip(names, weights):
        print(i, name, weight.shape)
        i += 1

def get_temporal_window(fr, time_lag, box_seq, mask_seq, det_cluster_id=None):
    temp_window_box = []
    temp_window_mask = []
    k_value = []
    keep_key_frame = {}
    for i in np.linspace(fr, fr - time_lag + 1, num=time_lag):

        print('fr in window', i)
        # TODO: check that at least one detection at t
        temp_windowb = box_seq[np.where(box_seq[:, 0] == i), :][0]
        temp_windowm = mask_seq[np.where(box_seq[:, 0] == i), :, :][0]
        k_value.append(len(temp_windowb[:, 0]))  # max value of instance at t
        if det_cluster_id is not None and i < fr:  # (fr=6, i=2,3,4,5 has already cluster id initialized detections)
            temp_windowb = det_cluster_id[np.where(det_cluster_id[:, 0] == i), :][0]
            assert len(temp_windowb) == len(temp_windowm)

        temp_window_box.append(temp_windowb)
        temp_window_mask.append(temp_windowm)
        keep_key_frame[len(temp_windowb)] = i

    temp_window_box, \
    temp_window_mask = expand_from_temporal_list(temp_window_box,
                                                  temp_window_mask)
    return temp_window_box, temp_window_mask, det_cluster_id, keep_key_frame, np.array(k_value)

def init_model(codeBasePath=None, dataset='mnist', MTL=None):
    if dataset == 'MNIST':
        img_y, img_x = 28, 28
        box_dim = 4
        filters = [16, 24, 48]
        embed_dim = 64
        channel_num = 1
        if MTL:
            checkpoint_path = os.path.join(codeBasePath, 'model/MNIST/MTL/cp-0221.ckpt')
        if not MTL:
            checkpoint_path = os.path.join(codeBasePath, 'model/MNIST/arbit_weight/cp-*.ckpt')

        final_model, bottleneck_model, mask_encoder_model, \
        box_encoder_model = DHAE(img_y, img_x, box_dim, filters, embed_dim, checkpoint_path, n_G=1,
                                      MTL=MTL, optimizer='adadelta', channel=channel_num).test_net()

        # load saved weights
        print('load checkpoint from {}'.format(checkpoint_path))
        final_model.load_weights(checkpoint_path)

        print_layers_info(final_model)

        bottleneck_model.set_weights(weights=final_model.get_weights()[:12])  # 18
        #box_encoder_model.set_weights(weights=final_model.get_weights()[8:10]) #14:16
        #mask_encoder_model.set_weights(weights=final_model.get_weights()[:8])  # 14

        # plot_model(final_model, to_file=codeBasePath + 'final_model/checkpoints_MOTS_128_kitti_appearance64/model.png')

    if dataset == 'SPRITE':
        img_y, img_x = 28, 28
        box_dim = 4
        filters = [16, 32, 32]
        embed_dim = 64
        channel_num = 1
        if MTL:
            checkpoint_path = os.path.join(codeBasePath, 'model/Sprites/MTL/cp-0866.ckpt')
        if not MTL:
            checkpoint_path = os.path.join(codeBasePath, 'model/Sprites/arbit_weight/cp-*.ckpt')

        final_model, bottleneck_model, mask_encoder_model, \
        box_encoder_model = DHAE(img_y, img_x, box_dim, filters, embed_dim, checkpoint_path, n_G=1,
                                 MTL=MTL, optimizer='adadelta', channel=channel_num).test_net()

        # load saved weights
        print('load checkpoint from {}'.format(checkpoint_path))
        final_model.load_weights(checkpoint_path)

        print_layers_info(final_model)

        bottleneck_model.set_weights(weights=final_model.get_weights()[:12])  # 18
        # box_encoder_model.set_weights(weights=final_model.get_weights()[8:10]) #14:16
        # mask_encoder_model.set_weights(weights=final_model.get_weights()[:8])  # 14
    return bottleneck_model, mask_encoder_model, box_encoder_model

def load_dataset(data_path=None, dataset='MNIST', USC_tracker=True):
    # load all the sequence frames as a single file
    x_m = np.load(os.path.join(data_path, 'masks_3digits.npy'), allow_pickle=True)
    x_b = np.load(os.path.join(data_path, 'boxs_3digits.npy'), allow_pickle=True)
    if USC_tracker:
        x_b = np.hstack((x_b, np.zeros((x_b.shape[0], 1), dtype=x_b.dtype)))
    return x_b, x_m

def init_dirs(codeBase, dataset='MNIST'):
    # set data path'
    data_path = os.path.join(codeBase, 'data/{}_MOT/test'.format(dataset))
    # output path
    out_path = os.path.join(codeBase, 'results/{}/tracking_results'.format(dataset))
    out_path_seq = os.path.join(out_path, 'vis')
    if not os.path.exists(out_path_seq):
        os.makedirs(out_path_seq)

    feature_path = os.path.join(out_path, 'feature')
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    gt_path = os.path.join(out_path, 'evaluation/{}_MOT/gt'.format(dataset))
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)

    tracker_file = open(os.path.join(out_path, '{}_MOT.txt'.format(dataset)), 'w')

    return data_path, out_path, out_path_seq, feature_path, gt_path, tracker_file

def save_gt(boxs, gt_path):
    # GT format: 42,15,1010,441,40,116,1,1,1
    assert os.path.exists(gt_path)
    boxs = formatting(boxs)
    gt = boxs
    gt[:, 6:10] = 1.
    gt = gt.astype('int')
    np.savetxt(os.path.join(gt_path, 'gt.txt'), gt, delimiter=',')

def init_colors(number_of_colors = 1500):
    # initialize color arrays for visualization
     # should be greater than possible IDs
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color


## Main Function ##############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--model_type', type=str, default='poseShape')
    parser.add_argument('--MTL', type=int, default=1)
    args = parser.parse_args()

    # set cuda
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    codeBase = codebase

    dataset = args.dataset
    init_clusters = 'new_det'#'kmpp'

    if args.model_type=='poseShape':
        model_compnt = 'loc+shape'
    elif args.model_type=='pose':
        model_compnt = 'loc'
    elif args.model_type == 'shape':
        model_compnt = 'shape'
    else:
        print('use --model_type poseShape or pose or shape')

    #TODO: _config file for all parameters
    MTL = args.MTL
    prior = 0 # 0:estimated k-value (no prior k)
    constraints = 1
    mot_evaluation = 1
    DHAE_clustering = 1
    usc_tracker = 1
    print_stat = 1
    keep_track_id_start = []
    dists_cls = {'MNIST': 2.5, 'SPRITE':1.0}
    embed_thr = dists_cls[dataset]

    # set params
    im_w = 128
    im_h = 128
    time_lag = 3
    tau = 2
    score_th = 0.1
    min_cluster_size = 2
    img_y = 28
    img_x = 28

    n_G = 1
    color = init_colors(number_of_colors=1500)
    vis = 1
    det_cluster_id = None
    # initialize tracker ID
    ID_ind = 1
    score_th = 0.2
    # tracklets array
    trackers = {}
    # clustering metrics
    ami = []
    nmi = []
    acc = []
    pur = []
    frame_count = 0

    #set data path
    data_path, out_path, \
    out_path_seq, feature_path,\
    gt_path, tracker_file = init_dirs(codeBase, dataset=dataset)
    # read feature
    x_b, x_m = load_dataset(data_path=data_path, dataset=dataset, USC_tracker=True)
    if mot_evaluation:
        save_gt(x_b, gt_path)
    #init DHAE
    bottleneck_model, \
    mask_encoder_model, \
    box_encoder_model = init_model(codeBasePath=codeBase, dataset=dataset, MTL=MTL)



    #data dimensions check
    ins_28, mask_y, mask_x = x_m.shape
    ins_4, _ = x_b.shape
    assert ins_28 == ins_4, 'number of instances for boxs and masks are unequal'

    # read benchmarq images
    files = glob.glob(os.path.join(data_path, 'imgs/*.png'))
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    total_target = len(np.unique(x_b[:,1]))
    fr=1

    #sequential data window
    fr_start = x_b[:, 0][0]
    fr_end = x_b[:, 0][-1]

    for names in files:
        frame_count+=1
        print(names)
        img = cv2.imread(names)
        img_h,img_w,_ = img.shape
        # check that frame has detection
        if fr in x_b[:, 0]:
            # Select the temporal window for clustering
            if(fr>=fr_start+time_lag-1):
                temp_window_box, \
                temp_window_mask, \
                det_cluster_id, \
                keep_key_frame, k_value = get_temporal_window(fr, time_lag, x_b, x_m,
                                                     det_cluster_id=det_cluster_id)

                # prepare mask and box features for DHAE
                _, pax_box_norm = prepare_data_box(temp_window_box[:,2:6], img_w, img_h)

                #current frame first
                assert pax_box_norm.shape[0]==temp_window_mask.shape[0],'pose.shape==mask.shape but ' \
                                                                         'found {} and {}'.format(pax_box_norm.shape,temp_window_mask.shape)
                assert temp_window_mask.max() in [255, 254], 'found max value of binary mask {}'.format(temp_window_mask.max())
                x_test = temp_window_mask.astype('float32') / 255.
                x_test = x_test.reshape([x_test.shape[0], x_test.shape[1] * x_test.shape[2]])

                # bottleneck feature and reconstruction for temporal window t-10
                mask_x_test = np.reshape(x_test, [x_test.shape[0], img_y, img_x, 1])

                X_frames = temp_window_box[:,0].astype(np.float32)
                X_frames = X_frames.copy(order='C')
                if model_compnt == 'loc+shape':
                    latent_feature = bottleneck_model.predict([mask_x_test, pax_box_norm])
                if model_compnt == 'shape':
                    mask_encoding = mask_encoder_model.predict([mask_x_test])
                if model_compnt == 'loc':
                    latent_feature = box_encoder_model.predict([pax_box_norm])

                if DHAE_clustering:
                    # Select k value as unique target in temporal window
                    if prior:
                        k_class = len(np.unique(temp_window_box[:,1]))
                    if not prior:
                        k_class = k_value.max()
                    # TODO: k-value remain same even though the some data points violate the constraints and unable to be clustered
                    if not constraints:
                        kmeans = KMeans(n_clusters=k_class)
                        kmeans = kmeans.fit(latent_feature)
                        labels = kmeans.predict(latent_feature)
                        cluster_centers = kmeans.cluster_centers_
                    if constraints:
                        labels, cluster_centers = con_kmeans_clustering(fr,
                                                                     X_frames,
                                                                     temp_window_box,
                                                                     latent_feature,
                                                                     temp_window_mask,
                                                                     k_class,
                                                                     names,
                                                                     color,
                                                                     im_h, im_w,
                                                                     tau=tau,
                                                                     embed_thr=embed_thr,
                                                                     init_k=init_clusters,
                                                                     detector='constant_k',
                                                                     vis=False,
                                                                     verbose=True)

                    labels_true = np.array(temp_window_box[:, 1])
                    labels_pred = labels
                    print('GT Labels: ', labels_true)
                    print('Predicted Labels: ', labels_pred)
                    assert len(labels_true) == len(labels_pred)


                    ami_f = metrics.adjusted_rand_score(labels_true, labels_pred)
                    nmi_f = metrics.normalized_mutual_info_score(labels_true, labels_pred)
                    acc_f = 1 - err_rate(labels_true,best_map(labels_true,labels_pred))
                    pur_f = compute_purity(labels_true, labels_pred)
                    ami.append(ami_f)
                    nmi.append(nmi_f)
                    acc.append(acc_f)
                    pur.append(pur_f)

                    if print_stat:
                        print('NMI: {} at Frame {}'.format(np.average(nmi_f),fr))
                        print('AMI or ARI: {} at Frame {}'.format(np.average(ami_f),fr))
                        print('ACC: {} at Frame {}'.format(np.average(acc_f),fr))
                        print('PUR: {} at Frame {}'.format(np.average(pur),fr))

                        # clustering performance score
                        print('Clustering Performance for window size: {}, and number of object: {}'.format(
                            time_lag, total_target))
                        print('Average NMI: {}'.format(np.average(nmi)))
                        print('Average AMI or ARI: {}'.format(np.average(ami)))
                        print('Average ACC: {}'.format(np.average(acc)))
                        print('Average PUR: {}'.format(np.average(pur)))
                        print('total frame: {}'.format(frame_count))

                if usc_tracker:
                    #if constraints:
                    k_class = int(np.max(k_value))
                    init_clusters = 'new_det'
                    key_frame = keep_key_frame[k_class]
                    labels, \
                    cluster_centers = con_kmeans_synthetic(fr,
                                                    X_frames,
                                                    temp_window_box,
                                                    latent_feature,
                                                    temp_window_mask,
                                                    k_class,
                                                    names,
                                                    color,
                                                    time_lag,
                                                    im_h,
                                                    im_w,
                                                    fr_start,
                                                    tau=tau,
                                                    key_frame=key_frame,
                                                    embed_thr=embed_thr,
                                                    init_k=init_clusters,
                                                    dataset_type=dataset,
                                                    verbose=True)

                    refined_det, \
                    final_mask, \
                    det_cluster_id, \
                    ID_ind, trackers = cluster_association_synthetic(fr,
                                                               latent_feature,
                                                               temp_window_box,
                                                               temp_window_mask,
                                                               labels,
                                                               cluster_centers,
                                                               ID_ind,
                                                               time_lag,
                                                               score_th,
                                                               min_cluster_size,
                                                               k_class,
                                                               names, color,
                                                               mask_y, mask_x,
                                                               trackers)
                    print('Total tracked ID:', ID_ind - 1)
        fr += 1
    save_tracks(trackers, tracker_file, out_path_seq, fr_start,
                fr_end, color, im_h, im_w, data_path, img_format='.png', vis=vis)

