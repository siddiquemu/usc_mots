from __future__ import division
import sys
import os
pathname = os.path.dirname(sys.argv[0])
sys.path.insert(0, pathname)
from src.cop_kmeans import cop_kmeans, cop_kmeans_constantK,cop_kmeans_dynamicK
from src.cop_kmeans_det import cop_kmeans_det
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import pairwise_distances
import cv2
import webcolors
from PIL import Image
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt1

def prepare_box_data(x_t,im_w,im_h):
    # x_t is the raw data
    x_0 = x_t[0] / im_w
    y_0 = x_t[1] / im_h
    w = x_t[2] / im_w#np.max(x_t[:, 4])
    h = x_t[3] / im_h#np.max(x_t[:, 5])
    Cx = (x_t[0] + x_t[2] / 2) / im_w
    Cy = (x_t[1] + x_t[3] / 2) / im_h
    x_f = np.array([x_0, y_0,w,h])
    #x_f = np.array([Cx, Cy, w, h,x_0,y_0,area,diag])
    x_f = np.transpose(x_f)
    #x_f = normalize(x_f)
    return x_f

def vectorized_iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

def get_iou(a,b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.
    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero
    Returns:
        (float) The Intersect of Union score.
    """
    # format conversion [x,y,w,h]>>[x1,y1,x2,y2]

    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap
    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou
def l2_distance(point1, point2,latent=None):#w-640, h-480
    if latent is None:
        w=1920#640
        h=1080#480
        point1 = prepare_box_data(point1, w, h)
        point2 = prepare_box_data(point2, w, h)
    if latent is not None:
        point1 = point1
        point2=point2
    return sum([(float(i)-float(j))**2 for (i, j) in zip(point1, point2)])

def detection_con_kmeans(X_angle,boxs_feature,latent_feature,n_clusters,verbose=False):
    must_link = []
    can_not_link = []
    for i, x in enumerate(X_angle):
        for j, y in enumerate(X_angle):
            if(i<j and x==y):
                can_not_link.append((i, j))
    if verbose:
        print('can not link list size:',len(can_not_link))

    labels, centers = cop_kmeans(dataset=latent_feature,
                                   k=n_clusters,
                                   ml = must_link,
                                   cl=can_not_link)
    return np.array(labels), np.array(centers)

def vis_iou_zero(x, boxs_x, y, boxs_y, img, color):
    for i in range(boxs_x.shape[0]):
        img = cv2.rectangle(img, (int(boxs_x[0]), int(boxs_x[1])), \
                            (int(boxs_x[2]),
                             int(boxs_x[3])),
                            webcolors.hex_to_rgb(color[int(x)]) , 4)
    for i in range(boxs_y.shape[0]):
        img = cv2.rectangle(img, (int(boxs_y[0]), int(boxs_y[1])), \
                            (int(boxs_y[2]),
                             int(boxs_y[3])),
                            webcolors.hex_to_rgb(color[int(y)]) , 4)
    return img

def dist_pairwise(boxx,boxy,x,y,img,color):
    l2dists = []
    for i in range(boxx.shape[0]):
        for j in range(boxy.shape[0]):
            iou = get_iou(boxx[i],boxy[j])
            boxs_x = prepare_box_data(boxx[i], 1920, 1080)
            boxs_y = prepare_box_data(boxy[j], 1920, 1080)
            #ious.append(iou)
            if(iou==0):
                img = vis_iou_zero(x, boxx[i], y, boxy[j], img, color)
                l2dist = l2_distance(boxs_x, boxs_y)
                l2dists.append(l2dist)
    return np.array(l2dists),img

    # compute histogram of the distance profile
def dist_thr(fr,i,j,x,y,X_frames,boxs,n_clusters,img,ml,cl,vis):
    #import matplotlib.pyplot as plt1
    boxs_x = boxs[np.where(X_frames==x),:][0]
    boxs_y = boxs[np.where(X_frames == y),:][0]
    #visualize two adjacent frame detection where iou(i,j)==0 pair found
    dist_split,img = dist_pairwise(boxs_x, boxs_y,x,y,img,color)

    if vis:
        im = Image.fromarray(img)
        if cl:
            im.save('/media/siddique/Data/CLASP2018/img/MOT/MOT17Det/train/MOT17-13/'
                'box_mask_overlaid/problems/cluster_on_image/' +str(fr)+ 'can_not_iou0' + str(i) + '_' + names[-10:])
        if ml:
            im.save('/media/siddique/Data/CLASP2018/img/MOT/MOT17Det/train/MOT17-13/'
                'box_mask_overlaid/problems/cluster_on_image/' +str(fr)+ 'must_iou0' + str(i) + '_' + names[-10:])


    #avg_dist = np.average(dist)
    hist_dist = np.histogram(dist_split, bins=n_clusters)
    #plt1.hist(dist_split,bins=n_clusters)
    #plt1.show()
    #plt1.pause(1)
    #plt1.close()
    #d_thr = (hist_dist[1][1]+hist_dist[1][0])/2
    d_thr = hist_dist[1][1]
    #print('distance threshold',d_thr)
    #print('minimum distance for merging splitted cluster',d_thr)
    return d_thr

def maskRLE(ref_box,final_mask,im_h,im_w):
    box_coord = ref_box[2:6].astype(int)
    # why only consider bbox boundary for mask???? box can miss part of the object
    # overlay mask on image frame
    x_0 = max(box_coord[0], 0)
    x_1 = min(box_coord[0] + box_coord[2] , im_w)
    y_0 = max(box_coord[1], 0)
    y_1 = min(box_coord[1] + box_coord[3] , im_h)
    '''
    w = ref_box[2] - ref_box[0] + 1
    h = ref_box[3] - ref_box[1] + 1
    w = np.maximum(w, 1)
    h = np.maximum(h, 1)
    '''
    if final_mask is not None:
        mask = cv2.resize(final_mask, (box_coord[2], box_coord[3]))
        # apply theshold on scoremap!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!cfg.MRCNN.THRESH_BINARIZE
        mask = np.array(mask >= 0.5, dtype=np.uint8)
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        # mask transfer on image cooordinate
        #im_mask[y_0:y_1, x_0:x_1] = mask
        im_mask[y_0:y_1, x_0:x_1] = mask[
                                    (y_0 - box_coord[1]):(y_1 - box_coord[1]),
                                    (x_0 - box_coord[0]):(x_1 - box_coord[0])
                                    ]
        # RLE format of instance binary mask
        rle = maskUtils.encode(np.asfortranarray(im_mask))
    return rle

def mask_iou(a, b, criterion="union"):
  is_crowd = criterion != "union"
  return maskUtils.iou([a.mask], [b.mask], [is_crowd])[0][0]

def maskIOU(boxs_feature_i,mask_i,boxs_feature_j,mask_j,im_h,im_w):
    #rle_i = maskRLE(boxs_feature_i,mask_i,im_h,im_w)
    #rle_j = maskRLE(boxs_feature_j,mask_j,im_h,im_w)
    rle_i = {'size': [im_h,im_w], 'counts': bytes(mask_i, 'utf-8')}
    rle_j = {'size': [im_h,im_w], 'counts': bytes(mask_j, 'utf-8')}
    #maskiou = maskUtils.area(maskUtils.merge([rle_i, rle_j], intersect=True)) / \
           # maskUtils.area(maskUtils.merge([rle_i, rle_j],intersect=False))
    return maskUtils.iou([rle_i], [rle_j], [False])[0][0]
#synthetic: only clustering evaluation
def con_kmeans_clustering(fr, X_frames, boxs_feature, latent_feature,mask,
                        n_clusters, names, color, im_h, im_w,init_k='kmpp',
                          tau=1, embed_thr=None, detector='noiseless',vis=False, verbose=False):
    # ------------------------------------------------------------------
    # Can not link graph: Temporally adjacent pairs should have smaller
    #                     distance than temporally distant pairs.
    # d_th(i,j): Compute distance threshold d_th from the already labelled
    #            targets in the sliding window.
    #
    # Parameter: MNIST: d_th = 2.5, delta_t = 1
    # ------------------------------------------------------------------
    track_neg_id = dict()
    must_link = []
    can_not_link_iou = []
    # for iou
    boxs = np.transpose(np.array([boxs_feature[:, 2], boxs_feature[:, 3], boxs_feature[:, 2]
                                  + boxs_feature[:, 4], boxs_feature[:, 3] + boxs_feature[:, 5]]))
    uni, cnts = np.unique(X_frames, return_counts=True)
    # for l2 distance (not normalized)
    box_f = np.transpose(np.array([boxs_feature[:, 2]/im_w, boxs_feature[:, 3]/im_h,
                                   boxs_feature[:, 4]/im_w, boxs_feature[:, 5]/im_h]))
    # Main function for G_w = f(G_cl,G_ml)
    for i, x in enumerate(X_frames):
        for j, y in enumerate(X_frames):
            # computer distance between the nodes of interest
            embed_dist = pairwise_distances(latent_feature[i].reshape(1, -1), latent_feature[j].reshape(1, -1))
            #dist_4D = pairwise_distances(box_f[i].reshape(1, -1), box_f[j].reshape(1, -1))
            boxIOU = get_iou(boxs[i], boxs[j])
            #CAN-NOT LINK
            if (i < j and x == y ):
                can_not_link_iou.append((i, j))
                continue
            # Focus on box IOU
            #if i < j and x > y and x-y<=1 and boxIOU <= 0: #  for temporal x-y==1
               # can_not_link_iou.append((i, j))
                #continue

            # Focus on embeddings
            if x>y and x - y <= tau and embed_dist>=embed_thr:
                can_not_link_iou.append((i, j))
                continue

    if verbose:
        print('can not link list size:', len(can_not_link_iou))

    if verbose:
        print('must link list size:', len(must_link))
    #For Noiseless Detections: all evaluation
    if detector == 'constant_k':
        labels, centers = cop_kmeans_constantK(dataset=latent_feature,
                                 labels=boxs_feature[:, 1],
                                 temporal_w=X_frames,
                                 current_f=fr,
                                 k=n_clusters,
                                 ml=must_link,
                                 cl=can_not_link_iou,
                                initialization=init_k)


    #For Noisy Detections: MOTS evaluaation
    if detector=='dynamic_k':
        labels, centers = cop_kmeans_dynamicK(dataset=latent_feature,
                                     labels=boxs_feature[:, 1],
                                     temporal_w=X_frames,
                                     current_f=fr,
                                     k=n_clusters,
                                     ml=must_link,
                                     cl=can_not_link_iou,
                                     initialization=init_k)

    return np.array(labels), np.array(centers)
#synthetic
def con_kmeans_synthetic(fr, X_frames, boxs_feature, latent_feature,mask_test,
                        n_clusters, names, color, t_lag, im_h, im_w, fr_start, key_frame=None,
                        tau=1, embed_thr=None,init_k=None, dataset_type=None, verbose=False):
    # ------------------------------------------------------------------
    # NIST-MOT, Sprites-MOT
    # Can not link graph: Temporally adjacent pairs should have smaller
    #                     distance than temporally distant pairs.
    # d_th(i,j): Compute distance threshold d_th from the already labelled
    #            targets in the sliding window.
    #
    # Parameter: MNIST: d_th = 2.5, delta_t = 2, SPRITE: d_th = 2.5, delta_t = 2
    # ------------------------------------------------------------------
    track_neg_id = dict()
    must_link = []
    can_not_link_iou = []
    vis = 0
    # for iou: (x1,y1), (x2,y2)
    boxs = np.transpose(np.array([boxs_feature[:, 2], boxs_feature[:, 3], boxs_feature[:, 2]
                                  + boxs_feature[:, 4], boxs_feature[:, 3] + boxs_feature[:, 5]]))
    uni, cnts = np.unique(X_frames, return_counts=True)
    # for l2 distance (not normalized)
    box_f = np.transpose(np.array([boxs_feature[:, 2]/im_w, boxs_feature[:, 3]/im_h,
                                   boxs_feature[:, 4]/im_w, boxs_feature[:, 5]/im_h]))
    # Main function for G_w = f(G_cl,G_ml)
    for i, x in enumerate(X_frames):
        track_neg_id[i] = set()
        for j, y in enumerate(X_frames):
            # computer distance between the nodes of interest
            embed_dist = pairwise_distances(latent_feature[i].reshape(1, -1), latent_feature[j].reshape(1, -1))
            #dist_4D = pairwise_distances(box_f[i].reshape(1, -1), box_f[j].reshape(1, -1))
            boxIOU = get_iou(boxs[i], boxs[j])
            # MUST LINK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # associated detections have same label?? and same class
            if (i<j and x>y and x!=fr
                    and boxs_feature[:, -1][i] > 0
                    and boxs_feature[:, -1][j] > 0
                    and boxs_feature[:, -1][i] == boxs_feature[:, -1][j]):
                must_link.append((i, j))
                if verbose: print('embed distance for ML-ID-{}: {}'.format(boxs_feature[:, -1][i], embed_dist))
                continue
            #CAN-NOT LINK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Detections from the same frame??
            if (i < j and x == y ):
                can_not_link_iou.append((i, j))
                continue
            # Detctions are already initialized and labels are different??
            if (i < j and x > y and boxs_feature[:, -1][i] > 0
                    and boxs_feature[:, -1][j] > 0
                    and boxs_feature[:, -1][i] != boxs_feature[:, -1][j]):
                can_not_link_iou.append((i, j))
                continue

            if fr>fr_start+t_lag-1 and i < j and x > y:
                # t_i-t_j <= tau : Focus on motion feature

                if x==fr and boxs_feature[:, -1][i] == 0:
                    if dataset_type=='SPRITE..':
                        if abs(x - y) <= tau and boxIOU <= 0: # TODO: box iou based constrains are bad for MNIST
                            can_not_link_iou.append((i, j))
                            continue
                    else:
                        if (abs(x - y) <=tau and embed_dist>=embed_thr):
                            can_not_link_iou.append((i, j))
                            continue
                # Detections are unlabeled at previous frames?
                if x!=fr and boxs_feature[:, -1][i]==0:
                    if dataset_type == 'SPRITE..':
                        if abs(x - y) <= tau and boxIOU <= 0:
                            can_not_link_iou.append((i, j))
                            continue
                    else:
                        if (abs(x - y) <=tau and embed_dist>=embed_thr):
                            can_not_link_iou.append((i, j))
                            continue

            else:# for first window in video
                # t_i-t_j <= tau : Focus on motion feature
                if dataset_type == 'SPRITE..':
                    if (i < j and x > y and abs(x - y) <= tau and boxIOU <= 0):
                        can_not_link_iou.append((i, j))
                        continue
                else:
                    if (i < j and x > y and abs(x - y) <= tau and embed_dist>=embed_thr):
                        can_not_link_iou.append((i, j))
                        continue

    if verbose:
        print('can not link list size:', len(can_not_link_iou))

    if verbose:
        print('must link list size:', len(must_link))

    '''
    labels, centers = cop_kmeans_dynamicK(dataset=latent_feature,
                                 labels=boxs_feature[:, -1],
                                 temporal_w=X_frames,
                                 current_f=fr,
                                 k=n_clusters,
                                 ml=must_link,
                                 cl=can_not_link_iou,
                                 initialization=init_k)
    '''
    labels, centers = cop_kmeans(dataset=latent_feature,
                             labels=boxs_feature[:, -1],
                             temporal_w=X_frames,
                             current_f=fr,
                             k=n_clusters,
                             ml=must_link,
                             cl=can_not_link_iou,
                             k_frame=key_frame,
                             initialization=init_k)
    return np.array(labels), np.array(centers)
#real
def constraints_kmeans(fr, X_frames, boxs_feature,
                       latent_feature,mask, n_clusters,
                       names, color, t_lag, im_h, im_w,
                       fr_start, iou_thr=0.0, k_frame=None, alpha_t=2,
                       graph_metric='mask_iou', initialization='kmpp', verbose=False):
    # ------------------------------------------------------------------
    # KITTI, MOT17, CLASP
    # boxs_feature:
    # mask: RLEs > used to compute mask-IoU
    # Can not link graph: Temporally adjacent pairs should have smaller
    #                     distance than temporally distant pairs.
    # d_th(i,j): Compute distance threshold d_th from the already labelled
    #            targets in the sliding window.
    #
    # Parameter: MNIST: d_th = 4, delta_t = 2 (only pose variation)
    # ------------------------------------------------------------------
    must_link = []
    can_not_link_iou = []
    vis = 0
    # for iou: (x1,y1), (x2,y2)
    boxs = np.transpose(np.array([boxs_feature[:, 2], boxs_feature[:, 3], boxs_feature[:, 2]
                                  + boxs_feature[:, 4], boxs_feature[:, 3] + boxs_feature[:, 5]]))
    uni, cnts = np.unique(X_frames, return_counts=True)
    extrapolate_flags = boxs_feature[:, 6]
    # for l2 distance (not normalized)
    #box_f = np.transpose(np.array([boxs_feature[:, 2]/im_w, boxs_feature[:, 3]/im_h,
                                   #boxs_feature[:, 4]/im_w, boxs_feature[:, 5]/im_h]))
    # Main function for G_w = f(G_cl,G_ml)
    for i, x in enumerate(X_frames):
        for j, y in enumerate(X_frames):
            embed_dist = pairwise_distances(latent_feature[i].reshape(1, -1), latent_feature[j].reshape(1, -1))
            # MUST LINK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # associated detections have same label?? and same class
            if (i<j and x>y and x!=fr
                    and boxs_feature[:, 8][i] > 0
                    and boxs_feature[:, 8][j] > 0
                    and boxs_feature[:, 7][i] == boxs_feature[:, 7][j]
                    and boxs_feature[:, 8][i] == boxs_feature[:, 8][j]):
                must_link.append((i, j))
                if verbose: print('embed distance for ML-ID-{}: {}'.format(boxs_feature[:, 8][i], embed_dist))
                continue
            #CAN-NOT LINK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Detections from the same frame??
            if (i < j and x == y ):
                can_not_link_iou.append((i, j))
                continue
            # Detctions are already initialized and labels are different??
            if (i < j and x > y and boxs_feature[:, 8][i] > 0
                    and boxs_feature[:, 8][j] > 0
                    and boxs_feature[:, 8][i] != boxs_feature[:, 8][j]):
                can_not_link_iou.append((i, j))
                continue


            #start_time = time.time()
            #maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
            #print(time.time()-start_time)
            #dist_4D = pairwise_distances(box_f[i].reshape(1, -1), box_f[j].reshape(1, -1))
            #boxIOU = get_iou(boxs[i], boxs[j])
            #boxIOU=maskiou
            if fr>fr_start+t_lag-1 and i < j and x > y:
                #for current frame detections
                if x==fr and boxs_feature[:, 8][i] == 0:
                    # apply box_iou for extrapolated box since we are not warping mask: simply propagate last segmented RGB
                    if graph_metric=='box_iou':
                        IOU = get_iou(boxs[i], boxs[j])

                    elif graph_metric=='mask_iou':
                        if 10 in [extrapolate_flags[i],extrapolate_flags[j]]:
                            IOU = get_iou(boxs[i], boxs[j])

                        else:
                            maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
                            IOU = maskiou

                    if (abs(x - y) <= alpha_t and IOU <= iou_thr): # TODO: CL only for xRy and x-y==1
                        can_not_link_iou.append((i, j))
                        continue

                    #if (abs(x - y) >alpha_t and embed_dist>=8):#kitti - car:4, ped: 8, clasp = 4
                        # TODO: check jth detection is already in CL
                       # can_not_link_iou.append((i, j))
                        #continue
                #for unassociated detection in the window (constraints in previous frame: if x=9, y=8)
                if x!=fr and boxs_feature[:, 8][i]==0:
                    if graph_metric=='box_iou':
                        IOU = get_iou(boxs[i], boxs[j])

                    elif graph_metric == 'mask_iou':
                        if 10 in [extrapolate_flags[i], extrapolate_flags[j]]:
                            IOU = get_iou(boxs[i], boxs[j])

                        else:
                            maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
                            IOU = maskiou

                    if (abs(x - y) <= alpha_t and IOU <= iou_thr):
                        can_not_link_iou.append((i, j))
                        continue

            else:# for first window in video
                if (i < j and x > y and abs(x - y) <= alpha_t ):
                    if graph_metric=='box_iou':
                        IOU = get_iou(boxs[i], boxs[j])

                    elif graph_metric == 'mask_iou':
                        maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
                        IOU = maskiou

                    if IOU <= iou_thr:
                        can_not_link_iou.append((i, j))
                    continue

    if verbose:
        print('can not link list size:', len(can_not_link_iou))

    if verbose:
        print('must link list size:', len(must_link))

    labels, centers = cop_kmeans(dataset=latent_feature,
                                 labels=boxs_feature[:, 8],
                                 temporal_w=X_frames,
                                 current_f=fr,
                                 k=n_clusters,
                                 ml=must_link,
                                 cl=can_not_link_iou,
                                 k_frame = k_frame,
                                 initialization=initialization,
                                 verbose=verbose)

    return np.array(labels), np.array(centers)