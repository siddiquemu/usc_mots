from __future__ import division
import sys
import os
pathname = os.path.dirname(sys.argv[0])
sys.path.insert(0, pathname)
from cop_kmeans import cop_kmeans, cop_kmeans_constantK,cop_kmeans_dynamicK
from cop_kmeans_det import cop_kmeans_det
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


def con_kmeans_det(fr, fr_angles, boxs_feature, latent_feature,mask,
                        n_clusters, names, color, im_h, im_w,  verbose=False):
    # ------------------------------------------------------------------
    # ***Can not link graph: detections from same orientation can not link together, detections from different orientation with IoU<=0.3 cant not link
    # ***Must Link graph: detections from different orientation with IoU>=0.8 and same class must link together
    # d_th(i,j): Compute distance threshold d_th from the already labelled
    #            targets in the sliding window.
    #
    # ------------------------------------------------------------------
    must_link = []
    can_not_link_iou = []
    # for iou
    boxs = np.transpose(np.array([boxs_feature[:, 2], boxs_feature[:, 3], boxs_feature[:, 2]
                                  + boxs_feature[:, 4], boxs_feature[:, 3] + boxs_feature[:, 5]]))
    # for l2 distance (not normalized)
    box_f = np.transpose(np.array([boxs_feature[:, 2]/im_w, boxs_feature[:, 3]/im_h,
                                   boxs_feature[:, 4]/im_w, boxs_feature[:, 5]/im_h]))
    #c_xy = np.transpose(np.array([box_f[:,0]+box_f[:,2]/2, box_f[:,1]+box_f[:,3]/2]))
    #structural_constraint = pairwise_distances(c_xy/temp_diff, c_xy/temp_diff)
    #latent_feature = np.concatenate((latent_feature, structural_constraint), axis=1)
    # Main function for G_w = f(G_cl,G_ml)
    for i, x in enumerate(fr_angles):
        for j, y in enumerate(fr_angles):
            # computer distance between the nodes of interest
            embed_dist = pairwise_distances(latent_feature[i].reshape(1, -1), latent_feature[j].reshape(1, -1))
            #dist_4D = pairwise_distances(box_f[i].reshape(1, -1), box_f[j].reshape(1, -1))
            boxIOU = get_iou(boxs[i], boxs[j])
            #CAN-NOT LINK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Detections from different class???
            # Detections from the same orientation (angles)??
            if (i < j and x == y ):
                can_not_link_iou.append((i, j))
                continue
            if i < j and x > y:
                if (embed_dist>=20):#6 # higher threshold value ensure no postively matched pairs in can not link
                    can_not_link_iou.append((i, j))
                    continue
                #maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
                #boxIOU = maskiou
                if boxIOU<0.2: # TODO: 0 or <0.3 or <0.5
                    can_not_link_iou.append((i, j))
                    continue
            # TODO: IoU confusion region: 0.5<=IoU<0.8 (k-means loss function can handle it??)
            # MUST LINK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Detections have same label?? and same class

            '''
            if (i<j and x>y
                    and boxIOU >= 0.8
                    and boxs_feature[:, 7][i] == boxs_feature[:, 7][j]):
                must_link.append((i, j))
            '''
    if verbose:
        print('can not link list size:', len(can_not_link_iou))
    if verbose:
        print('must link list size:', len(must_link))
    """"
    labels, centers = cop_kmeans_det(dataset=latent_feature,
                                 current_f=fr,
                                 k=n_clusters,
                                 ml=must_link,
                                 cl=can_not_link_iou)
    """
    labels, centers = cop_kmeans_constantK(dataset=latent_feature,
                             labels=boxs_feature[:, 1],
                             temporal_w=fr_angles,
                             current_f=fr,
                             k=n_clusters,
                             ml=must_link,
                             cl=can_not_link_iou)

    return np.array(labels), np.array(centers)

def temporal_con_kmeans_clustering(fr, X_frames, boxs_feature, latent_feature,mask,
                        n_clusters, names, color, im_h, im_w,init_k='kmpp',
                                   detector='noiseless',vis=False, verbose=False):
    # ------------------------------------------------------------------
    # Can not link graph: Temporally adjacent pairs should have smaller
    #                     distance than temporally distant pairs.
    # d_th(i,j): Compute distance threshold d_th from the already labelled
    #            targets in the sliding window.
    #
    # Parameter: MNIST: d_th = 4, delta_t = 2 (only pose variation)
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

            if fr == 10 and vis:
                mng = plt1.get_current_fig_manager()
                mng.full_screen_toggle()
                plt1.subplot(1,2,1)
                plt1.imshow(mask[i])
                plt1.subplot(1,2,2)
                plt1.imshow(mask[j])
                plt1.text(0, 5, 'IOU: {}, Distance: {} at delta_t = {}'.format(boxIOU, embed_dist[0], x-y),color='cyan', fontsize=15)
                plt1.axis('off')
                plt1.pause(1)
                plt1.waitforbuttonpress(0)
                plt1.close()
            #CAN-NOT LINK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Detections from different class???
            # Detections from the same frame??

            if (i < j and x == y ):
                can_not_link_iou.append((i, j))
                continue

            #if i < j and x > y and x-y<=1 and boxIOU <= 0: #  for temporal x-y==1
                #can_not_link_iou.append((i, j))
                #continue

            # TODO: for x-y>2 t_i-t_j > tau : Focus on embeddings
            if x>y and x - y >= 1 and embed_dist>=4:
                can_not_link_iou.append((i, j))
                continue


            # MUST LINK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Detections have same label?? and same class
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

def temporal_con_kmeans_revisit(fr, X_frames, boxs_feature, latent_feature,mask_test,
                        n_clusters, names, color, t_lag, im_h, im_w, fr_start, verbose=False):
    # ------------------------------------------------------------------
    # KITTI, MOT17, CLASP
    # Can not link graph: Temporally adjacent pairs should have smaller
    #                     distance than temporally distant pairs.
    # d_th(i,j): Compute distance threshold d_th from the already labelled
    #            targets in the sliding window.
    #
    # Parameter: MNIST: d_th = 4, delta_t = 2 (only pose variation)
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
    #c_xy = np.transpose(np.array([box_f[:,0]+box_f[:,2]/2, box_f[:,1]+box_f[:,3]/2]))
    #temp_diff = pairwise_distances(X_frames.max()-X_frames,X_frames.max()-X_frames, metric='l1')
    #temp_diff[temp_diff.nonzero()]=1
    #structural_constraint = pairwise_distances(c_xy/temp_diff, c_xy/temp_diff)
    #latent_feature = np.concatenate((latent_feature, structural_constraint), axis=1)
    # Main function for G_w = f(G_cl,G_ml)
    for i, x in enumerate(X_frames):
        track_neg_id[i] = set()
        for j, y in enumerate(X_frames):
            # computer distance between the nodes of interest
            embed_dist = pairwise_distances(latent_feature[i].reshape(1, -1), latent_feature[j].reshape(1, -1))
            #dist_4D = pairwise_distances(box_f[i].reshape(1, -1), box_f[j].reshape(1, -1))
            boxIOU = get_iou(boxs[i], boxs[j])

            if fr == 10 and vis:
                mng = plt1.get_current_fig_manager()
                mng.full_screen_toggle()
                plt1.subplot(1,2,1)
                plt1.imshow(mask_test[i])
                plt1.subplot(1,2,2)
                plt1.imshow(mask_test[j])
                plt1.text(0, 5, 'IOU: {}, Distance: {} at delta_t = {}'.format(boxIOU, embed_dist[0], abs(x-y)),color='cyan', fontsize=12)
                plt1.axis('off')
                plt1.pause(1)
                plt1.waitforbuttonpress(0)
                plt1.close()
            #CAN-NOT LINK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Detections from different class???
            #if (i < j and boxs_feature[:, 7][i] != boxs_feature[:, 7][j] ):
                #can_not_link_iou.append((i, j))
                #continue
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

            if fr>fr_start+t_lag-1 and i < j and x > y:
                # t_i-t_j <= tau : Focus on motion feature
                # Detections at current frame?? TODO: Min-Max Game?? How to assign new detections to already initialized tracklets or create new tracklets??
                if(x==fr):#or boxs_feature[:, 8][j]==0)  x==fr and
                    # TODO: immediate parents may have at t_lag - min_cluster_size
                    #maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
                    #boxIOU = maskiou
                    if (abs(x - y) <= 1 and boxIOU <= 0): # TODO: Replace bboxIOU with maskIOU
                        # pairwise_distances(latent_feature[i].reshape(1, -1), latent_feature[j].reshape(1, -1))
                        #boxs_feature[j][8] = 0 > Some data oints may reamin un-initialized (unable to clustered)
                        #
                        if boxs_feature[j][8] not in track_neg_id[i] and boxs_feature[j][8]>0:
                            assert boxs_feature[j][8] not in track_neg_id[i],'duplicate target pairs found in can not link'
                            Pj = np.where(boxs_feature[:,8]==boxs_feature[j][8])[0] # set of negative targets for all previous frames
                            #print('y and ID',y,boxs_feature[j][8])
                            #print('iou=0 vs embed', embed_dist)
                            #assert boxs_feature[j][8]>0, 'Unlabeled vs unlabeled matching found at current frame'
                            for k in Pj: # all negative predecessors
                                can_not_link_iou.append((i, k))
                            track_neg_id[i].add(boxs_feature[j][8])
                        #print('Embedding distance {}, reliable distance {} at can-not link: '.format(exp_dist,d_threshold))
                    '''
                        if (x - y < 1 and boxIOU > 0):
                        # test for N_iou>1
                        Bj_ind = np.where(X_frames == y)[0]
                        Bj = boxs[ Bj_ind]
                        i_ious = vectorized_iou(boxs[i].reshape(1, -1), Bj)[0]
                        iou_nonzero_ind = Bj_ind[np.where(i_ious>0)]
                        iou_nonzero = i_ious[np.where(i_ious>0)]
                        N_iou = len(iou_nonzero_ind)
                        #pairwise_distances(latent_feature[i].reshape(1, -1), latent_feature[Bj_ind])
                        #if N_iou>1: # new detection overlap with multiple track heads
                        #pairwise_distances(latent_feature[i].reshape(1, -1), latent_feature[iou_nonzero_ind])
                        neg_id = iou_nonzero_ind [np.where(iou_nonzero<0.1)] # TODO: for x-y=1, minimum iou 0.3 is considered, embedding may be used? mask iou?
                        if len(neg_id)>0:
                            for k in neg_id:
                                Pj = np.where(boxs_feature[:, 8] == boxs_feature[k][8])[0]
                                for l in Pj:  # all negative parents
                                    can_not_link_iou.append((i, l))
                    '''

                    # t_i-t_j > tau : Focus on embeddings
                    #r_thr = np.array(list(max_dist_thr[i])) + np.array(list(max_dist_thr[i])) / 2
                    #TODO: Structural and motion constraints: location and velocity difference between objects
                    #TODO: Velocity difference will consider that the objects moving in different direction

                    if (abs(x - y) >1 and embed_dist>=4):#kitti - car:4, ped: 8, clasp = 4
                        # TODO: check jth detection is already in CL
                        can_not_link_iou.append((i, j))
                        if fr==161 or fr==162 or fr==163:
                            print('debug embeddings')
                        continue


                        #print('x-y: r_thr',x-y,r_thr)
                # Detections are unlabeled at previous frames?
                #maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
                #boxIOU = maskiou
                if x!=fr and boxs_feature[:, 8][i]==0:
                    if (abs(x - y) <= 1 and boxIOU <= 0):
                        can_not_link_iou.append((i, j))
                        continue

            else:# for first window in video
                # t_i-t_j <= tau : Focus on motion feature
                #maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
                #boxIOU = maskiou
                if (i < j and x > y and abs(x - y) <= 1 and boxIOU <= 0):
                    can_not_link_iou.append((i, j))
                    continue
                # TODO: for x-y>2 t_i-t_j > tau : Focus on embeddings
                '''
                if (i < j and x > y and x - y > 2):  # and get_iou(boxs[i], boxs[j]) <=0):#x-y<=2 and
                    # Focus on location feature
                    if (exp_dist > .027):  # exp_dist*1.2???
                        print('added in can not link........')
                        can_not_link_iou.append((i, j))
                '''
            # MUST LINK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Detections have same label?? and same class
            if (i<j and x>y
                    and boxs_feature[:, 8][i] > 0
                    and boxs_feature[:, 8][j] > 0
                    and boxs_feature[:, 7][i] == boxs_feature[:, 7][j]
                    and boxs_feature[:, 8][i] == boxs_feature[:, 8][j]):
                must_link.append((i, j))

                # print('distance at must link: ',exp_dist)

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
                                 cl=can_not_link_iou)

    return np.array(labels), np.array(centers)

def temporal_con_kmeans_mnist(fr, X_frames, boxs_feature, latent_feature,mask_test,
                        n_clusters, names, color, t_lag, im_h, im_w, fr_start, key_frame=None,
                        embed_thr=None,init_k=None, dataset_type=None, verbose=False):
    # ------------------------------------------------------------------
    # KITTI, MOT17, CLASP
    # Can not link graph: Temporally adjacent pairs should have smaller
    #                     distance than temporally distant pairs.
    # d_th(i,j): Compute distance threshold d_th from the already labelled
    #            targets in the sliding window.
    #
    # Parameter: MNIST: d_th = 4, delta_t = 2 (only pose variation)
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

            if fr == 10 and vis:
                mng = plt1.get_current_fig_manager()
                mng.full_screen_toggle()
                plt1.subplot(1,2,1)
                plt1.imshow(mask_test[i])
                plt1.subplot(1,2,2)
                plt1.imshow(mask_test[j])
                plt1.text(0, 5, 'IOU: {}, Distance: {} at delta_t = {}'.format(boxIOU, embed_dist[0], abs(x-y)),color='cyan', fontsize=12)
                plt1.axis('off')
                plt1.pause(1)
                plt1.waitforbuttonpress(0)
                plt1.close()

            tau=1
            if fr>fr_start+t_lag-1 and i < j and x > y:
                # t_i-t_j <= tau : Focus on motion feature

                if x==fr and boxs_feature[:, -1][i] == 0:
                    if dataset_type=='SPRITE..':
                        if abs(x - y) <= tau and boxIOU <= 0: # TODO: bos iou based constrains are bad for MNIST
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
                        if (abs(x - y) <=tau and embed_dist>=embed_thr):#mnist, sprite:4
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

def temporal_constraints(fr, X_frames, boxs_feature, latent_feature, mask, mask_test,
                                n_clusters, names, color, t_lag, im_h, im_w, fr_start, verbose=False):
    # ------------------------------------------------------------------
    # KITTI, MOT17, CLASP
    # Can not link graph: Temporally adjacent pairs should have smaller
    #                     distance than temporally distant pairs.
    # d_th(i,j): Compute distance threshold d_th from the already labelled
    #            targets in the sliding window.
    #
    # Parameter: MNIST: d_th = 4, delta_t = 2 (only pose variation)
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
    box_f = np.transpose(np.array([boxs_feature[:, 2] / im_w, boxs_feature[:, 3] / im_h,
                                   boxs_feature[:, 4] / im_w, boxs_feature[:, 5] / im_h]))
    # c_xy = np.transpose(np.array([box_f[:,0]+box_f[:,2]/2, box_f[:,1]+box_f[:,3]/2]))
    # temp_diff = pairwise_distances(X_frames.max()-X_frames,X_frames.max()-X_frames, metric='l1')
    # temp_diff[temp_diff.nonzero()]=1
    # structural_constraint = pairwise_distances(c_xy/temp_diff, c_xy/temp_diff)
    # latent_feature = np.concatenate((latent_feature, structural_constraint), axis=1)
    # Main function for G_w = f(G_cl,G_ml)
    for i, x in enumerate(X_frames):
        track_neg_id[i] = set()
        for j, y in enumerate(X_frames):
            # computer distance between the nodes of interest
            embed_dist = pairwise_distances(latent_feature[i].reshape(1, -1), latent_feature[j].reshape(1, -1))
            # dist_4D = pairwise_distances(box_f[i].reshape(1, -1), box_f[j].reshape(1, -1))
            boxIOU = get_iou(boxs[i], boxs[j])

            if fr == 10 and vis:
                mng = plt1.get_current_fig_manager()
                mng.full_screen_toggle()
                plt1.subplot(1, 2, 1)
                plt1.imshow(mask_test[i])
                plt1.subplot(1, 2, 2)
                plt1.imshow(mask_test[j])
                plt1.text(0, 5, 'IOU: {}, Distance: {} at delta_t = {}'.format(boxIOU, embed_dist[0], abs(x - y)),
                          color='cyan', fontsize=12)
                plt1.axis('off')
                plt1.pause(1)
                plt1.waitforbuttonpress(0)
                plt1.close()
            # CAN-NOT LINK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Detections from different class???
            # if (i < j and boxs_feature[:, 7][i] != boxs_feature[:, 7][j] ):
            # can_not_link_iou.append((i, j))
            # continue
            # Detections from the same frame??
            if (i < j and x == y):
                can_not_link_iou.append((i, j))
                continue
            # Detctions are already initialized and labels are different??
            if (i < j and x > y and boxs_feature[:, 8][i] > 0
                    and boxs_feature[:, 8][j] > 0
                    and boxs_feature[:, 8][i] != boxs_feature[:, 8][j]):
                can_not_link_iou.append((i, j))
                continue

            if fr > fr_start + t_lag - 1 and i < j and x > y:
                # t_i-t_j <= tau : Focus on motion feature
                # Detections at current frame?? TODO: Min-Max Game?? How to assign new detections to already initialized tracklets or create new tracklets??
                # TODO: immediate parents may have at t_lag - min_cluster_size
                # maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
                # boxIOU = maskiou
                if (abs(x - y) <= 1 and boxIOU <= 0):  # TODO: Replace bboxIOU with maskIOU
                    can_not_link_iou.append((i, j))

                # t_i-t_j > tau : Focus on embeddings
                # r_thr = np.array(list(max_dist_thr[i])) + np.array(list(max_dist_thr[i])) / 2
                # TODO: Structural and motion constraints: location and velocity difference between objects
                # TODO: Velocity difference will consider that the objects moving in different direction
                if (abs(x - y) > 1 and embed_dist >= 4):  # kitti - car:4, ped: 8, clasp = 4
                    # TODO: check jth detection is already in CL
                    can_not_link_iou.append((i, j))
                    continue

            else:  # for first window in video
                # t_i-t_j <= tau : Focus on motion feature
                # maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
                # boxIOU = maskiou
                if (i < j and x > y and abs(x - y) <= 1 and boxIOU <= 0):
                    can_not_link_iou.append((i, j))
                    continue
                # TODO: for x-y>2 t_i-t_j > tau : Focus on embeddings
            # MUST LINK>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Detections have same label?? and same class
            if (i < j and x > y
                    and boxs_feature[:, 8][i] > 0
                    and boxs_feature[:, 8][j] > 0
                    and boxs_feature[:, 7][i] == boxs_feature[:, 7][j]
                    and boxs_feature[:, 8][i] == boxs_feature[:, 8][j]):
                must_link.append((i, j))
                # print('distance at must link: ',exp_dist)

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
                                 cl=can_not_link_iou)

    return np.array(labels), np.array(centers)

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
                        boxIOU = get_iou(boxs[i], boxs[j])

                    elif graph_metric=='mask_iou':
                        if 10 in [extrapolate_flags[i],extrapolate_flags[j]]:
                            boxIOU = get_iou(boxs[i], boxs[j])

                        else:
                            maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
                            boxIOU = maskiou

                    if (abs(x - y) <= alpha_t and boxIOU <= iou_thr): # TODO: CL only for xRy and x-y==1
                        can_not_link_iou.append((i, j))
                        continue

                    #if (abs(x - y) >alpha_t and embed_dist>=8):#kitti - car:4, ped: 8, clasp = 4
                        # TODO: check jth detection is already in CL
                       # can_not_link_iou.append((i, j))
                        #continue
                #for unassociated detection in the window (constraints in previous frame: if x=9, y=8)
                if x!=fr and boxs_feature[:, 8][i]==0:
                    if graph_metric=='box_iou':
                        boxIOU = get_iou(boxs[i], boxs[j])

                    elif graph_metric == 'mask_iou':
                        if 10 in [extrapolate_flags[i], extrapolate_flags[j]]:
                            boxIOU = get_iou(boxs[i], boxs[j])

                        else:
                            maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
                            boxIOU = maskiou

                    if (abs(x - y) <= alpha_t and boxIOU <= iou_thr):
                        can_not_link_iou.append((i, j))
                        continue

            else:# for first window in video
                if (i < j and x > y and abs(x - y) <= alpha_t ):
                    if graph_metric=='box_iou':
                        boxIOU = get_iou(boxs[i], boxs[j])

                    elif graph_metric == 'mask_iou':
                        maskiou = maskIOU(boxs_feature[i], mask[i], boxs_feature[j], mask[j], im_h, im_w)
                        boxIOU = maskiou

                    if boxIOU <= iou_thr:
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