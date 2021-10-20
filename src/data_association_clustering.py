from __future__ import division
import numpy as np
from collections import Counter
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.cluster import k_means
from PIL import Image
import webcolors
import cv2
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import pdb
import copy
import glob
import os
from scipy.interpolate import interp1d

# from scipy.stats import multivariate_normal

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

def one2all_iou(one,all):
    iou_vector = np.array([get_iou(i, j, epsilon=1e-5) for i in one for j in all]).reshape(one.shape[0], all.shape[0])
    return iou_vector

def pairwise_iou_non_diag(a):
    iou_matrix = np.array([get_iou(i, j, epsilon=1e-5) for i in a for j in a]).reshape(a.shape[0], a.shape[0])
    xu, yu = np.triu_indices_from(iou_matrix, k=1)
    xl, yl = np.tril_indices_from(iou_matrix, k=-1)
    x = np.concatenate((xl, xu))
    y = np.concatenate((yl, yu))
    non_diag_ious = iou_matrix[(x, y)].reshape(iou_matrix.shape[0], iou_matrix.shape[0]-1)

    return non_diag_ious


def batch_iou(a, b, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # COORDINATES OF THE INTERSECTION BOXES
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou

def better_np_unique(arr):
    sort_indexes = np.argsort(arr)
    arr = np.asarray(arr)[sort_indexes]
    vals, first_indexes, inverse, counts = np.unique(arr,
        return_index=True, return_inverse=True, return_counts=True)
    indexes = np.split(sort_indexes, first_indexes[1:])
    for x in indexes:
        x.sort()
    return vals, indexes, inverse, counts

def pair_dist_ind(X,Y):
    dist = pairwise_distances(X, Y, metric='euclidean') #X by Y size
    centroid_ind = np.sum(dist,axis=1).argmin()
    return centroid_ind

def final_det_mask(cluster_Q, cluster_i_mask, cluster_i_embed, cluster_center, time_lag):
    #
    # final tracklet member selection from the survived cluster
    #
    cluster_prob_score = np.sum(cluster_Q[:, 6]) / time_lag
    score = np.around(cluster_Q[:, 6], decimals=2)  # cluster score rounded upto two decimal points
    print('cluster score', cluster_prob_score)
    refined_det = cluster_Q[np.where(score == score.max())]  # single or multiple detections might be the representtive members of a cluster
    # refined_det = cluster_Q[np.where(cluster_Q[:,0] == cluster_Q[:,0].max())][0]
    # feature_center_ini = cluster_i_shifted_points[np.where(score == score.max())]
    refined_mask = cluster_i_mask[np.where(score == score.max())]
    refined_embed = cluster_i_embed[np.where(score == score.max())]
    # refined_mask = cluster_i_mask[np.where(cluster_Q[:,0] == cluster_Q[:,0].max())][0]
    # refined_embed = cluster_i_embed[np.where(cluster_Q[:,0] == cluster_Q[:,0].max())][0]
    if (len(refined_det) > 1):
        centroid_ind = pair_dist_ind(refined_embed, cluster_center)
        refined_det = refined_det[centroid_ind]  # select closest one from multiple representative
        refined_mask = refined_mask[centroid_ind, :, :]
        refined_det[6] = cluster_prob_score  # * refined_det[7]  # det score weighted by cluster probability
    else:
        refined_det = refined_det.flatten()
        refined_det[6] = cluster_prob_score  # * refined_det[7] # det score weighted by cluster probability

    return refined_det, refined_mask

def expand_from_tracklet_list(box_all):
    box_list = [b for b in box_all if len(b) > 0]
    box_all = np.concatenate(box_list)
    return box_all

def Coarse2ImMask(box,final_mask,im_shape):
    box_coord = box[2:6].astype(int)
    x_0 = max(box_coord[0], 0)
    x_1 = min(box_coord[0] + box_coord[2], im_shape[1])
    y_0 = max(box_coord[1], 0)
    y_1 = min(box_coord[1] + box_coord[3], im_shape[0])
    if final_mask.dtype != 'uint8':
        final_mask = final_mask.astype('uint8')

    mask = cv2.resize(final_mask, (box_coord[2], box_coord[3]))
    # apply theshold on scoremap!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!cfg.MRCNN.THRESH_BINARIZE
    mask = np.array(mask >= 0.1, dtype=np.uint8)
    im_mask = np.zeros(im_shape, dtype=np.uint8)
    # mask transfer on image cooordinate
    im_mask[y_0:y_1, x_0:x_1] = mask
    rle = maskUtils.encode(np.asfortranarray(im_mask))
    #im_mask[y_0:y_1, x_0:x_1] = mask[
     #                           (y_0 - box_coord[1]):(y_1 - box_coord[1]),
      #                          (x_0 - box_coord[0]):(x_1 - box_coord[0])
       #                       ]
    return rle


def plot_cluster(cluster_Q, names, colors, posXY):
    img = cv2.imread(names)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxs = np.transpose(np.array(
        [cluster_Q[:, 2], cluster_Q[:, 3], cluster_Q[:, 2] + cluster_Q[:, 4],
         cluster_Q[:, 3] + cluster_Q[:, 5]]))

    for i in range(boxs.shape[0]):
        color = colors[int(cluster_Q[i, 8])]
        img = cv2.rectangle(img, (int(boxs[i, 0]), int(boxs[i, 1])), \
                            (int(boxs[i, 2]),
                             int(boxs[i, 3])),
                            webcolors.hex_to_rgb(color) , 4)
        cv2.putText(img, '{}'.format(int(cluster_Q[i, 0])),
                    (int(posXY[0]), int(posXY[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, webcolors.hex_to_rgb(color), 2, cv2.LINE_AA)
        posXY[1]+=50
    return img

def association_constraints_det_others(fr,X_frames,latent_feature, det_frame,
                                    labels,cluster_center,
                                    ID_ind, time_lag,score_th,min_cluster_size,
                                   names,color):
    final_det = []
    labels_unique, counts = np.unique(labels, return_counts=True)
    print('Number of estimated cluster',len(labels_unique))

    #img = cv2.imread(names)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (fr == 16000):
        img = plot_cluster(det_frame, names, color)
        im = Image.fromarray(img)
        im.save('/media/siddique/Data/CLASP2018/img/MOT/MOT17Det/train/MOT17-13/'
                'box_mask_overlaid/problems/cluster_on_image/' + 'cluster' + str(fr) + '_' + names[-10:])

    for j in labels_unique: #cluster wise loop
        # filter noise by selecting single detection foe a cluster at an angle
        # ignore repeated angle sample in det_frame
        if j == -11:
            det_frame[np.where(labels == j), :][0][:, 8] = 0  # outlier sample remain unassociated
        if j>=0:
            track_new = 0
            if(fr==291):
              print('debug')
            cluster_Q = det_frame[np.where(labels == j), :][0] # all angles of Fa cluster
            cluster_i_embed = latent_feature[np.where(labels == j), :][0]
            '''
            for i in range(len(labels_unique)):
                    cluster_i_embed1 = latent_feature[np.where(labels == 0), :][0]
                    cluster_i_embed2 = latent_feature[np.where(labels == i), :][0]
                    print(pairwise_distances(cluster_i_embed1,cluster_i_embed2, metric='l2'))      
            '''
            cluster_label = cluster_Q[:,8]
            print('Cluster Label',cluster_label)
            cluster_prob_score = np.sum(cluster_Q[:, 6]) / time_lag
            cluster_size = cluster_Q[:, 8].shape[0]
            print('Frame Pattern',cluster_Q[:,0])
            if(fr==16000):
                img = plot_cluster(cluster_Q, names, color[j])
                im = Image.fromarray(img)
                im.save('/media/siddique/Data/CLASP2018/img/MOT/MOT17Det/train/MOT17-13/'
                        'box_mask_overlaid/problems/cluster_on_image/' 'cluster'+str(j)+'_'+ names[-10:])

            # Case1: All samples are unassociated [cluster size <= temporal window]
            # TODO: Instead of sample IOU comparison, use frame pattern as decision metrics for cluster quality test
            boxs = np.transpose(np.array(
                [cluster_Q[:, 2], cluster_Q[:, 3], cluster_Q[:, 2] + cluster_Q[:, 4], cluster_Q[:, 3] + cluster_Q[:, 5]]))
            iou_matrix_non_diag = pairwise_iou_non_diag(boxs)
            avg_iou = np.mean(iou_matrix_non_diag, axis=1)
            uni, cnt = np.unique(cluster_label, return_counts=True)
            # [0,0,0,0,0] :
            if (sum(cluster_Q[:, 8]) == 0 and cluster_size <= time_lag and cluster_size>=min_cluster_size):
               # All cluster ssamples are unassociated (zero id)
               print('average_IoU of cluster samples:',avg_iou)
               cluster_Q[:, 8] = ID_ind  # initialize all detections in a cluster with cluster ID
               # ID assignment to new sample using temporal differences
               #cluster_Q = temporal_diff_test(cluster_Q,cluster_i_embed, cluster_center[j], ID_ind)
               det_frame[np.where(labels == j), :] = cluster_Q
               print('new trajectory', ID_ind)
               print('new trajectory labels after association', det_frame[np.where(labels == j), 8])
               #print('Frame Pattern',cluster_Q[:, 0])
               ID_ind += 1
               track_new = 1

               refined_det = cluster_Q[np.where(cluster_Q[:,0]==cluster_Q[:,0].max())][0]
               refined_det[6] = np.sum(cluster_Q[:, 6]) / time_lag
               final_det.append(refined_det)
            #
            # Cluster contain both associated and unassociated samples > [6,6,6,6,0] or [6 6 0 0 0] or [6 1 6 0 0]
            # what about [6,6,6,6,6]; new cluster might have zero id sample or not
            # Use cluster_new = [0,1] (in all zero case) identifier to consider new cluster with no zero id
            #len(cluster_label[np.where(avg_iou>=0.2)])>=min_cluster_size and
            elif (len(cluster_label[np.where(cluster_label==0)]) > 0 and track_new==0 and
                  cluster_size>=min_cluster_size and cluster_size<=time_lag):
                  #and len(cluster_label[np.where(avg_iou>=0.1)])>=min_cluster_size):#cluster_prob_score >= score_th
                #[6 6 0 0 0] ????
                if(len(cluster_label[np.where(cluster_label==0)])>=min_cluster_size
                        and uni[np.where(cnt == cnt.max())].max()==0):# and sum(cluster_Q[:, 6])/time_lag >=score_th):
                    # Initiate new object trajectroy when all IDs are zero with cluster score>=0.5
                    print('average_IoU of cluster samples:', avg_iou)
                    #cluster_label[np.where(avg_iou >= 0.1)] = ID_ind  # cluster_label[np.where(avg_iou == avg_iou.max())][0] uni[np.argmax(cnt)]
                    cluster_label = ID_ind
                    print('new trajectory labels before association', cluster_Q[:,8])
                    #cluster_label[::]=ID_ind
                    print('new trajectory', ID_ind)
                    print('new trajectory labels after association', cluster_label)
                    ID_ind += 1
                    track_new == 1

                # [6 6 6 0 0]?? or #[6 6 6 6 0]-yes
                else:
                    non_zero_id = cluster_label[cluster_label.nonzero()]
                    uni, cnt = np.unique(non_zero_id, return_counts=True)
                    print('average_IoU of cluster samples:', avg_iou)
                    #apply majority voting during association of new samples with existing labeled samples
                    voted_id = uni[np.where(cnt == cnt.max())].max()
                    cluster_label = voted_id
                    print('labels after association', cluster_label)
                    track_new == 1

                cluster_Q[:, 8] = cluster_label
                # TODO: obtain final mask and det from the last corner of the temporal window
                # TODO: use average Jaccard index to get final det and mask
                refined_det = cluster_Q[np.where(cluster_Q[:, 0] == cluster_Q[:, 0].max())][0]
                if(cluster_Q[:,8][-1]==0):
                    refined_det[8] = cluster_Q[:,8].max()
                refined_det[6] = np.sum(cluster_Q[:, 6]) / time_lag
                final_det.append(refined_det)

                det_frame[np.where(labels == j), :] = cluster_Q

            # [6 6], [6, 6, 6] ... But these are not new track or already associated track
            elif(track_new == 0 and cluster_size>=min_cluster_size
                    and cluster_size<=time_lag):
                # TODO: obtain final mask and det from the last corner of the temporal window
                refined_det = cluster_Q[np.where(cluster_Q[:,0]==cluster_Q[:,0].max())][0]
                refined_det[6] = np.sum(cluster_Q[:, 6]) / time_lag
                final_det.append(refined_det)

                print('No New Association:', cluster_Q[:, 8])
                print('Frame Pattern:', cluster_Q[:, 0])
                print('average_IoU of cluster samples:', avg_iou)
                det_frame[np.where(labels == j), :] = cluster_Q

    return np.array(final_det), det_frame, ID_ind

def majority_voted_id(cluster_Q,cluster_i_mask,cluster_label):
    #----------------------------------------------------------
    # Majority voting applied on labeled targets in a cluster to propagate the label to the new associated detection
    #
    #
    #    Return -- Voted detection and mask
    #--------------------------------------------------------
    non_zero_id = cluster_label[cluster_label.nonzero()]
    uni, cnt = np.unique(non_zero_id, return_counts=True)
    voted_id = int(uni[np.where(cnt == cnt.max())].max())
    if cluster_Q[:,8][0]==0:
        cluster_Q[:,8][0]=voted_id
        cluster_label[0]=voted_id
    det_voted = cluster_Q[cluster_label == voted_id]
    mask_voted = cluster_i_mask[cluster_label == voted_id]
    refined_mask = mask_voted[np.where(det_voted[:, 0] == det_voted[:, 0].max())][0]

    refined_det = det_voted[np.where(det_voted[:, 0] == det_voted[:, 0].max())][0]

    return refined_det, refined_mask, voted_id, cluster_label, cluster_Q

def majority_voted_id_mnist(cluster_Q,cluster_i_mask,cluster_label):
    #----------------------------------------------------------
    # Majority voting applied on labeled targets in a cluster to propagate the label to the new associated detection
    #
    #
    #    Return -- Voted detection and mask
    #--------------------------------------------------------
    non_zero_id = cluster_label[cluster_label.nonzero()]
    uni, cnt = np.unique(non_zero_id, return_counts=True)
    voted_id = int(uni[np.where(cnt == cnt.max())].max())
    if cluster_Q[:,-1][0]==0:
        cluster_Q[:,-1][0]=voted_id
        cluster_label[0]=voted_id
    return voted_id

def majority_voted_id_patch(cluster_Q,cluster_i_mask,cluster_i_patch,cluster_label):
    #----------------------------------------------------------
    # Majority voting applied on labeled targets in a cluster to propagate the label to the new associated detection
    #
    #
    #    Return -- Voted detection and mask
    #--------------------------------------------------------
    non_zero_id = cluster_label[cluster_label.nonzero()]
    uni, cnt = np.unique(non_zero_id, return_counts=True)
    voted_id = int(uni[np.where(cnt == cnt.max())].max())
    if cluster_Q[:, 8][0] == 0:
        cluster_Q[:,8][0]=voted_id
        cluster_label[0] = voted_id
    det_voted = cluster_Q[cluster_label == voted_id]
    mask_voted = cluster_i_mask[cluster_label == voted_id]
    patch_voted = cluster_i_patch[cluster_label == voted_id]
    refined_mask = mask_voted[np.where(det_voted[:, 0] == det_voted[:, 0].max())][0]
    refined_patch = patch_voted[np.where(det_voted[:, 0] == det_voted[:, 0].max())][0]

    refined_det = det_voted[np.where(det_voted[:, 0] == det_voted[:, 0].max())][0]

    return refined_det, refined_mask, refined_patch, voted_id, cluster_label, cluster_Q

def check_lost_track(fr,lost_counter,refined_det, refined_mask,cluster_label):
    #-----------------------------------------------------------------------
    # Setup termination rule  using dummy observation from previous frame
    # until t_lag
    #      Arguement-- lost_track = { {'ID':{},'dummy_count':{}}, {...}, ... }
    #
    #-----------------------------------------------------------------------
    if fr>refined_det[0]:
        lost_counter[refined_det[8]]+=1
    return lost_counter

def estimate_mmotion(box_xs):
    x_diff = []
    for i, x in enumerate(box_xs):
        try:
            x_diff.append(box_xs[i] - box_xs[i + 1])
        except:
            print('done')
    v_t = (1/len(box_xs))*sum(x_diff)
    return v_t

def linearly_interpolate(cluster_Q, fr):
    vxt = estimate_mmotion(cluster_Q[:,2])
    vyt = estimate_mmotion(cluster_Q[:, 3])
    vwt = estimate_mmotion(cluster_Q[:, 4])
    vht = estimate_mmotion(cluster_Q[:, 5])
    xt1 = max(0,cluster_Q[0, 2] + vxt)
    yt1 = max(0,cluster_Q[0, 3] + vyt)
    wt1 = max(0,cluster_Q[0, 4] + vwt)
    ht1 = max(0,cluster_Q[0, 5] + vht)
    interpolated_trklt_head = cluster_Q[0,:]
    interpolated_trklt_head[2:6] = xt1, yt1, wt1, ht1
    interpolated_trklt_head[0] = fr
    return interpolated_trklt_head

def interpolate(tracks,fr,classID=3):
    #collected from treacktor repo
    #interpolate left tracks for 30 frames to associate with immediately associated tracks
    #from scipy import interpolate

    #x = np.arange(0,10)
    #y = np.exp(-x/3.0)
    #f = interpolate.interp1d(x, y, fill_value='extrapolate')

    #print f(9)
    #print f(11)
    interpolated = []
    frames = []
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for bb in tracks:
        frames.append(bb[0])
        x0.append(bb[2])
        y0.append(bb[3])
        x1.append(bb[2]+bb[4])
        y1.append(bb[3]+bb[5])

    x0_inter = interp1d(frames, x0, fill_value='extrapolate')
    y0_inter = interp1d(frames, y0, fill_value='extrapolate')
    x1_inter = interp1d(frames, x1, fill_value='extrapolate')
    y1_inter = interp1d(frames, y1, fill_value='extrapolate')
    if fr not in frames:
        for f in range(int(max(frames))+1, fr+1):# predict tracklet head next frame to current frame
            x1 = max(0,x0_inter(f))
            y1 = max(0,y0_inter(f))

            x2 = min(bb[8], x1_inter(f))
            y2 = min(bb[7], y1_inter(f))
            w = x2 - x1
            h = y2 - y1
            if 0<w<bb[8] and 0<h<bb[7]:
                assert w>0 and h>0, 'found w {} h {}'.format(w,h)
                interpolated.append(np.array([f, -1, x1, y1, w, h, bb[6], classID, bb[1]]))
    else:
        assert fr in frames, 'interpolate only when track fails to find new detection at current frame'
    return np.array(interpolated[::-1])

def delete_all(demo_path, fmt='png'):
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)

def align_centroids(fr, cluster_Q):
    if fr in cluster_Q[:,0] and len(cluster_Q[:,0])>1:
        cluster_boxs = copy.deepcopy(cluster_Q)
        fr_cxcy = cluster_boxs[cluster_boxs[:,0]==fr][0][2:4] + cluster_boxs[cluster_boxs[:,0]==fr][0][4:6]/2.
        cluster_boxs[:,2] = fr_cxcy[0] - cluster_boxs[:,4]/2.
        cluster_boxs[:,3] = fr_cxcy[1] - cluster_boxs[:,5]/2.
    else:
        cluster_boxs = cluster_Q
    return cluster_boxs


def cluster_association(fr, X_frames, latent_feature, det_frame, det_at_t_pmask,
                        labels, ID_ind, time_lag, score_th, min_cluster_size,
                        k_value, names, color, mask_y, mask_x, im_shape, trackers=None,
                        vis_pred=False, verbose=False, lost_tracks=None, vis=False,
                        apply_reid=False):
    #---------------------------------------------------------------------------
    # Used in: TCT_KITTI
    # trackers - key: ID, value: (fr,x,y,w,h,score,im_h,im_w,rle_counts)
    # det_at_t_pmask: 28*28 binary object mask
    # im_shape: (h,w)
    #---------------------------------------------------------------------------
    final_det = []
    final_mask = []
    dets_align_centers = copy.deepcopy(det_frame)
    labels_unique, counts = np.unique(labels, return_counts=True)

    if verbose: print('unique clusters {}'.format(labels_unique))
    for j in labels_unique: #cluster wise loop
        cluster_label = det_frame[np.where(labels == j), :][0][:, 8]
        cluster_Q = det_frame[np.where(labels == j), :][0]
        cluster_i_embed = latent_feature[np.where(labels == j), :][0]
        cluster_i_mask = det_at_t_pmask[labels == j]

        if j>=0:
            uni, cnt = np.unique(cluster_label, return_counts=True)
            tracker_update = 0
            cluster_label = cluster_Q[:,8]
            if verbose: print('Cluster Label',cluster_label)
            cluster_prob_score = np.sum(cluster_Q[:, 6]) / time_lag
            cluster_size = cluster_Q[:, 8].shape[0]
            if verbose: print('Frame Pattern',cluster_Q[:,0])
            if vis:
                posXY = [50, 50]
                img = plot_cluster(cluster_Q, names, color, posXY)
                im = Image.fromarray(img)
                im.save(imgPath1 + str(j) + '_' + names[-10:])
            # Case1: All samples are unassociated [cluster size <= temporal window]
            # [0,0,0,0,0] :
            if (sum(cluster_Q[:, 8]) == 0  and cluster_size>=min_cluster_size
                    and tracker_update == 0 and cluster_prob_score>=score_th):#and cluster_size <= time_lag

               cluster_Q[:, 8] = ID_ind  # initialize all detections in a cluster with cluster ID
               if verbose: print('new trajectory', ID_ind)
               if verbose: print('new trajectory labels after association', cluster_Q[:, 8])
               #print('Frame Pattern',cluster_Q[:, 0])
               voted_id = int(ID_ind)
               #refined_det = cluster_Q[np.where(cluster_Q[:,0]==cluster_Q[:,0].max())][0]
               #refined_mask = cluster_i_mask[np.where(cluster_Q[:, 0] == cluster_Q[:, 0].max())][0]
               #final_mask.append(refined_mask.reshape(mask_x,mask_y))
               #refined_det[6] = np.sum(cluster_Q[:, 6]) / time_lag

               #TODO: previous frame target considered at current frame? or predict dummy? update tracks if only found det at current fr??
               #refined_det[0] = fr
               #final_det.append(refined_det)
               det_frame[np.where(labels == j), :] = cluster_Q
               #trackers[str(voted_id-1)] = refined_det
               #create new set for new tracker
               trackers[str(voted_id)] = set()
               # save all initialized tracklets box+rle
               for ind,tracker in enumerate(cluster_Q):
                    #rle = Coarse2ImMask(tracker, cluster_i_mask[ind], im_shape)
                    rle = {'size': [im_shape[0], im_shape[1]], 'counts': bytes(cluster_i_mask[ind], 'utf-8')}
                    trackers[str(voted_id)].add((tracker[0], voted_id, tracker[2], tracker[3], tracker[4], tracker[5],tracker[6],rle['size'][0],rle['size'][1],rle['counts']))
               tracker_update = 1
               ID_ind += 1
            # Cluster contain both associated and unassociated samples > [6,6,6,6,0] or [6 6 0 0 0] or [6 1 6 0 0]
            # what about [6,6,6,6,6]; must create dummy since cluster does not found any association with current frame
            #len(cluster_label[np.where(avg_iou>=0.2)])>=min_cluster_size and
            elif (len(cluster_label[np.where(cluster_label==0)]) > 0 and  tracker_update==0 and
                  cluster_size>=min_cluster_size and cluster_prob_score>=score_th):# and cluster_size<=time_lag
                # before starting label propagation we need to verify some rules
                # [6 6 7 8 0 0] ????: different id should not be associated in same subspace
                #assert len(np.unique(cluster_label[np.where(cluster_label > 0)])) == 1, \
                    #'A cluster should not contain multiple ids but found in {}'.format(
                        #cluster_label)
                #if len(np.unique(cluster_label[np.where(cluster_label > 0)])) > 1:
                    #continue
                #TODO: check cluster_size<=time_lag
                  #and len(cluster_label[np.where(avg_iou>=0.1)])>=min_cluster_size):#cluster_prob_score >= score_th
                #[6 6 0 0 0] ????
                #refine cluster when two new detection in the same cluster: mask-iou
                if(len(cluster_label[np.where(cluster_label==0)])>=min_cluster_size
                        and uni[np.where(cnt == cnt.max())].max()==0) and cluster_prob_score>=score_th:
                    cluster_Q_new = cluster_Q[cluster_label==0]
                    cluster_i_mask_new = cluster_i_mask[cluster_label==0]
                    #cluster_label[np.where(avg_iou >= 0.1)] = ID_ind  # cluster_label[np.where(avg_iou == avg_iou.max())][0] uni[np.argmax(cnt)]
                    #TODO: Why two different ID associate in same cluster
                    #voted_id = int(ID_ind)
                    voted_id =  int(cluster_label.max()) # use the label of immediate prev frame
                    cluster_label[np.where(cluster_label==0)] = voted_id
                    cluster_Q[:, 8] = cluster_label

                    if verbose: print('new detections labels before association', cluster_Q_new[:, 8])
                    cluster_Q_new[:,8] = voted_id
                    if verbose: print('new detections labels after association',  cluster_Q_new[:,8])


                    refined_det = cluster_Q[np.where(cluster_Q[:, 0] == cluster_Q[:, 0].max())][0]
                    refined_mask = cluster_i_mask[np.where(cluster_Q[:, 0] == cluster_Q[:, 0].max())][0]

                    # TODO: previous frame target considered at current frame? or predict dummy?
                    # save all initialized tracklets box+rle
                    # TODO: tracker should not have the duplicate item: clustering failure
                    # verify trackers[str(voted_id)] has already the item in new cluster: repeated frame
                    for ind, tracker in enumerate(cluster_Q_new): #cluster_Q
                        #rle = Coarse2ImMask(tracker, cluster_i_mask_new[ind], im_shape)
                        rle = {'size': [im_shape[0], im_shape[1]], 'counts': bytes(cluster_i_mask_new[ind], 'utf-8')}
                        trackers[str(int(voted_id))].add((tracker[0],voted_id, tracker[2], tracker[3], tracker[4], tracker[5],
                                                     tracker[6], rle['size'][0], rle['size'][1], rle['counts']))
                #[6 6 6 6 0]-yes
                else:
                    #if len(np.unique(cluster_label[np.where(cluster_label > 0)])) > 1:
                        #continue
                    #assert len(np.unique(cluster_label[np.where(cluster_label > 0)])) == 1, \
                        #'A cluster should not contain multiple ids but found in {}'.format(
                           # cluster_label)

                    #assert len(cluster_label[np.where(cluster_label == 0)]) == 1, \
                        #'two new detections association in a same cluster is not allowed but found in {}'.format(
                           # cluster_label.max())

                    #***TODO: For two equal voted id, why we select maximum instead of minimum?
                    refined_det, refined_mask, voted_id, cluster_label, cluster_Q = majority_voted_id(cluster_Q, cluster_i_mask, cluster_label)
                    # TODO: sanity check so that unlabelled target from immediate previous frame are assigned from voted label
                    #cluster_label[np.where(cluster_label==0)] = voted_id
                    if verbose: print('labels before association', cluster_Q[:,8])
                    if verbose: print('labels after association', cluster_label)
                    #refined_det[0] = fr
                    refined_det[6] = np.sum(cluster_Q[:, 6]) / time_lag
                    # save all initialized tracklets box+rle
                    tracker = refined_det
                    #rle = Coarse2ImMask(tracker, refined_mask, im_shape)
                    rle = {'size': [im_shape[0], im_shape[1]], 'counts': bytes(refined_mask, 'utf-8')}
                    trackers[str(int(voted_id))].add((tracker[0], voted_id, tracker[2], tracker[3], tracker[4], tracker[5],
                                                 tracker[6], rle['size'][0], rle['size'][1], rle['counts']))
                cluster_Q[:, 8] = cluster_label
                # TODO: obtain final mask and det from the last corner of the temporal window
                # TODO: Update frame id of the tracklet heads to fillup the missed detection gap
                #final_mask.append(refined_mask.reshape(mask_x,mask_y))
                refined_det[6] = np.sum(cluster_Q[:, 6]) / time_lag
                final_det.append(refined_det)

                det_frame[np.where(labels == j), :] = cluster_Q
                tracker_update = 1
                #img = plot_cluster(cluster_Q, img, color)
            # [6], [6 6], [6, 6, 6] ... But these are not new track or already associated track
            # Todo: update frame index (time stamp) for occluded or to be left targets until cluster satisfy the minimum size criterion
            elif( 0 not in cluster_label  and  tracker_update==0): #cluster_size>=min_cluster_size, and cluster_size<=time_lag
                #if len(np.unique(cluster_label[np.where(cluster_label > 0)])) > 1:
                    #continue
                #assert len(np.unique(cluster_label[np.where(cluster_label > 0)])) == 1, \
                    #'A cluster should not contain multiple ids but found in {}'.format(
                        #cluster_label)
                # TODO: apply voting instead of greedy choice of maximum labeled target in a cluster
                refined_det, refined_mask, voted_id, cluster_label, cluster_Q = majority_voted_id(cluster_Q, cluster_i_mask, cluster_label)
                #final_mask.append(refined_mask.reshape(mask_x,mask_y))
                refined_det[6] = np.sum(cluster_Q[:, 6]) / time_lag
                #refined_det[0] = fr
                final_det.append(refined_det)

                if verbose: print('No New Association:', cluster_Q[:, 8])
                if verbose: print('Frame Pattern:', cluster_Q[:, 0])
                # save all initialized tracklets box+rle
                tracker = refined_det
                rle = {'size': [im_shape[0], im_shape[1]], 'counts': bytes(refined_mask, 'utf-8')}

                #rle = Coarse2ImMask(tracker, refined_mask, im_shape)
                #if str(int(voted_id)) not in trackers.keys():
                    #trackers[str(int(voted_id))].add((tracker[0], voted_id, tracker[2], tracker[3], tracker[4], tracker[5],
                                                 #tracker[6], rle['size'][0], rle['size'][1], rle['counts']))

                if fr not in cluster_Q[:,0] and apply_reid and len(cluster_Q)<=2:
                    if int(voted_id) not in lost_tracks:
                        print('id {} is lost at frame {}'.format(voted_id, fr))
                        lost_tracks[int(voted_id)] = {'reid_patience':1,
                                                 'last_pose':[tracker[0], voted_id, tracker[2], tracker[3], tracker[4], tracker[5],
                                                 tracker[6], rle['size'][0], rle['size'][1], rle['counts']],
                                                 'lost_track':trackers[str(int(voted_id))]}
                    #else:
                       # lost_tracks[voted_id]['reid_patience']+=1
                        #if lost_tracks[voted_id]['reid_patience']>10:
                            #del ost_tracks[voted_id]

                det_frame[np.where(labels == j), :] = cluster_Q
                tracker_update=1
            # TODO: terminate tracklets
            if j==-1:
                pdb.set_trace()
                det_frame[np.where(labels == j), :] = cluster_Q
    return np.array(final_det), final_mask, det_frame, ID_ind, trackers #, lost_tracks


def cluster_constraints_dynamicK(fr,latent_feature, det_frame,det_at_t_pmask,det_at_t_pmask_28,
                                 labels, cluster_center,t_lag,minimum_cluster_size,score_th_det,names,
                                 mask_y,mask_x,color,verbose=False):
    #----------------------------------------------------------------------
    # Final mask and detections using constrained clusters from cop-kmeans
    #----------------------------------------------------------------------
    final_det = []
    final_mask = []
    final_mask_28 = []
    labels_unique, counts = np.unique(labels, return_counts=True)
    if verbose:
        print('Number of estimated cluster',len(labels_unique))
    for j in labels_unique: #cluster wise loop
        cluster_Q = det_frame[np.where(labels == j), :][0] # all angles of Fa cluster
        cluster_i_embed = latent_feature[np.where(labels == j), :][0]
        cluster_i_mask = det_at_t_pmask[np.where(labels == j), :, :][0]
        cluster_i_mask_28 = det_at_t_pmask_28[np.where(labels == j), :, :][0]
        cluster_label = labels[labels==j]
        cluster_size = counts[j]
        cluster_Q[:,8]=cluster_label
        if verbose:
            print('Cluster Labels',cluster_label)
        #boxs = np.transpose(np.array(
            #[cluster_Q[:, 2], cluster_Q[:, 3], cluster_Q[:, 2] + cluster_Q[:, 4], cluster_Q[:, 3] + cluster_Q[:, 5]]))
        #iou_matrix_non_diag = pairwise_iou_non_diag(boxs)
        #avg_iou = np.mean(iou_matrix_non_diag, axis=1)
        #uni, cnt = np.unique(cluster_label, return_counts=True)
        # TODO: refine cluster based on bbox centroid distance in the image
        # cluster representative selection
        score_sum_cluster = np.sum(cluster_Q[:, 6])
        cluster_prob_score = score_sum_cluster / t_lag
        score = np.around(cluster_Q[:, 6], decimals=2)  # cluster score rounded upto two decimal points score==score.max()
        # feature_center_ini = cluster_i_shifted_points[np.where(score == score.max())]
        if cluster_prob_score>=score_th_det:
            if cluster_label[0]==100:
                refined_det = cluster_Q[np.where(cluster_label == 0)]
                refined_mask = cluster_i_mask[np.where(cluster_label == 0)]
                refined_mask_28 = cluster_i_mask_28[np.where(cluster_label == 0)]
                refined_embed = cluster_i_embed[np.where(cluster_label == 0)]
            else:
                refined_det = cluster_Q[np.where(cluster_Q[:, 0] == cluster_Q[:, 0].max())]#cluster_Q[:, 0] == cluster_Q[:, 0].max()
                refined_mask = cluster_i_mask[np.where(cluster_Q[:, 0] == cluster_Q[:, 0].max())]  # single or multiple detections might be the representtive members of a cluster
                refined_mask_28 = cluster_i_mask_28[np.where(cluster_Q[:, 0] == cluster_Q[:, 0].max())]
                refined_embed = cluster_i_embed[np.where(cluster_Q[:, 0] == cluster_Q[:, 0].max())]
            if (len(refined_det) > 1):
                centroid_ind = pair_dist_ind(refined_embed, cluster_center)
                refined_det = refined_det[centroid_ind]  # select closest one from multiple representative
                refined_mask = refined_mask[centroid_ind, :, :]
                refined_mask_28 = refined_mask_28[centroid_ind, :, :]
                refined_det[6] = cluster_prob_score  # * refined_det[7]  # det score weighted by cluster probability
            else:
                refined_det = refined_det.flatten()
                refined_det[6] = cluster_prob_score  # * refined_det[7] # det score weighted by cluster probability
            final_mask_28.append(refined_mask_28.reshape(mask_y,mask_x))
            final_mask.append(refined_mask.reshape(128,128))
            final_det.append(refined_det)
    return np.array(final_det), np.array(final_mask), np.array(final_mask_28)


def expand_from_temporal_list(box_all=None, mask_all=None):
    if box_all is not None:
        box_list = [b for b in box_all if len(b) > 0]
        box_all = np.concatenate(box_list)
    if mask_all is not None:
        mask_list = [m for m in mask_all if len(m) > 0]
        masks_all = np.concatenate(mask_list)
    else:
        masks_all =[]
    return box_all, masks_all


def cluster_association_synthetic(fr,latent_feature, det_frame, det_at_t_pmask,
                                labels, cluster_center,ID_ind, time_lag,score_th,min_cluster_size,
                                k_value,names,color,mask_y,mask_x, trackers, vis_pred=False,vis=False):
    #---------------------------------------------------------------------------
    # Used in: TCT_mnist
    # trackers - key: ID, value: (fr,x,y,w,h,score,im_h,im_w,rle_counts)
    # det_at_t_pmask: 28*28 binary object mask
    #
    #---------------------------------------------------------------------------
    final_det = []
    final_mask = []
    labels_unique, counts = np.unique(labels, return_counts=True)


    print('unique clusters {}'.format(labels_unique))
    if fr==1602:
        print('debug')
    for j in labels_unique: #cluster wise loop
        cluster_Q = det_frame[labels == j]
        cluster_i_mask = det_at_t_pmask[labels == j]

        if j>=0:
            tracker_update = 0
            cluster_label = cluster_Q[:,-1]
            uni, cnt = np.unique(cluster_label, return_counts=True)
            print('Cluster Label',cluster_label)
            cluster_prob_score = len(cluster_Q) / time_lag
            cluster_size = len(cluster_Q)
            print('Frame Pattern',cluster_Q[:,0])
            if vis:
                posXY = [50, 50]
                img = plot_cluster(cluster_Q, names, color, posXY)
                im = Image.fromarray(img)
                #im.save(imgPath1 + str(j) + '_' + names[-10:])
            # Case1: All samples are unassociated [cluster size <= temporal window]
            # [0,0,0,0,0] :
            if (sum(cluster_Q[:, -1]) == 0  and cluster_size>=min_cluster_size and tracker_update == 0):#and cluster_size <= time_lag

               cluster_Q[:, -1] = ID_ind  # initialize all detections in a cluster with cluster ID
               print('new trajectory', ID_ind)
               print('new trajectory labels after association', cluster_Q[:, -1])
               #print('Frame Pattern',cluster_Q[:, 0])
               voted_id = int(ID_ind)
               refined_det = cluster_Q[np.where(cluster_Q[:,0]==cluster_Q[:,0].max())][0]
               refined_mask = cluster_i_mask[np.where(cluster_Q[:, 0] == cluster_Q[:, 0].max())][0]
               final_mask.append(refined_mask.reshape(mask_x,mask_y))
               #TODO: previous frame target considered at current frame? or predict dummy? update tracks if only found det at current fr??
               #refined_det[0] = fr
               final_det.append(refined_det)
               det_frame[np.where(labels == j), :] = cluster_Q
               #create new set for new tracker
               trackers[str(voted_id)] = set()
               # save all initialized tracklets box+rle
               for ind,tracker in enumerate(cluster_Q):
                    trackers[str(voted_id)].add((tracker[0], voted_id, tracker[2], tracker[3], tracker[4], tracker[5], cluster_prob_score, -1, -1, -1))
               tracker_update = 1
               ID_ind += 1
            # Cluster contain both associated and unassociated samples > [6,6,6,6,0] or [6 6 0 0 0] or [6 1 6 0 0]
            elif (len(cluster_label[np.where(cluster_label==0)]) > 0 and  tracker_update==0 and
                  cluster_size>=min_cluster_size):# and cluster_size<=time_lag
                # before starting label propagation we need to verify some rules
                # [6 6 7 8 0 0] ????: different id should not be associated in same subspace
                #assert len(np.unique(cluster_label[np.where(cluster_label > 0)])) == 1, \
                    #'A cluster should not contain multiple ids but found in {}'.format(
                        #cluster_label)
                #if len(np.unique(cluster_label[np.where(cluster_label > 0)])) > 1:
                    #continue
                #[6 6 0 0 0] ????
                #refine cluster when two new detection in the same cluster: mask-iou
                if(len(cluster_label[np.where(cluster_label==0)])>=min_cluster_size
                        and uni[np.where(cnt == cnt.max())].max()==0):
                    cluster_Q_new = cluster_Q[cluster_label==0]
                    cluster_i_mask_new = cluster_i_mask[cluster_label==0]
                    #TODO: apply new trackelt ID to uninitialized detections
                    voted_id =  int(cluster_label.max())
                    cluster_label[np.where(cluster_label==0)] = voted_id
                    cluster_Q[:, -1] = cluster_label

                    print('new detections labels before association', cluster_Q_new[:, -1])
                    cluster_Q_new[:,-1] = voted_id
                    print('new detections labels after association',  cluster_Q_new[:,-1])

                    refined_det = cluster_Q_new[cluster_Q_new[:, 0] == cluster_Q_new[:, 0].max()][0]
                    #refined_det[0] = fr
                    refined_mask = cluster_i_mask_new[cluster_Q_new[:, 0] == cluster_Q_new[:, 0].max()][0]

                    for ind, tracker in enumerate(cluster_Q_new): #cluster_Q
                        trackers[str(voted_id)].add((tracker[0],voted_id, tracker[2], tracker[3], tracker[4], tracker[5],
                                                     cluster_prob_score, -1, -1, -1))
                #[6 6 6 6 0]-yes
                else:
                    #if len(np.unique(cluster_label[np.where(cluster_label > 0)])) > 1:
                        #continue
                    #assert len(np.unique(cluster_label[np.where(cluster_label > 0)])) == 1, \
                        #'A cluster should not contain multiple ids but found in {}'.format(
                            #cluster_label)

                    #assert len(cluster_label[np.where(cluster_label == 0)]) == 1, \
                        #'two new detections association in a same cluster is not allowed but found in {}'.format(
                            #cluster_label.max())
                    voted_id = majority_voted_id_mnist(cluster_Q, cluster_i_mask, cluster_label)
                    cluster_label[np.where(cluster_label==0)] = voted_id
                    cluster_Q[:,-1] = cluster_label
                    # TODO: sanity check so that unlabelled target from immediate previous frame are assigned from voted label
                    refined_det = cluster_Q[cluster_Q[:, 0] == cluster_Q[:, 0].max()][0]
                    #refined_det[0] = fr
                    refined_mask = cluster_i_mask[cluster_Q[:, 0] == cluster_Q[:, 0].max()][0]

                    cluster_prob_score = len(cluster_Q) / time_lag
                    # save all initialized tracklets box+rle
                    tracker = refined_det
                    trackers[str(voted_id)].add((tracker[0], voted_id, tracker[2], tracker[3], tracker[4], tracker[5],
                                                 cluster_prob_score, -1, -1, -1))
                # TODO: obtain final mask and det from the last corner of the temporal window
                # TODO: Update frame id of the tracklet heads to fillup the missed detection gap
                final_mask.append(refined_mask.reshape(mask_x,mask_y))
                final_det.append(refined_det)

                det_frame[labels == j] = cluster_Q
                tracker_update = 1
                #img = plot_cluster(cluster_Q, img, color)
            # [6], [6 6], [6, 6, 6] ... But these are not new track or already associated track
            # Todo: update frame index (time stamp) for occluded or to be left targets until cluster satisfy the minimum size criterion
            elif( 0 not in cluster_label  and  tracker_update==0): #cluster_size>=min_cluster_size, and cluster_size<=time_lag
                #if len(np.unique(cluster_label[np.where(cluster_label > 0)])) > 1:
                    #continue
                #assert len(np.unique(cluster_label[np.where(cluster_label > 0)])) == 1, \
                    #'A cluster should not contain multiple ids but found in {}'.format(
                        #cluster_label)
                voted_id = majority_voted_id_mnist(cluster_Q, cluster_i_mask, cluster_label)
                # TODO: sanity check so that unlabelled target from immediate previous frame are assigned from voted label
                refined_det_voted = cluster_Q[cluster_label == voted_id]
                refined_mask_voted = cluster_i_mask[cluster_label == voted_id]
                refined_mask = refined_mask_voted[refined_det_voted[:, 0] == refined_det_voted[:, 0].max()][0]
                refined_det = refined_det_voted[refined_det_voted[:, 0] == refined_det_voted[:, 0].max()][0]

                final_mask.append(refined_mask.reshape(mask_x,mask_y))
                cluster_prob_score = len(cluster_Q) / time_lag
                #refined_det[0] = fr
                final_det.append(refined_det)

                print('No New Association:', cluster_Q[:, -1])
                print('Frame Pattern:', cluster_Q[:, 0])
                # save all initialized tracklets box+rle
                tracker = refined_det
                if str(voted_id) not in trackers:
                    trackers[str(voted_id)] = set()
                trackers[str(voted_id)].add((tracker[0], voted_id, tracker[2], tracker[3], tracker[4], tracker[5],
                                             cluster_prob_score, -1, -1, -1))
                det_frame[labels == j, :] = cluster_Q
                tracker_update=1
            # TODO: terminate tracklets
            if j==-1:
                pdb.set_trace()
                det_frame[np.where(labels == j), :] = cluster_Q
    return final_det, final_mask, det_frame, ID_ind, trackers
