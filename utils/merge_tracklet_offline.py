import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment
import copy
import pdb
#import cv2
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def linear_sum_assignment_with_inf(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        if values.size == 0:
            cost_matrix = np.full(cost_matrix.shape,1000) #workaround for the cast of no finite costs
        else:
            m = values.min()
            M = values.max()
            n = min(cost_matrix.shape)
            # strictly positive constant even when added
            # to elements of the cost matrix
            positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
            if max_inf:
                place_holder = (M + (n - 1) * (M - m)) + positive
            if min_inf:
                place_holder = (m + (n - 1) * (m - M)) - positive

            cost_matrix[np.isinf(cost_matrix)] = place_holder
    return linear_sum_assignment(cost_matrix)

def match_tl(det1, det2, t_thr=30, d_thr=10,metric='iou'):
# Associate pairs of tracklets with maximum overlap between the last and first detections
# using the Hungarian algorithm
# det1 (fr inst_ind, x, y, w, h, score, class_id, track_id): all the confirmed track (size n)
# det2 (fr inst_ind, x, y, w, h, score, class_id, track_id): all the confirmed track(size n)
# match_tl will find the optimized candidate to propagate the temporal identifier
    new_track = []
    prev_track = []
    matches = []
    for tr in det1:
        prev_track.append(tr[-1]) # end of previously dead cluster
    for tr in det2:
        new_track.append(tr[0]) # start of newly born cluster

    # Compute the cost matrix IoU between each pair of last and first detections
    cost = np.full((len(prev_track), len(new_track)),np.inf) # cost for n to 1 association
    i = 0
    for prev_det in prev_track:
        j = 0
        for new_det in new_track:
            # This is the only difference from the single-camera function
            # TODO: Make both a common function that take this test as a parameter
            delta_t = new_det[0] - prev_det[0]
            #dist = np.linalg.norm(new_det[1:5]-prev_det[1:5]) #default
            # TODO: try embedding distance
            # compute distance using list operation
            #pdb.set_trace()
            ndet = np.asarray(new_det[2:9])
            pdet =  np.asarray(prev_det[2:9])
            if metric=='center_dist':
                new_Cxy = (ndet[0:2]+ndet[2:4]/2.0)/ndet[5:7][::-1]
                prev_Cxy = (pdet[0:2]+pdet[2:4]/2.0)/ndet[5:7][::-1]
                dist = np.linalg.norm(new_Cxy - prev_Cxy)
                if 0 < delta_t < t_thr and dist < d_thr:
                    print('found matching {} and {} at d_thr {}'.format(i,j,d_thr))
                    cost[i,j] = dist
            if metric=='iou':
                iou = bb_intersection_over_union(
                   [ndet[0], ndet[1], ndet[0] + ndet[2], ndet[1] + ndet[3]],
                    [pdet[0], pdet[1], pdet[0] + pdet[2], pdet[1] + pdet[3]])
                if 1-iou < d_thr and  0 < delta_t < t_thr:
                    # for iou=1, cost=0, for iou=0, cost=1
                    print('found matching {} and {} at d_iou {}'.format(i, j, 1-iou))
                    cost[i, j] = 1-iou
            j = j + 1
        i = i + 1

    row_ind, col_ind = linear_sum_assignment_with_inf(cost)
    # DEBUG
    #import matplotlib.pyplot as plt
    #hist = np.histogram(cost[cost < 100])
    #plt.hist(cost[cost < 100])
    # Find the maximum IoU for each pair
    for i in row_ind:
        if cost[i,col_ind[i]] < d_thr:#100:#1
            matches.append((cost[i, col_ind[i]], i, col_ind[i]))

    return matches


def merge_tracklets(filtered_tracker, unassigned_tracklets,t_thr=30, d_thr=10, metric='center_dist'):
    # Iterate a few times to join pairs of tracklets that correspond to multiple
    # fragments of the same track
    # TODO: Find a more elegant solution -- Possibly using DFS
    del_tl = []
    for merge_steps in range(0,5):
        print('merge steps {}'.format(merge_steps))
        mt = match_tl(filtered_tracker,unassigned_tracklets,t_thr, d_thr,metric)
        print(mt)

        for (c, k, l) in mt:
            print('{} merges with {} for cost {} during step {}'.format(k+1,l+1,c,merge_steps))
            filtered_tracker[k] = filtered_tracker[k] + unassigned_tracklets[l] # merge two matched tracklet list
            if l not in del_tl:
                del_tl.append(l)

    del_tl.sort(reverse=True)
    # delete associated tracklets to discard the duplicity
    for l in del_tl:
        del(filtered_tracker[l])
        del(unassigned_tracklets[l])

    return filtered_tracker, unassigned_tracklets
def get_box_mots(tracker_i):
    trackers = []
    for i in range(len(tracker_i)):
        trackers.append(tracker_i[i][0:6])
    return  np.array(trackers)

def get_tracklets(trackers):
    tracklets = []

    for tid in range(1, len(trackers)+1):
        tracker_i = trackers.get(str(tid))
        tracker_i = list(tracker_i)
        #tracker_i = get_box_mots(tracker_i)
        tracker_i = sorted(tracker_i, key=lambda x: x[0])
        tracklets.append(tracker_i)
    return tracklets

def filter_tracklets(trackers,min_size):
    # check if duplicate frame in a tracker (TOO: correct this)
    # Remove tracklets with <60 detections
    return [tl for tl in trackers if len(tl)>=min_size]

def apply_scta(trackers, temp_th=10, dist_th=4, trk_min_size=10,  onlyFilter=False, d_metric='center_dist'):
    #
    long_tracklets = get_tracklets(trackers)
    long_tracklets = filter_tracklets(long_tracklets, min_size=trk_min_size)
    short_tracklets =copy.deepcopy(long_tracklets)
    # long_tracklets = filter_tracklets(long_tracklets, min_size=60)
    # short_tracklets = filter_tracklets(short_tracklets, min_size=60)
    if not onlyFilter:
        print('start offline {} tracklet association...'.format(len(long_tracklets)))
        [tct_tracklets, short_tracklets] = merge_tracklets(long_tracklets, short_tracklets, t_thr=temp_th, d_thr=dist_th, metric=d_metric)
        print('After merging {} tracklets survive...'.format(len(long_tracklets)))

    #trackers = filter_tracklets(trackers, min_size=trk_min_size)
    totalTrack = len(tct_tracklets)
    return tct_tracklets, totalTrack