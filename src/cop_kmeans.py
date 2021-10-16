# -*- coding: utf-8 -*-
import random
import numpy as np
from scipy.interpolate import interp1d
import statistics
def cop_kmeans_constantK(dataset,labels, temporal_w,
               current_f, k, ml=[], cl=[],
               initialization='kmpp',
               max_iter=50, tol=1e-4):

    ml, cl = transitive_closure(ml, cl, len(dataset))
    ml_info = get_ml_info(ml, dataset)
    tol = tolerance(tol, dataset)
    #cluster init
    if initialization=='new_det':
        dataset_f = dataset[temporal_w==current_f]
        k=len(dataset_f)
        #TODO: If current frame has no detection: search previous frame for k
        #Note: If current frame has no detection =, we do not call association function
        if k==0:
            dataset_f = dataset[temporal_w == current_f-1]
            k = len(dataset_f)
        if k==0:
            dataset_f = dataset[temporal_w == current_f-2]
            k = len(dataset_f)
        assert len(dataset_f)==k, 'number of new detection {} at current frmae must be equal to k {}'.format(len(dataset_f),k)
        centers = initialize_centers(dataset_f, k, initialization)
    else:
        centers = initialize_centers(dataset, k, initialization)
    iter=0
    for _ in range(max_iter):
        clusters_ = [-1] * len(dataset)
        for i, d in enumerate(dataset):
            indices, _ = closest_clusters(centers, d)
            counter = 0
            if clusters_[i] == -1:
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    index = indices[counter]
                    if not violate_constraints(i, index, clusters_, ml, cl):
                        found_cluster = True
                        clusters_[i] = index
                        for j in ml[i]:
                            clusters_[j] = index
                    counter += 1
                if not found_cluster:
                    #if no cluster found for a sample return -1
                    #return None, None
                    # TODO: How to handle -1 candidates? or indices[0] for clustering all dataponts!!!!!
                    #clusters_[i] = -1
                    print('Fails to assign data points')

        if len(set(clusters_))<k: # cluster center fails to associate
            k = len(set(clusters_))

        clusters_, centers_,knew = compute_centers(clusters_, dataset, k, ml_info, temporal_w)
        shift = sum(l2_distance(centers[i], centers_[i]) for i in range(k))
        iter += 1
        #***** TODO
        if knew-k==1:
            print('outlier remain unclustered')
            clusters_ = [ -1 if c==knew-1 else c for c in clusters_ ]
            centers_ = centers_[0:-1]

        if shift <= tol:
            print('Iteration', iter)
            print('shift <= tol', shift,tol)

            break
        centers = centers_
    return clusters_, centers_

def cop_kmeans_dynamicK(dataset,labels, temporal_w,
               current_f, k, ml=[], cl=[],
               initialization='kmpp',
               max_iter=50, tol=1e-4):

    ml, cl = transitive_closure(ml, cl, len(dataset))
    ml_info = get_ml_info(ml, dataset)
    tol = tolerance(tol, dataset)
    #cluster init
    if initialization=='new_det':
        # centers = initialize_centers(dataset, k, initialization)
        # TODO: initialize cluster centers using current frame detection only
        # TODO: select more elegant way to initialize cluster centers
        dataset_f = dataset[temporal_w == current_f]
        if k == len(dataset_f):
            # current frame belongs to maximum cardinality of detection
            k = len(dataset_f)
        else:
            # which frame belongs to k?
            unq, cnt = np.unique(temporal_w, return_counts=True)
            fk = unq[::-1][np.argmax(cnt[::-1])]
            dataset_f = dataset[temporal_w == fk]
            k = len(dataset_f)
        centers = initialize_centers(dataset_f, k, initialization)
    else:
        centers = initialize_centers(dataset, k, initialization)
    iter=0
    for _ in range(max_iter):
        clusters_ = [-1] * len(dataset)
        for i, d in enumerate(dataset):
            indices, _ = closest_clusters(centers, d)
            counter = 0
            if clusters_[i] == -1:
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    index = indices[counter]
                    if not violate_constraints(i, index, clusters_, ml, cl):
                        found_cluster = True
                        clusters_[i] = index
                        for j in ml[i]:
                            clusters_[j] = index
                    counter += 1
                if not found_cluster:
                    #if no cluster found for a sample return -1
                    #return None, None
                    # TODO: How to handle -1 candidates? or indices[0] for clustering all dataponts!!!!!
                    if iter>0:
                        print('new cluster')
                        clusters_[i] = k
                        centers.append(d)
                        k = k + 1
                    else:
                        print('already initialized tracklets violate the constraints!!!')

        if len(set(clusters_))<k: # cluster center fails to associate
            k = len(set(clusters_))

        clusters_, centers_,knew = compute_centers(clusters_, dataset, k, ml_info, temporal_w)
        #clusters_, centers_, k_new = compute_centers_new(clusters_, dataset, k, ml_info, temporal_w)
        shift = sum(l2_distance(centers[i], centers_[i]) for i in range(k))
        iter += 1
        #***** TODO
        if knew-k==1:
            print('outlier remain unclustered')
            clusters_ = [ -1 if c==knew-1 else c for c in clusters_ ]
            centers_ = centers_[0:-1]

        if shift <= tol:
            print('Iteration', iter)
            print('shift <= tol', shift,tol)
            break
        centers = centers_
    return clusters_, centers_

def cop_kmeans(dataset,labels, temporal_w,
               current_f, k, ml=[], cl=[],
               k_frame=None,
               initialization='kmpp', verbose=0,
               max_iter=50, tol=1e-4):
   #used in: TCT_KITTI
    ml, cl = transitive_closure(ml, cl, len(dataset))
    ml_info = get_ml_info(ml, dataset)
    tol = tolerance(tol, dataset)
    if initialization=='kmpp' and k_frame is None:
        centers = initialize_centers(dataset, k, initialization)
    else:
        #TODO: initialize cluster centers using current frame detection only
        #TODO: select more elegant way to initialize cluster centers
        dataset_f = dataset[temporal_w==current_f]
        if k==len(dataset_f):
          #current frame belongs to maximum cardinality of detection
          print('current frame {} belongs to k'.format(current_f))
          k=len(dataset_f)
        else:
           #which frame belongs to k?
           #unq, cnt = np.unique(temporal_w, return_counts=True)
           #fk = unq[::-1][np.argmax(cnt[::-1])]
           #dataset_f = dataset[temporal_w == fk]
           #k = len(dataset_f)
           #not better than fk
           dataset_f = dataset[temporal_w == k_frame]
           #print('key frame: {}'.format(k_frame))
           assert k==len(dataset_f)
           k = len(dataset_f)
        #TODO: If current frame has no detection: search previous frame for k
        #Note: If current frame has no detection =, we do not call association
        # function since we are not considering the dummy obervation
        #if k==0:
            #dataset_f = dataset[temporal_w == current_f-1]
            #k = len(dataset_f)
        #if k==0:
            #dataset_f = dataset[temporal_w == current_f-2]
            #k = len(dataset_f)
        assert len(dataset_f)==k, 'number of new detection {} at current frmae must be equal to k {}'.format(len(dataset_f),k)
        centers = initialize_centers(dataset_f, k, initialization)
    iter=0
    #TODO: Implement cluster size constraints with ML and CL
    for _ in range(max_iter):
        #''''''''''''''''''
        # dictionary of both initialized and new detection labels, Note: For 1st current frame all detections are uninitialized
        labels_dict = dict.fromkeys(range(len(labels)), [])
        for i, l in enumerate(labels):
            labels_dict[i] = l
        # TODO: mapping is not yet used in clustering
        #trackID2clusterID = dict()
        #for i in set(labels):
            #trackID2clusterID[i] = set()
        #'''''''''''''''
        clusters_ = [-1] * len(dataset)
        for i, d in enumerate(dataset):
            indices, _ = closest_clusters(centers, d)
            counter = 0
            if clusters_[i] == -1:
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    index = indices[counter]
                    if not violate_constraints(i, index, clusters_, ml, cl):
                        found_cluster = True
                        clusters_[i] = index
                        # map the cluster assignment to labels dictionary to find any unassigned detection
                        labels_dict[i] = index
                        #'''''''''
                        #if labels[i] in trackID2clusterID:
                            #trackID2clusterID[labels[i]].add(index)
                        for j in ml[i]:
                            clusters_[j] = index
                            labels_dict[j] = index
                            #''''''
                            #if labels[j] in trackID2clusterID and labels[j]>0:
                                #trackID2clusterID[labels[j]].add(index)
                    counter += 1

                if not found_cluster:
                    #if no cluster found for a sample return -1
                    #return None, None
                    # TODO: How to handle -1 candidates? or indices[0]
                    # TODO: initiate new cluster with datapoint as center? when label=0, for label!=0?
                    if  iter>=0: #labels_dict[i]==0 and
                        if verbose: print('new cluster')
                        clusters_[i] = k
                        labels_dict[i] = k
                        # check the must-link list for new cluster
                        if ml[i]:
                            for j in ml[i]:
                                clusters_[j] = k
                                labels_dict[j] = k
                        centers.append(d)
                        k = k + 1
                    else:
                        if verbose: print('already initialized tracklets violate the constraints!!!')
                        #if labels[i] in trackID2clusterID:
                            #print('tracked id to cluster label mapped result',trackID2clusterID[labels[i]])
                            #if trackID2clusterID[labels[i]] is not None:
                                #clusters_[i]=k
                                #centers.append(d)
                               #k=k+1
        '''
        ### Verify that all the data points in the subspace find closest cluster or create new cluster if unable to find the centroids.
        ### No data points should remain unassociated
        ### We will filter out the dead tracks or accept new tracks based on dummy counts and cluster score 
        '''
        if len(set(clusters_))<k: # new cluster center fails to associate with any data points for subsequent iteration:
            # initialized center data point might be associated with other centers: CLUSTER MERGING
           k = len(set(clusters_))

        #clusters_, centers_, k_new = compute_centers(clusters_, dataset, k, ml_info, temporal_w)
        clusters_, centers_, k_new  = compute_centers_new(clusters_, dataset, k, ml_info, temporal_w)

        shift = sum(l2_distance(centers[i], centers_[i]) for i in range(k))
        iter += 1
        #set extended cluster labels to -1
        #***** TODO: Is it really need to consider????????????????????????????????????????????????????
        if k_new-k==1:
            if verbose: print('outlier remain unclustered')
            clusters_ = [ -1 if c==k_new-1 else c for c in clusters_ ]
            centers_ = centers_[0:-1]

        if shift <= tol:
            if verbose: print('Iteration', iter)
            if verbose: print('shift <= tol', shift,tol)
            break
        centers = centers_
    return clusters_, centers_

def l2_distance(point1, point2):
    return sum([(float(i)-float(j))**2 for (i, j) in zip(point1, point2)])

# taken from scikit-learn (https://goo.gl/1RYPP5)
def tolerance(tol, dataset):
    n = len(dataset)
    dim = len(dataset[0])
    averages = [sum(dataset[i][d] for i in range(n))/float(n) for d in range(dim)]
    variances = [sum((dataset[i][d]-averages[d])**2 for i in range(n))/float(n) for d in range(dim)]
    return tol * sum(variances) / dim

def closest_clusters(centers, datapoint):
    # TODO: add structural constraint from size and motion difference cost: l2_distance(center, datapoint) + structural_constraint_cost
    # Structural_constraint_cost = l2_distance()
    distances = [l2_distance(center, datapoint) for
                 center in centers]
    return sorted(range(len(distances)), key=lambda x: distances[x]), distances
def initialize_centers(dataset, k, method):
    if method == 'random':
        ids = list(range(len(dataset)))
        random.shuffle(ids)
        return [dataset[i] for i in ids[:k]]

    if method == 'new_det':
        ids = list(range(len(dataset)))
        random.shuffle(ids)
        return [dataset[i] for i in ids[:k]]

    elif method == 'kmpp':
        chances = [1] * len(dataset)
        centers = []

        for _ in range(k):
            if(sum(chances)==0):
                print ('debug')
            chances = [x/sum(chances) for x in chances]
            r = random.random()
            acc = 0.0
            for index, chance in enumerate(chances):
                if acc + chance >= r:
                    break
                acc += chance
            centers.append(dataset[index])

            for index, point in enumerate(dataset):
                cids, distances = closest_clusters(centers, point)
                chances[index] = distances[cids[0]]

        return centers

def violate_constraints(data_index, cluster_index, clusters, ml, cl):
    for i in ml[data_index]:
        if clusters[i] != -1 and clusters[i] != cluster_index:
            return True

    for i in cl[data_index]:
        if clusters[i] == cluster_index:
            return True

    return False

def interpolate(cluster,fr,classID=3):
    # cluster: (?,M) where M = embedding dimention
    # (x0,y0) and (x1, y1)
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
    for bb in cluster:
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

def compute_centers_new(clusters, dataset, k, ml_info, temporal_w):
    # step1: collect cluster members from window dets
    # step2: interpolate each clusters to get new centers: centers will shift to next frame det pose
    #        and only the current frame det will be the closest member since we do not need to
    #        associate already grouped dets (trasitive-closure relation will make sure their association)
    # Step3: map det to embeddings as center features
    # Step4: return new centers for cop-kmeans update
    # issue: interpole 128d embed or use intpolated det and mask (use current frame or closest frame to current)
    assert  len(clusters)==len(dataset)==len(temporal_w)

    cluster_ids = set(clusters)
    k_new = len(cluster_ids)
    id_map = dict(zip(cluster_ids, range(k_new)))
    #
    clusters = [id_map[x] for x in clusters]

    dim = len(dataset[0])
    centers = [[0.0] * dim for i in range(k_new)] # one center is increased due to unclustered samples
    c_center_ind = {}
    for cid in cluster_ids:
        #current_frame or closest to current frame

        c_temporal = temporal_w[np.array(clusters) == cid]
        d_temporal = np.where(np.array(clusters) == cid)[0]
        if len(c_temporal)>0:
            c_center_ind[cid] = d_temporal[np.argmax(c_temporal)]
        else:
            continue
            #assert len(c_temporal)==len(d_temporal)>0

    for c in c_center_ind.keys():
        for i in range(dim):
            centers[c][i] = dataset[c_center_ind[c]][i]
    return clusters, centers, k_new

def compute_centers(clusters, dataset, k, ml_info, temporal_w):
    #TODO: Compute cluster center in each iteration by interpolating the embeddings (or derts)
    # of a window so that the cluster become current frame detection centric. In this case
    # If we interpolate tracks we can easily cluster the dummy observation. We have no risk If we
    # we use interpolation for k-center updates since the CL and ML will take care of previous frame
    # already associated dets.
    # ****I believe this attempt will improve the results a lot since there is no way to associate new dets
    # with previous frame dets!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cluster_ids = set(clusters)
    #
    #positive_ids = set(clusters)
    #positive_ids.discard(-1)
    k_new = len(cluster_ids)
    id_map = dict(zip(cluster_ids, range(k_new)))
    #
    clusters = [id_map[x] for x in clusters]

    dim = len(dataset[0])
    centers = [[0.0] * dim for i in range(k_new)] # one center is increased due to unclustered samples

    counts = [0] * k_new
    for j, c in enumerate(clusters):
        #if c!=-1:
        for i in range(dim):
            #
            centers[c][i] += dataset[j][i]
        counts[c] += 1
    # TODO: Why each dimensions of the centers divide by the cluster population??
    for j in range(k_new):
        for i in range(dim):
            centers[j][i] = centers[j][i]/float(counts[j])
            # new_center = mean_of_cluster_members

    assert k_new >= k, 'k_new must follow k_new>=k, i.e, number of clusters will be increased based ' \
                       'on the new/dead tracks but conditions fails for clusters {}, k {}, k_new {}'.format(clusters,k,k_new)
    if k_new < k:
        ml_groups, ml_scores, ml_centroids = ml_info
        current_scores = [sum(l2_distance(centers[clusters[i]], dataset[i])
                              for i in group)
                          for group in ml_groups]
        group_ids = sorted(range(len(ml_groups)),
                           key=lambda x: current_scores[x] - ml_scores[x],
                           reverse=True)

        for j in range(k-k_new):
            gid = group_ids[j]
            cid = k_new + j
            centers[cid] = ml_centroids[gid]
            for i in ml_groups[gid]:
                clusters[i] = cid

    return clusters, centers, k_new # return only k centers

def get_ml_info(ml, dataset):
    flags = [True] * len(dataset)
    groups = []
    for i in range(len(dataset)):
        if not flags[i]: continue
        group = list(ml[i] | {i})
        groups.append(group)
        for j in group:
            flags[j] = False

    dim = len(dataset[0])
    scores = [0.0] * len(groups)
    centroids = [[0.0] * dim for i in range(len(groups))]

    for j, group in enumerate(groups):
        for d in range(dim):
            for i in group:
                centroids[j][d] += dataset[i][d]
            centroids[j][d] /= float(len(group))

    scores = [sum(l2_distance(centroids[j], dataset[i])
                  for i in groups[j])
              for j in range(len(groups))]

    return groups, scores, centroids

def transitive_closure(ml, cl, n):
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in cl:
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                #*** TODO
                #
                #ml_graph[i].discard(j) # TODO: Set changed size during iteration at 0013, 0016 sequence
                #cl_graph[i].discard(j)
                continue
                #print('Ignore exception from ml_graph')
                #raise Exception('inconsistent constraints between %d and %d' %(i, j))

    return ml_graph, cl_graph
