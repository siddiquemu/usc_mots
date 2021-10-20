import numpy as np
import cv2
from pycocotools import mask as maskUtils
from munkres import Munkres
from sklearn import metrics
from sklearn.cluster import KMeans
from collections import Counter
from statistics import mode
def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x

def convert_to_30(mask):
    x_t = np.zeros((mask.shape[0], 30, 30), dtype='float')
    x_t[0:mask.shape[0], 1:29, 1:29] = mask
    #x_t = normalize(x_t)
    return x_t


def plot_latent_feature(x_test, encoded_imgs, decoded_imgs, img_y, img_x, names):
    num_images = 20
    np.random.seed(42)
    random_test_images = np.random.randint(x_test.shape[0], size=num_images)
    minval, maxval = encoded_imgs.min(), encoded_imgs.max()

    plt.figure(figsize=(30, 4))

    for i, image_idx in enumerate(random_test_images):
        # plot original image
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(x_test[image_idx].reshape(img_y, img_x))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot encoded image
        ax = plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(encoded_imgs[image_idx].reshape(4, 8), cmap='hot', vmin=minval, vmax=maxval)
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot reconstructed image
        ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(decoded_imgs[image_idx].reshape(img_y, img_x))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # plt.show()
    plt.savefig(
        '/home/MARQNET/0711siddiqa/Mask_Instance_Clustering/MOT17/MOT17-02/latent_feature/' + names[-10:],
        dpi=300)
    plt.close()

def bandwidth_Nd(feature):
    # function to compute bw for multidimentional latent feature in mean-shift
    bw_vector = []
    for i in range(feature.shape[1]):
        kernel_i = np.var(feature[:, i], axis=0)
        kernel_i = float("{0:.5f}".format(kernel_i))
        if (kernel_i == 0):
            # covariance matrix should not be the singular matrix
            kernel_i = kernel_i + 0.00000001
        bw_vector.append(kernel_i)
    bw_vector = np.array(bw_vector)
    return bw_vector

def visualize_box_reconstruction(x_test, encoded_imgs, decoded_imgs,names):
    # visualize compressed encoded feature
    num_images = 30
    np.random.seed(42)
    random_test_images = np.random.randint(x_test.shape[0], size=num_images)
    plt.figure(figsize=(30, 4))
    for i, image_idx in enumerate(random_test_images):
        # plot original image
        ax = plt.subplot(3, num_images, i + 1)
        plt.stem(x_test[image_idx])
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot encoded image
        ax = plt.subplot(3, num_images, num_images + i + 1)
        plt.stem(encoded_imgs[image_idx])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot reconstructed image
        ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.stem(decoded_imgs[image_idx])
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig('/media/siddique/Data/CLASP2018/cluster_result/6A/cam9/final_model/box_recon/'+ names[-10:], dpi=300)
    plt.close()

def normalize_standard(x,inverse=False):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    if inverse:
        x = preprocessing.StandardScaler().inverse_transform(x)
    else:
        x = preprocessing.StandardScaler().fit_transform(x)
    return x

def prepare_data_mask(x_t):
    #x_t = normalize(x_t)
    #cam9: 13000, 13790, cam11: 10000, 11209
    x_train = x_t[0:int(x_t.shape[0]*0.9),:,:,:].astype('float32') / 255. #normalize pixel values between 0 and 1 [coarse mask has already in 0-1 scale]
    x_test = x_t[int(x_t.shape[0]*0.9):x_t.shape[0],:,:,:].astype('float32') / 255.
    print('Training Data Preparation Done...')
    return x_train, x_test

def prepare_data_box(x_t,im_w,im_h):
    #x_t = normalize(x_t)
    im_w, im_h = float(im_w),float(im_h)
    import copy
    x_norm = copy.deepcopy(x_t.astype('float'))
    # CxCy
    x_norm[:, 0:2] = (x_t[:, 0:2]+ x_t[:, 2:4] / 2.) / [im_w, im_h] #+ x_t[:, 2:4] / 2.
    # wh
    x_norm[:, 2:4] = x_t[:, 2:4] / [float(max(x_t[:, 2])), float(max(x_t[:, 3]))]
    return x_t, x_norm
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

def cluster_mode_det(fr,latent_feature, labels,cluster_center,
                     det_frame,det_at_t_pmask,n_angle,score_th,ID_ind,
                     min_cluster_size,k_value,names,color,img_y,img_x,trackers):
    # labels comes from k-means (raw id): associate id???
    det_frame = det_frame # contain all info [CXbox,CYbox, x, y, w, h, classID, angle,fr, score,mask_cx,mask_cy,area,pixels,arc_length]

    final_det, final_mask, det_frame, ID_ind, trackers = cluster_association_mnist(fr, latent_feature,
                                                                               det_frame,det_at_t_pmask,labels,
                                                                               cluster_center,ID_ind, n_angle,
                                                                               score_th,min_cluster_size,k_value,
                                                                               names,color,img_y,img_x,trackers)
    return  np.array(final_det), np.array(final_mask), det_frame, ID_ind, trackers

def expand_from_temporal_list(box_all=None, mask_30=None):
    if box_all is not None:
        box_list = [b for b in box_all if len(b) > 0]
        box_all = np.concatenate(box_list)
    if mask_30 is not None:
        mask_list = [m for m in mask_30 if len(m) > 0]
        masks_30 = np.concatenate(mask_list)
    else:
        masks_30 =[]
    return box_all, masks_30

def box_mask_overlay(ref_box, final_mask, ax, im_h, im_w, score_text,color_mask,box_color):
    box_coord = ref_box[2:6].astype(int)
    # why only consider bbox boundary for mask???? box can miss part of the object
    x_0 = box_coord[0]
    x_1 = box_coord[0] + box_coord[2]
    y_0 = box_coord[1]
    y_1 = box_coord[1] + box_coord[3]
    rle=None
    if final_mask is not None:
        mask = cv2.resize(final_mask, (box_coord[2], box_coord[3]))
        #mask_ = final_mask[int(final_mask.shape[0]/2 - box_coord[3] / 2):int(final_mask.shape[0]/2 + box_coord[3] / 2),
                #int(final_mask.shape[0]/2 - box_coord[2] / 2):int(final_mask.shape[0]/2 + box_coord[2] / 2)]
        # apply theshold on scoremap!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!cfg.MRCNN.THRESH_BINARIZE
        mask = np.array(mask > 0, dtype=np.uint8)
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        # mask transfer on image cooordinate
        im_mask[y_0:y_1, x_0:x_1] = mask
        # RLE format of instance binary mask
        rle = maskUtils.encode(np.asfortranarray(im_mask))
        # overlay both mask and box on original image
        im_mask = np.uint8(im_mask * 255)
        _,contours,_ = cv2.findContours(im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours = skimage.measure.find_contours(im_mask, 0,'high')
        for c in contours:
            polygon = Polygon(
                c.reshape((-1,2)),
                fill=True, facecolor=color_mask,
                edgecolor='y', linewidth=1.2,
                alpha=0.5)
            ax.add_patch(polygon)

    # show box
    ax.add_patch(
        plt.Rectangle((x_0, y_0),
                      box_coord[2],
                      box_coord[3],
                      fill=False, edgecolor=box_color,
                      linewidth=1, alpha=0.8))
    ax.text(
        box_coord[0], box_coord[1] - 2,
        score_text,
        fontsize=8,
        family='serif',
        bbox=dict(
            facecolor=box_color, alpha=0.5, pad=0, edgecolor='none'),
        color='red')
    return ax, rle

def concat_images(img1,img2,names,path):
    img_final = np.hstack([img1,img2])
    im = Image.fromarray(img_final)
    im.save(path+'/all_box_on_img/'+names[-10:])

def denormalized_box(x_t,im_w,im_h):
    w = x_t[:, 2]* im_w #/ 1920#np.max(x_t[:, 4])
    h = x_t[:, 3]* im_h# / 1080#np.max(x_t[:, 5])
    x_0 = x_t[:, 0]*im_w - w/2
    y_0 = x_t[:, 1]*im_h - h/2
    x_f = np.array([x_0, y_0, w, h])#, x_0, y_0, area,diag])
    x_f = np.transpose(x_f)
    return x_f

def all_det_on_image(img1,input_box_f,decoded_box_f,color,names,unique,path): #decode_box_f: (n_sample,4), input_box_f: (n_sample,9)
    #decoded_box_f = denormalized_box(decoded_box_f)
    for i in range(input_box_f.shape[0]):
        input_box = input_box_f[i,:]

        img1 = cv2.rectangle(img1, (int(input_box[2]),int(input_box[3])),\
                                      (int(input_box[2]+input_box[4]), int(input_box[3]+input_box[5])),color, 4 )
        #img2 = cv2.rectangle(img2, (int(decoded_box[0]),int(decoded_box[1])),\
                                  # (int(decoded_box[0]+decoded_box[2]), int(decoded_box[1]+decoded_box[3])),color, 5 )
    for i in range(len(unique)):
        decoded_box = decoded_box_f[i, :]
        cluster_score = decoded_box[6]
        score_text = ' {:0.2f}'.format(cluster_score)
        img1 = cv2.rectangle(img1, (int(decoded_box[2]), int(decoded_box[3])), \
                         (int(decoded_box[2]+decoded_box[4]), int(decoded_box[3]+decoded_box[5])),(0,0,255), 4 )
        cv2.putText(img1, score_text, (int(decoded_box[2]), int(decoded_box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1, cv2.LINE_AA)
    im = Image.fromarray(img1)
    im.save(path + '/all_box_on_img/' + names[-10:])
    #concat_images(img1, img2, names,path)

def plot_tSNE(fr,box_raw,mask_encoding,box_encoding,latent, labels,path): #fr,pax_box_norm,mask_encoding,box_encoding,latent_feature,
    feature_name = ['4D Box','Mask Encoding','Box Encoding','Joint Embedding']
    unique_labels = np.unique(labels)

    #tsne = TSNE(n_components=2, verbose=1, perplexity=104, n_iter=300)
    perplexity = 20
    ini_dim = 104
    Y1 = tsne(box_raw, 2, ini_dim,perplexity)
    #Y1 = tsne.fit_transform(box_raw)
    Y2 = tsne(mask_encoding, 2,ini_dim,perplexity)
    #Y2 = tsne.fit_transform(mask_encoding)
    Y3 = tsne(box_encoding, 2, ini_dim,perplexity)
    #Y3 = tsne.fit_transform(box_encoding)
    Y4 = tsne(latent, 2, ini_dim,perplexity)
    #Y4 = tsne.fit_transform(latent)
    # color map
    colors = []
    for i in labels:
        if unique_labels[0]==i:
            colors.append('red')
        if unique_labels[1]==i:
            colors.append('green')
        if unique_labels[2]==i:
            colors.append('blue')
        if unique_labels[3] == i:
            colors.append('cyan')
        if unique_labels[4] == i:
            colors.append('magenta')
    plt.figure()
    plt.xlabel('1st Dimension')
    plt.xlabel('2nd Dimension')
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    Y = [Y1, Y2, Y3, Y4]
    #Y = Y/max(Y)
    for i in range(len(Y)):
        ax = plt.subplot(2,2,i+1)
        #fig = plt.figure(frameon=False)
        #ax.('X1')
        #ax.get_yaxis('X2')
        ax.text(0,0,feature_name[i],fontsize=8)
        ax.scatter(Y[i][:, 1], Y[i][:, 0], 20, c=colors, cmap='Dark2',alpha=0.8)#/abs(Y[i][:, 0].max())
        #ax.axis('off')
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
    plt.savefig(os.path.join(path,'{:04d}'.format(fr)), dpi=300)
    plt.close()

def check_IOU_mask_at_fr(fr,im_h,im_w,mask,combined_mask_per_frame):
    sanity_checked_mask = mask
    if maskUtils.area(maskUtils.merge([combined_mask_per_frame[fr], mask], intersect=True)) > 0.0:
        print("Objects with overlapping masks in frame " + fr)
        sanity_checked_mask = None
    return sanity_checked_mask

def formatting(boxs):
    b6 = np.insert(boxs, 6, values=1, axis=1)
    b7 = np.insert(b6, 7, values=1, axis=1)
    b8 = np.insert(b7, 8, values=0, axis=1)
    return b8

def fig_initialize(name):
    im = cv2.imread(name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_h, im_w, _ = im.shape
    # define figure
    dpi = 200
    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)
    return im,ax,fig,im_h,im_w

def save_initialized_tracklets(tracklets_box,tracklets_mask,fr,PAX_Tracker,time_lag,seqs,im_w,im_h,vis):
    for i in np.unique(tracklets_box[:,0]):
        if vis:
            # To overlay the boxs and masks on original image
            path_i = seqs[1] + '/%04d' % (i) + '.png'
            im = cv2.imread(path_i)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_h, im_w, _ = im.shape
            # define figure
            dpi = 200
            fig = plt.figure(frameon=False)
            fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')
            fig.add_axes(ax)
            ax.imshow(im)
        # frame by frame collect tracklet detections
        dets_fr_prev = tracklets_box[tracklets_box[:, 0] == i]
        masks_fr_prev = tracklets_mask[tracklets_box[:, 0] == i]
        # save tracked ID for previous frames of time lag
        init_ids = dets_fr_prev[:, -1]
        for id in init_ids:
            refined_det_lag = dets_fr_prev[np.where(dets_fr_prev[:, -1] == id)][0]
            final_mask = masks_fr_prev[np.where(dets_fr_prev[:, -1] == id)][0]
            # save detection (MOT) and mask (MOTS) to evaluate the overall performance
            refined_det_lag[0] = i
            # pax_eval.append(refined_det_lag)
            color_mask = color[
                int(refined_det_lag[-1])]  # np.array([0, 1, 0], dtype='float32')  # green
            box_color = color[int(refined_det_lag[-1])]
            score_text = str(int(refined_det_lag[-1]))
            PAX_Tracker.writelines(str(i) + ' ' + str(refined_det_lag[-1]) + ' ' + str(
                refined_det_lag[2]) + ' ' + str(refined_det_lag[3])
                                   + ' ' + str(refined_det_lag[4]) + ' ' + str(
                refined_det_lag[5]) + ' ' + str(1) + ' ' + str(-1) + ' ' + str(-1) + ' ' + str(-1) + '\n')
            if vis:
                # RLE format of instance binary mask
                ax, rle = box_mask_overlay(refined_det_lag, final_mask, ax, im_h, im_w,
                                           score_text, color_mask, box_color)
        if fr % 1 == 0:
            fig.savefig(out_path_seq  + path_i[-8:], dpi=dpi)
        plt.close()

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate

def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    # TODO: some data points remain unassociated (labels = -1): unique predicted class label might be larger than gt??
    # TODO: currently the unassociated data points considered as wrong assignment (replace with any class) which harm the clustering score
    # TODO: We can use dont care for unassocaited data ponts: since they might be associated correctly in future!!!!
    # find unassociated data points and replace with random class: wrong assignment
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    if -1 in Label2 and nClass1==nClass2:
        L2[L2==-1]=max(L2)+1
    if -1 in Label2 and nClass1<nClass2:
        L2[L2==-1]=L2[L2>=0][-1]
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    #assert  nClass1==nClass2,'In cluster prediction, the K value should be similar to GT'
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def compute_purity(y_true, y_pred):
    """
    Calculate the purity, a measurement of quality for the clustering
    results.

    Each cluster is assigned to the class which is most frequent in the
    cluster.  Using these classes, the percent accuracy is then calculated.

    Returns:
      A number between 0 and 1.  Poor clusterings have a purity close to 0
      while a perfect clustering has a purity of 1.

    """

    # get the set of unique cluster ids
    clusters = set(y_pred)

    # find out what class is most frequent in each cluster
    cluster_classes = {}
    correct = 0
    for cluster in clusters:
        # get the indices of rows in this cluster
        indices = np.where(y_pred == cluster)[0]

        cluster_labels = y_true[indices]
        majority_label = np.argmax(np.bincount(cluster_labels))
        correct += np.sum(cluster_labels == majority_label)

    return float(correct) / len(y_pred)

def cumsum_diff(k_window):
    cumsum = 0
    for i, O in enumerate(k_window):
        if i == 0:
            cumsum += O
        else:
            diff = k_window[i] - k_window[i-1]
            if diff > 0:
                cumsum += diff
    return cumsum

def remove_duplicate(trklt):
    filtered_trklt=[]
    t_stamps=[]
    for track in trklt:
        if track[0] not in t_stamps:
            filtered_trklt.append(track)
            t_stamps.append(track[0])
    return filtered_trklt

def save_tracks(trackers,track_saver,out_path_seq,fr_start,fr_end,color,
                im_h,im_w,seq,img_format, vis=False):
    start_lag = 0
    for fr in range(fr_start,fr_end+1):
        if vis:
            path_i = os.path.join(seq, 'imgs/{:04d}.png'.format(fr - start_lag))
            im, ax, fig, im_h, im_w = fig_initialize(path_i)

        fr_box = []
        for id in trackers.keys():
            trklt = trackers[id]
            # TODO: verify that tracker should not contatin duplicate frame
            trklt = remove_duplicate(trklt)
            for i, bbm in enumerate(trklt):
                if bbm[0] == fr:
                    fr_box.append(bbm)
        # select frame instances and solve mask  instances
        if fr_box:
            #refined_dets = fr_box
            #refined_masks = fr_mask
            for i,bb in enumerate(fr_box):
                cluster_score = bb[6]
                color_mask = color[int(bb[1])]  # np.array([0, 1, 0], dtype='float32')  # green
                box_color = color[int(bb[1])]
                score_text = str(int(bb[1]))
                track_saver.writelines(str(bb[0]) + ' ' + str(bb[1]) + ' ' + str(bb[2]) + ' ' + str(bb[3])
                                       + ' ' + str(bb[4]) + ' ' + str( bb[5]) + ' ' + str(-1) + ' ' +
                                       str(-1) + ' ' + str(-1) + ' ' + str(-1) + '\n')
                if vis:
                    final_mask = None
                    ax, rle = box_mask_overlay(np.array(list(bb)), final_mask, ax, im_h, im_w,
                                               score_text, color_mask, box_color)
            if vis and fr % 1 == 0:
                fig.savefig(out_path_seq + path_i[-8:], dpi=200)
                plt.close()
            print('Frame {}...'.format(fr-start_lag))
    print('...............done................')