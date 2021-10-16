import cv2
import numpy as np
import glob
import os
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from utils.t_SNE_plot import *
from PIL import ImageColor
from PIL import Image
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances
import pdb

# initialize color arrays for visualization
number_of_colors = 1500  # should be greater than possible IDs

def xywh2x1y1x2y2(t_window):
    for bb in t_window:
        bb[4] = bb[4] + bb[2]
        bb[5] = bb[5] + bb[3]
    return t_window

def x1y12xywh(t_window):
    for bb in t_window:
        bb[4] = bb[4] - bb[2]
        bb[5] = bb[5] - bb[3]
    return t_window

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

def prepare_box_data(x_t,im_h,im_w):
    # x_t is the raw data
    #x_t = normalize(x_t)
    im_w,im_h = float(im_w), float(im_h)
    x_0 = x_t[:, 2] / im_w
    y_0 = x_t[:, 3] / im_h
    # instead of variable scaling the width and height, use a fixed value (average width and height)
    w = x_t[:, 4] / im_w #np.max(x_t[:, 4]).astype('float')
    h = x_t[:, 5] / im_h #np.max(x_t[:, 5]).astype('float')
    Cx = (x_t[:, 2] + x_t[:, 4] / 2.) / im_w#np.max((x_t[:, 2] + x_t[:, 4] / .2))
    Cy = (x_t[:, 3] + x_t[:, 5] / 2.) / im_h#np.max((x_t[:, 2] + x_t[:, 4] / .2))
    area = (x_t[:, 4] * x_t[:, 5]) / (im_w*im_h)
    diag = np.sqrt(x_t[:, 4]**2 + x_t[:, 5]**2)/np.sqrt(im_w**2+im_h**2)
    # prepare dim = 8:[Cx,Cy,x,y,w,h,wh,class]
    x_f = np.array([Cx, Cy,w,h])
    #x_f = np.array([Cx, Cy, w, h,x_0,y_0,area,diag])
    x_f = np.transpose(x_f)
    #x_f = normalize(x_f)
    return x_t, x_f

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

def box_mask_overlay(ref_box, final_mask, ax, im_h, im_w, score_text,color_mask,box_color):
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
        #mask = cv2.resize(final_mask, (box_coord[2], box_coord[3]))
        # apply theshold on scoremap!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!cfg.MRCNN.THRESH_BINARIZE
        #mask = np.array(mask >= 0.5, dtype=np.uint8)
        #im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        # mask transfer on image cooordinate
        #im_mask[y_0:y_1, x_0:x_1] = mask
        #im_mask[y_0:y_1, x_0:x_1] = mask[
         #                           (y_0 - box_coord[1]):(y_1 - box_coord[1]),
          #                          (x_0 - box_coord[0]):(x_1 - box_coord[0])
           #                         ]

        # RLE format of instance binary mask
        im_mask = final_mask
        #rle = maskUtils.encode(np.asfortranarray(im_mask))
        # overlay both mask and box on original image
        im_mask = np.uint8(im_mask * 255)
        _,contours,_ = cv2.findContours(im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours = skimage.measure.find_contours(im_mask, 0,'high')

        for c in contours:
            polygon = Polygon(
                c.reshape((-1,2)),
                fill=True, facecolor=color_mask,
                edgecolor='y', linewidth=1,
                alpha=0.5)
            ax.add_patch(polygon)


    # show box

    ax.add_patch(
        plt.Rectangle((x_0, y_0),
                      box_coord[2],
                      box_coord[3],
                      fill=False, edgecolor=box_color,
                      linewidth=2, alpha=0.8))

    ax.text(
        int(box_coord[0]), int(box_coord[1]),
        score_text,
        fontsize=10,
        family='serif',
        #bbox=dict(facecolor=box_color,alpha=0.5, pad=0, edgecolor='none'),#
        color='red')
    return ax

def plot_latent_feature(x_test, encoded_imgs, decoded_imgs, img_y, img_x, names,out_path, seq):
    latent_feature_path = os.path.join(out_path,'latentf',seq)
    if not os.path.exists(latent_feature_path):
        os.makedirs(latent_feature_path)
    num_images = len(x_test)
    np.random.seed(42)
    #random_test_images = np.random.randint(x_test.shape[0], size=num_images)
    random_test_images = range(x_test.shape[0])
    minval, maxval = encoded_imgs.min(), encoded_imgs.max()

    plt.figure(figsize=(len(x_test), 4))

    for i, image_idx in enumerate(random_test_images):
        # plot original image
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(x_test[i].reshape(img_y, img_x, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot encoded image
        ax = plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(encoded_imgs[i].reshape(16, 8), cmap='hot', vmin=minval, vmax=maxval)
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot reconstructed image
        ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(decoded_imgs[i].reshape(img_y, img_x, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # plt.show()
    plt.savefig(
        os.path.join(latent_feature_path, names[-10:]),
        dpi=300)
    plt.close()

def plot_embed_affinity(x_test, encoded_imgs, decoded_imgs, img_y, img_x, names,out_path, seq,color):
    latent_feature_path = os.path.join(out_path,'latentf',seq)
    if not os.path.exists(latent_feature_path):
        os.makedirs(latent_feature_path)
    #compute affinity array
    embed_affinity = pairwise_distances(encoded_imgs)*10
    # visualize affinity in 3D plot
    # Create plot
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()

    '''
    ax = fig.add_subplot(2, 1, 1)
    # plot reconstructed image
    for i, image_idx in enumerate(decoded_imgs):
        ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(decoded_imgs[image_idx].reshape(img_y, img_x, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    '''
    ax1 = fig.add_subplot(111, projection='3d')
    xpos = []
    ypos = []
    zpos = []
    dx = []
    dy = []
    dz = []
    clrs = []
    for i in range(len(embed_affinity)):
        for j in range(len(embed_affinity)):
            xpos.append(i)
            ypos.append(j)
            zpos.append(0)
            dx.append(0.1)
            dy.append(0.1)
            if i!=j:
                dz.append(embed_affinity[i,j])
            else:
                dz.append(10)
            clrs.append(color[i])
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=clrs)
    plt.savefig(
        os.path.join(latent_feature_path, 'affinity'+names[-10:]),
        dpi=300)
    plt.close()



def plot_tSNE(boxs, X, labels, path, fr, color):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    Y = tsne(X, 2, 104, 20.4)
    #separate datapoints for each cluster labels

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('t')

    for l in np.unique(labels):
        data = Y[labels == l]
        frs = boxs[:,0][labels == l]
        x, y, z =data[:,0]/data[:,0].max(),data[:,1]/data[:,1].max(),frs
        clr = ImageColor.getcolor(color[l], "RGB")
        ax.scatter(x, y, z, alpha=0.8, c='g', edgecolors='none', s=30, label="C{}".format(l))
        #plt.scatter(x, y, z, 30, clr, cmap='hot')

    #plt.legend(loc=2)
    #plt.savefig(out_dir+'{}_{}.png'.format(i,j), dpi=300)

    #plt.scatter(Y[:, 0], Y[:, 1], 30, labels, cmap='hot')
    plt.legend(loc=2)
    plt.savefig(path+'/%06d' % (fr - 1) + '.png', dpi=300)
    plt.close()

def check_IOU_mask_at_fr(fr,mask,combined_mask_per_frame):
    sanity_checked_mask = mask
    if maskUtils.area(maskUtils.merge([combined_mask_per_frame, mask], intersect=True)) > 0.0:
        print("Objects with overlapping masks in frame " + fr)
        sanity_checked_mask = None
    return sanity_checked_mask

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

def Coarse2ImMask(box,final_mask,im_h,im_w,identity):
    box_coord = box[2:6].astype(int)
    x_0 = max(box_coord[0], 0)
    x_1 = min(box_coord[0] + box_coord[2], im_w)
    y_0 = max(box_coord[1], 0)
    y_1 = min(box_coord[1] + box_coord[3], im_h)

    mask = cv2.resize(final_mask, (box_coord[2], box_coord[3]))
    # apply theshold on scoremap!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!cfg.MRCNN.THRESH_BINARIZE
    mask = np.array(mask >= 0.5, dtype=np.uint8)
    mask[mask>0]=identity
    im_mask = np.zeros(im_shape, dtype=np.uint8)
    # mask transfer on image cooordinate
    im_mask[y_0:y_1, x_0:x_1] = mask
    #im_mask[y_0:y_1, x_0:x_1] = mask[
     #                           (y_0 - box_coord[1]):(y_1 - box_coord[1]),
      #                          (x_0 - box_coord[0]):(x_1 - box_coord[0])
       #                       ]
    return im_mask

def mask2box(iMask):
    _,contours,_ = cv2.findContours(iMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>1:
        merge_contours = []
        for cnt in contours:
            merge_contours = merge_contours+cnt.tolist()
        box = cv2.boundingRect(np.array(merge_contours))
    else:
        box = cv2.boundingRect(contours[0])
    return np.array(box)

def instance_mask(CombinedMask,refined_det):
    refined_mask = []
    for i,box in enumerate(refined_det):
        iMask = np.copy(CombinedMask)
        col_id = box[1]
        if box[1] > 255:
            col_id = box[1] - 200
        iMask[iMask!=col_id] = 0
        iMask[iMask == col_id] = 1
        #refined_det[i,2:6] = np.array(maskUtils.toBbox(maskUtils.encode(np.asfortranarray(iMask))))

        #box_coord = mask2box(iMask)
        #mask_crop = iMask[box_coord[1]:box_coord[1] + box_coord[3], box_coord[0]:box_coord[0] + box_coord[2]]
        #mask_28 = masks_resize(box_coord, mask_crop, 28, 28, 0)
        #refined_mask.append( cv2.resize(mask_crop,(28,28)))
        #imask = imask.astype('uint8')
        refined_mask.append(iMask)
    #_,refined_mask = expand_from_temporal_list(None, refined_mask)
    return np.array(refined_det), np.array(refined_mask)

def save_combined_mask(combined_mask,fr,color,start_lag,img_format,out_mask_path):
    im = cv2.cvtColor(combined_mask,cv2.COLOR_GRAY2RGB)
    ids = np.unique(im)[1:]
    for id in ids:
        r,g,b = ImageColor.getrgb(color[id])
        im[:, :, 0][im[:, :, 0] == id] = r
        im[:, :, 1][im[:, :, 1] == id] = g
        im[:, :, 2][im[:, :, 2] == id] = b
    FrMasks = Image.fromarray(im)
    FrMasks.save(out_mask_path  + '%04d' % (fr-start_lag) + img_format)

def refine_overlapped_mask(refined_det,final_mask,im_shape,fr,color):
    #det: [fr,instanceID,x,y,w,h,score,classID(person=1,car=3)]: (n,9)
    #mask: (n,28,28)
    # CombinedMask: single image mask for all instances after XOR operation
    for i,box in enumerate(refined_det):
        # assign color id (mapped from target id) for each mask
        if i==0:
            # image size mask with color id
            CombinedMask = Coarse2ImMask(box, final_mask[i],im_shape,box[-1])
            if fr==141:
                print('debug')
        if i>0: #apply bitwise XOR operation on all mask at current frame to remove overlap
            CombinedMask = cv2.bitwise_xor(CombinedMask,Coarse2ImMask(box,final_mask[i],im_shape,box[-1]))
            # should pass IoU test
            #sanity_checked_mask =  check_IOU_mask_at_fr(fr, final_mask[i], CombinedMask)
    # extract modified box and mask using mask color id

    #CombinedMask[CombinedMask>0]=255
    #import imageio
    #pdb.set_trace()
    save_combined_mask(CombinedMask, fr, color)

    refined_det,final_mask = instance_mask(CombinedMask,refined_det)
    #print (CombinedMask.shape)
    return refined_det,final_mask
def save_initialized_tracklets(tracklets_box,tracklets_mask,fr,mots_result,mots,time_lag,seqs,color,vis):
    #---------------------------------------------------------------------------------------------------------------
    # ***tracklet_mask - hold both tracklet head mask and traklet childs mask when tracklet is just initialized
    # ***tracklets_box - hold both tracklet head box and traklet childs box when tracklet is just initialized
    #
    #----------------------------------------------------------------------------------------------------------------
    #filter unassocited instances - '0' id
    #tracklets_mask = tracklets_mask[tracklets_box[:, 8] > 0]
    #tracklets_box = tracklets_box[tracklets_box[:,8] > 0]

    #handle mask overlap using bitwise XOR operation
    write_object = 1
    for i in np.unique(tracklets_box[:,0]):
        #mots_fr = {str(int(i-1)):[]}
        if vis and i==fr:
            # To overlay the boxs and masks on original image
            # TODO: make sure that all the tracklets head visualize at current frame
            path_i = seqs + '/%06d' % (i-1) + '.jpg'
            pdb.set_trace()
            im,ax,fig,im_h,im_w = fig_initialize(path_i)
            isDuplicate = False

        elif vis and i<fr and False:
            # To overlay the boxs and masks on original image
            # TODO: make sure that all the
            path_i = '/home/MARQNET/0711siddiqa/mots_tools/KITTI/tracking_results/vis/'+seqs.split('/')[-1] + '/%06d' % (i-1) + '.png'
            if not os.path.exists(path_i):
                path_i = seqs + '/%06d' % (i-1) + '.jpg'
            im, ax, fig, im_h, im_w = fig_initialize(path_i)
            #TODO: update full traclet instead saving the duplicate instances
            isDuplicate = True

        # frame by frame collect tracklet detections
        dets_fr_prev = tracklets_box[tracklets_box[:, 0] == i]
        masks_fr_prev = tracklets_mask[tracklets_box[:, 0] == i]

        dets_fr_prev, masks_fr_prev = refine_overlapped_mask(dets_fr_prev, masks_fr_prev, im.shape[:2], i, color)
        assert dets_fr_prev.shape[0] == masks_fr_prev.shape[0]
        #TODO: remove the duplicate instances from the dictionary
        # for new born tracklets wee need to update the previous frame instances as well to overcome mask overlap
        '''
        if isDuplicate:
            # find image id and remove previously saved result
            try:
                # find already written instances and del
                if mots_result['motsFr']:
                    pdb.set_trace()
                    del_index = []
                    for del_item in range(len(mots_result['motsFr'])):
                        if i-1 == mots_result['motsFr'][del_item]['image_id']:
                            del_index.append(del_item)
                            #del mots_result['motsFr'][0]
                            print('duplicate item deleted at {}'.format(mots_result['motsFr'][del_item]['image_id']+1))
            except:
                print('instance not available to override for newly born tracklet')
        '''
        # save tracked ID for previous frames of time lag
        init_ids = dets_fr_prev[:, 8]
        for id in init_ids:
            refined_det_lag = dets_fr_prev[np.where(dets_fr_prev[:, 8] == id)][0]
            final_mask = masks_fr_prev[np.where(dets_fr_prev[:, 8] == id)][0]
            cluster_score = refined_det_lag[6]
            # save detection (MOT) and mask (MOTS) to evaluate the overall performance
            refined_det_lag[0] = i
            # pax_eval.append(refined_det_lag)
            color_mask = color[
                int(refined_det_lag[8])]  # np.array([0, 1, 0], dtype='float32')  # green
            box_color = color[int(refined_det_lag[8])]
            score_text = str(int(refined_det_lag[8]))
            if refined_det_lag[7]==3:
                class_label = 1
            if refined_det_lag[7]==1:
                class_label = 2
            class_ID = str(int(class_label))+'{:03d}'.format(int(refined_det_lag[8]))
            if vis:
                # RLE format of instance binary mask
                ax, rle = box_mask_overlay(refined_det_lag, final_mask, ax, im_h, im_w,
                                           score_text, color_mask, box_color)
            # append dictionaies for frame-wise instances
            mots.writelines(str(int(i - 1)) + ' ' + class_ID + ' ' + str(int(class_label)) + ' ' + str(im_h) + ' ' + str(im_w) + ' ' + rle['counts'].decode("utf-8") + '\n')
            #mots_fr = write_mots_frame(refined_det_lag, rle, int(i-1), id, class_ID, mots_fr)
            #mots_result = write_mots_info(refined_det_lag, rle, int(i - 1), id, class_ID, mots_result)
        if fr % 1 == 0:
            fig.savefig(out_path_seq  + path_i[-8:], dpi=200)
        plt.close()
        #mots_result.update(str(int(fr)))
        #mots_result[str(int(fr))] = mots_fr
    #return mots_result

def refine_mask_mots(fr_box, fr_mask, im_h,im_w, fr, color, start_lag, img_format, out_mask_path):
    for i,box in enumerate(fr_box):
        # assign color id (mapped from target id) for each mask
        # TODO: verify that single channel mask with their id as pixel value can do accurate xor operation
        if i==0:
            # image size mask with color id: mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
            #if isinstance(fr_mask[i], bytes):
            mask = maskUtils.decode({'size': [im_h,im_w], 'counts':fr_mask[i]})
            #else:
                #mask = maskUtils.decode({'size': [im_h, im_w], 'counts': bytes(fr_mask[i], 'utf-8')})
            #mask=mask.astype('uint16')
            col_id = box[1]
            if  box[1]>255:
                col_id = box[1]-200
            mask[mask > 0] = col_id
            #CombinedMask = Coarse2ImMask(box, mask,im_h,im_w,box[1])
            CombinedMask = mask
        if i>0: #apply bitwise XOR operation on all mask at current frame to remove overlap
            #if isinstance(fr_mask[i], bytes):
            imask = maskUtils.decode({'size': [im_h,im_w], 'counts':fr_mask[i]})
            #else:
                #imask = maskUtils.decode({'size': [im_h, im_w], 'counts': bytes(fr_mask[i],'utf-8')})
            #imask=imask.astype('uint16')
            col_id = box[1]
            if  box[1]>255:
                col_id = box[1]-200
            imask[imask > 0] = col_id
            CombinedMask = cv2.bitwise_xor(CombinedMask,imask)
    #save_combined_mask(CombinedMask, fr, color,start_lag, img_format, out_mask_path)
    refined_dets,refined_masks = instance_mask(CombinedMask,fr_box)
    return refined_dets, refined_masks


def filter_tracklet(mots_result,totalID,min_tracklet_size=5):
    del_index = []
    for id in range(1,totalID+1):
        tracklet = []
        for bbm in list(mots_result[str(id)]):
            tracklet.append(list(bbm))
        mots_result[str(id)] = tracklet
        if len(tracklet)<min_tracklet_size:
            del_index.append(id)
    if del_index:
        for id in del_index:
            del mots_result[str(id)]
    return mots_result, del_index

def remove_duplicate(trklt):
    filtered_trklt=[]
    t_stamps=[]
    for track in trklt:
        if track[0] not in t_stamps:
            filtered_trklt.append(track)
            t_stamps.append(track[0])
    return filtered_trklt

def warp_pos(pos, warp_matrix):
    p1 = np.array([pos[2], pos[3], 1])
    p2 = np.array([pos[4], pos[5], 1])
    p1_n = np.matmul(warp_matrix, p1.reshape(3,1))
    p2_n = np.matmul(warp_matrix, p2.reshape(3,1))
    align_box = np.concatenate((p1_n, p2_n)).reshape(1,4)[0]
    pos[2:6] = align_box
    return pos

def align_window(fr, bbs, img_set, number_of_iterations=100, termination_eps=0.00001, warp_mode=cv2.MOTION_EUCLIDEAN):
    """Aligns the positions of active and inactive tracks depending on camera motion."""
    """bbs: current frame bbox warped to previous frame."""
    im1 = cv2.imread(img_set[int(fr-1)])
    im2 = cv2.imread(img_set[int(fr)])
    #im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    #im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    # https://kite.com/python/docs/cv2.findTransformECC
    # im2 warped to im1 based on similarity in image intensity
    cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
    #warp_matrix = torch.from_numpy(warp_matrix)
    # if self.im_index>10:
    # pdb.set_trace()
    for bb in bbs:
        bb = warp_pos(bb, warp_matrix)
        #align_box = clip_boxes2img(align_box, im1.shape[:2])
        bb[2] = max(0,bb[2])
        bb[2] = min(bb[2], im1.shape[1])
        bb[3] = max(0,bb[3])
        bb[3] = min(bb[3], im1.shape[0])
        bb[4] = max(0,bb[4])
        bb[4] = min(bb[4], im1.shape[1])
        bb[5] = max(0,bb[5])
        bb[5] = min(bb[5], im1.shape[0])
    return bbs

def plot_window(t_window, names, colors, posXY):
    img = cv2.imread(names)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    uniq_fr = []
    for box in t_window:
        color = colors[int(box[0])]
        img = cv2.rectangle(img, (int(box[2]), int(box[3])), \
                            (int(box[2]+box[4]),
                             int(box[3]+box[5])),
                            webcolors.hex_to_rgb(color) , 4)
        if box[0] not in uniq_fr:
            cv2.putText(img, '{}'.format(int(box[0])),
                        (int(posXY[0]), int(posXY[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, webcolors.hex_to_rgb(color), 2, cv2.LINE_AA)
            posXY[1]+=50
            uniq_fr.append(box[0])
    return img

def delete_all(demo_path, fmt='png'):
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)

def save_mots(mots_result,mots,totalID,fr_start,fr_end,color,im_h,im_w,
              out_mask_path,out_path_seq,seq, dataset, img_format,
              class_label=2, scta=False, min_track_size=2,vis=False):
    print('Saving MOTS results for evaluation...')
    if dataset == 'KITTI':
        start_lag = 1
    else:
        start_lag=0
    if scta:
        # apply scta to merge short tracklet: offline
        # to apply scta, need successful tracklet termination: end item should not cluster with newly born track in k-means???
        #mots_result, del_index = filter_tracklet(mots_result, totalID, min_tracklet_size=5)
        #if metrix=='center_dist', dist_th = 0.007, for metric=='iou',dist_th=1-iou=0.99
        tct_tracklets, totalTrack = apply_scta(mots_result, temp_th=5, dist_th=0.01,
                                               trk_min_size=5, onlyFilter=False,
                                               d_metric='center_dist')#'center_dist'
        start = 0
    else:
        # filter trackers by tracker size, target area?
        # filter any duplicate detection (fr) for a cluster
        mots_result, del_index = filter_tracklet(mots_result, totalID, min_tracklet_size = min_track_size)
        totalTrack = totalID
        start = 1

    for fr in range(fr_start,fr_end+1):
        if vis:
            path_i = seq + '/%06d' % (fr - start_lag) + img_format
            im, ax, fig, im_h, im_w = fig_initialize(path_i)

        fr_box = []
        fr_mask = []
        for id in range(start, totalTrack+start):
            if scta:
                trklt = tct_tracklets[id]
                #pdb.set_trace()
                #update tracklet ids since we have merged tracklets
            else:
                if id not in del_index:
                    trklt = mots_result[str(id)]
                else: continue
            # TODO: verify that tracker should not contatin duplicate frame
            trklt = remove_duplicate(trklt)
            for i, bbm in enumerate(trklt):
                if bbm[0] == fr:
                    #update new id based on SCTA
                    if scta:
                        bbm = list(bbm)
                        bbm[1] = id+1
                        bbm = tuple(bbm)

                    fr_box.append(bbm[:-1])
                    fr_mask.append(bbm[-1])
        # select frame instances and solve mask  instances
        if fr_box:
            refined_dets,refined_masks = refine_mask_mots(fr_box, fr_mask, im_h,im_w, fr, color, start_lag, img_format, out_mask_path)
            #refined_dets = fr_box
            #refined_masks = fr_mask
            for i,bb in enumerate(refined_dets):
                cluster_score = bb[6]
                # save detection (MOT) and mask (MOTS) to evaluate the overall performance
                #bb[0] = fr-1
                # pax_eval.append(refined_det_lag)
                color_mask = color[int(bb[1])]  # np.array([0, 1, 0], dtype='float32')  # green
                box_color = color[int(bb[1])]
                score_text = str(int(bb[1]))
                if vis:
                    # RLE format of instance binary mask
                    ax = box_mask_overlay(bb, refined_masks[i], ax, im_h, im_w,
                                               score_text, color_mask, box_color)

                rle = maskUtils.encode(np.asfortranarray(refined_masks[i]))

                class_ID = str(int(class_label))+'{:03d}'.format(int(bb[1]))
                mots.writelines(str(int(fr-start_lag)) + ' '+class_ID + ' ' + str(int(class_label))
                                + ' ' + str(im_h) + ' ' + str(im_w) + ' ' + rle['counts'].decode("utf-8")+ '\n')
            if vis and fr % 1 == 0:
                fig.savefig(out_path_seq + path_i[-8:], dpi=200)
                plt.close()
            print('Frame {}...'.format(fr-start_lag))
    print('...............done................')

def get_temporal_window(fr, time_lag, pax_boxs,
                        pax_boxs_align, pax_mask, mask_rles,
                        pax_patch_rgb, det_cluster_id,
                        interpolation=False, img_align=False):

    temp_window_pbox = []
    t_window_aligned = []
    t_cluster_aligned = []
    temp_window_pmask = []
    temp_window_mask_rles = []
    temp_window_patch = []
    k_value = []
    d_thr = []
    keep_key_frame = {}
    for i in np.linspace(fr, fr - time_lag + 1, num=time_lag):
        print('fr in window', i)
        # TODO: check that at least one detection at t
        # current frame detections
        temp_windowb = pax_boxs[np.where(pax_boxs[:, 0] == i), :][0]
        temp_windowb_align = pax_boxs_align[pax_boxs_align[:, 0] == i, :]
        #temp_cluster_align = pax_boxs_align[pax_boxs_align[:, 0] == i, :]
        k_value.append(len(temp_windowb[:, 0]))  # max value of instance at t
        # get previous frame detections
        if det_cluster_id is not None and i < fr:  # (fr=6, i=2,3,4,5 has already cluster id initialized detections)
            temp_windowb = det_cluster_id[np.where(det_cluster_id[:, 0] == i)] # without dummy
            #temp_cluster_align = dets_align_centers[dets_align_centers[:, 0] == i, :]

        keep_key_frame[len(temp_windowb[:, 0])] = i

        temp_window_pbox.append(temp_windowb)
        temp_windowm = pax_mask[np.where(pax_boxs[:, 0] == i), :, :][0]
        temp_window_pmask.append(temp_windowm)

        temp_windowm_rles = mask_rles[pax_boxs[:, 0] == i]
        temp_window_mask_rles.append(temp_windowm_rles)


        # current approach: get appearance from the augmented set of RGB patches with dummy observ.
        temp_windowpatch = pax_patch_rgb[np.where(pax_boxs[:, 0] == i), :, :][0]
        temp_window_patch.append(temp_windowpatch)

        #aligned bbox set
        t_window_aligned.append(temp_windowb_align)
        #t_cluster_aligned.append(temp_cluster_align)

    temp_window_pbox, temp_window_pmask = expand_from_temporal_list(temp_window_pbox,
                                                                    temp_window_pmask)
    _, temp_window_mask_rles = expand_from_temporal_list(None, temp_window_mask_rles)
    _, temp_window_patch = expand_from_temporal_list(None, temp_window_patch)

    t_window_aligned, _ = expand_from_temporal_list(t_window_aligned, None)
    #t_cluster_aligned, _ = expand_from_temporal_list(t_cluster_aligned, None)

    k_value = np.array(k_value)
    return temp_window_pbox, temp_window_pmask, temp_window_mask_rles, temp_window_patch, k_value, t_window_aligned, keep_key_frame

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