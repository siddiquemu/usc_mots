from utils.utils_data import resize_image

from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from utils.tct_utils import expand_from_temporal_list, fig_initialize
from matplotlib.patches import Polygon
import numpy as np
import random
import cv2
import os
import glob
import pdb
class public_detection_loader(object):
    def __init__(self,data_path,img_path,out_path,
                 img_fmt='.png', rgb_dim=128, frame_list=None):
        self.file = data_path
        self.out_path = out_path
        self.img_path = img_path
        self.img_fmt = img_fmt
        self.data_dim = rgb_dim
        self.frame_list = frame_list

    def extract_masked_patch_RGB(self, name, ref_box, img_mask, im):
        #ref_box = ref_box.astype('int')
        #im = cv2.imread(name)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_h, im_w, _ = im.shape
        # prepare binary mask image
        #mask = cv2.resize(img_mask, (ref_box[2], ref_box[3]))
        img_mask = np.array(img_mask >= 0.5, dtype=np.uint8)
        #im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        x_0 = max(ref_box[0], 0)
        x_1 = min(ref_box[0] + ref_box[2], im_w)
        y_0 = max(ref_box[1], 0)
        y_1 = min(ref_box[1] + ref_box[3], im_h)
        # image size binary mask for each instance
        ROI_binary = img_mask[y_0:y_1, x_0:x_1]
        #im_mask[y_0:y_1, x_0:x_1] = mask[
                                    #(y_0 - ref_box[1]):(y_1 - ref_box[1]),
                                    #(x_0 - ref_box[0]):(x_1 - ref_box[0])]
        rgbMask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)
        rgb_areaImg = im * rgbMask
        # Now crop
        #(y, x) = np.where(rgb_areaImg[:, :, 0] != 0)
        #(topy, topx) = (np.min(y), np.min(x))
        #(bottomy, bottomx) = (np.max(y), np.max(x))
        ROI_rgb = rgb_areaImg[y_0:y_1, x_0:x_1]
        #ROI_rgb = rgb_areaImg[topy:bottomy+1, topx:bottomx+1]
        return ROI_rgb, ROI_binary

    def get_masked_patch(self, name, box, img_mask, im):
        ref_box = box[2:6].astype('int')
        #print(ref_box)
        patch_rgb, mask_binary = self.extract_masked_patch_RGB(name, ref_box, img_mask, im)

        assert len(patch_rgb)>0
        # resize mask and patch to DHAE accepted aspect ratio
        #patch_rgb = resize_image(patch_rgb, ref_box, min_dim=128, max_dim=128, min_scale=1, mode='square')
        patch_rgb = cv2.resize(patch_rgb,(self.data_dim, self.data_dim),interpolation=cv2.INTER_CUBIC)
        mask = resize_image(mask_binary, ref_box, min_dim=self.data_dim, max_dim=self.data_dim, min_scale=1, mode='square')
        #mask_28 = resize_image(mask_binary, ref_box, min_dim=28, max_dim=28, min_scale=1, mode='square')
        mask_28 = cv2.resize(mask_binary,(28,28),interpolation=cv2.INTER_AREA)
        return mask, mask_28, patch_rgb

    def load_MRCNNX151_dets(self, color, dataset=None, vis=False):
        """
        # Load public detector instance predictions
        """
        boxs = []
        masks = []
        masks_28 = []
        patch = []
        with open(self.file, "r") as f:
            ins=1
            for line in f:
                line = line.strip()
                fields = line.split(" ")
                frame = int(fields[0])
                x1 = float(fields[1])
                y1 = float(fields[2])
                w = int(float(fields[3]))
                h = int(float(fields[4]))
                #x2 = float(fields[3])
                #y2 = float(fields[4])
                #w = x2 - x1
                #h = y2 - y1
                score = float(fields[5])
                #print('score: {}'.format(score))
                if score>=0.0:
                    class_id = int(fields[6])
                    img_h = int(fields[7])
                    img_w = int(fields[8])
                    img_mask = maskUtils.decode({'size': [img_h, img_w], 'counts': fields[9]})
                    img_mask = np.ascontiguousarray(img_mask, dtype=np.uint8)
                    assert img_h==img_mask.shape[0] and img_w==img_mask.shape[1]
                    # get DHAE feature
                    # create image file name: seq, fr
                    if frame==87:
                        print(frame)
                    if frame not in self.frame_list:
                        name = os.path.join(self.img_path,'{:06d}'.format(frame)+self.img_fmt)
                        print(name)
                        im = cv2.imread(name)
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        self.frame_list.append(frame)

                    box = np.array([frame, 0, x1, y1, w, h, score, class_id, 0])
                    #print(box)
                    if w>0 and h>0:
                        mask, mask_28, patch_rgb = self.get_masked_patch(name, box, img_mask, im)
                        if dataset=='KITTI':
                            boxs.append([frame+1, 0, x1, y1, w, h, score, class_id, 0])
                        if dataset=='MOT17':
                            boxs.append([frame, 0, x1, y1, w, h, score, class_id, 0])

                        masks.append(mask)
                        #instead of saving 28*28 mask, save rle (to reduce computation in tracker)
                        #masks_28.append(mask_28)
                        masks_28.append(fields[9])
                        patch.append(patch_rgb)
                        if vis:
                            if frame in [0,1,2,3,4]:
                                ins+=1
                                plt.imshow(patch_rgb)
                                plt.axis('off')
                                plt.savefig('/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/'
                                            'mots_tools/KITTI/tracking_results/vis/0000/paper_imgs/{}_{:06d}.png'.format(ins,frame))
                                plt.close()
        # visualize last frame to verify the conversions
        if vis:
            im, ax, fig, im_h, im_w = fig_initialize(name)
            # RLE format of instance binary mask
            boxs = np.array(boxs)
            bbs = boxs[boxs[:,0]==frame+1]
            masks_28 = np.array(masks_28)
            msks = masks_28[boxs[:,0]==frame+1]
            for i, box in enumerate(bbs):
                color_mask = color[int(box[0]+i)]  # np.array([0, 1, 0], dtype='float32')  # green
                box_color = color[int(box[0]+i)]
                score_text = '{:.2f}'.format(box[6])
                ax = self.box_mask_overlay(box, msks[i], ax, im_h, im_w,
                                  score_text, color_mask, box_color)
        if vis:
            fig.savefig(
                '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/mots_tools/KITTI/tracking_results/vis/0000/det/' +
                name.split('/')[-1], dpi=200)
            plt.close()
        return boxs, masks, masks_28, patch

    def box_mask_overlay(self, ref_box, final_mask, ax, im_h, im_w, score_text, color_mask, box_color):
        box_coord = ref_box[2:6].astype(int)

        x_0 = max(box_coord[0], 0)
        x_1 = min(box_coord[0] + box_coord[2], im_w)
        y_0 = max(box_coord[1], 0)
        y_1 = min(box_coord[1] + box_coord[3], im_h)

        #mask = cv2.resize(final_mask, (box_coord[2], box_coord[3]))
        im_mask = maskUtils.decode({'size': [im_h, im_w], 'counts': bytes(final_mask, 'utf-8')})
        im_mask = np.ascontiguousarray(im_mask, dtype=np.uint8)
        # apply theshold on scoremap!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!cfg.MRCNN.THRESH_BINARIZE
        #im_mask = np.array(mask >= 0.5, dtype=np.uint8)
        #im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        # mask transfer on image cooordinate
        #im_mask[y_0:y_1, x_0:x_1] = mask
        # overlay both mask and box on original image
        im_mask = im_mask * 255
        _, contours, _ = cv2.findContours(im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours = skimage.measure.find_contours(im_mask, 0,'high')
        for c in contours:
            polygon = Polygon(
                c.reshape((-1, 2)),
                fill=True, facecolor=color_mask,
                edgecolor='y', linewidth=1,
                alpha=0.3)
            ax.add_patch(polygon)
        # show box
        ax.add_patch(
            plt.Rectangle((x_0, y_0),
                          box_coord[2],
                          box_coord[3],
                          fill=False, edgecolor=box_color,
                          linewidth=2, alpha=0.8))
        #show score
        ax.text(
            int(box_coord[0] + box_coord[2] / 2.0), int(box_coord[1] + box_coord[3] / 2.0),
            score_text,
            fontsize=6,
            family='serif',
            # bbox=dict(facecolor=box_color,alpha=0.5, pad=0, edgecolor='none'),#
            color='red')
        return ax

if __name__ == '__main__':
    number_of_colors = 1500
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61'
    dataset='KITTI'
    data_type = 'det'
    rgb_dim = 128
    #load sequence publication detections
    if dataset=='KITTI':
        folders = glob.glob('/media/siddique/RemoteServer/LabFiles/MOTS/KITTI/training/image_02/' + '*')
        folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))

        TRCNN_det = glob.glob(storage+'/KITTI_MOTS/trainval/' + '*')
        out_path = storage+'/mots_tools/KITTI/trackrcnn_dhae_noise'
        img_fmt = '.png'
        seq_lag=2
    if dataset=='MOT17':
        folders = glob.glob('/media/siddique/RemoteServer/LabFiles/MOTS/MOT17/imgs/' + '*')
        folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        if data_type=='det':
            TRCNN_det = glob.glob(storage+ '/MOTSChallenge/trainval/' + '*')
            out_path = storage+'/mots_tools/MOT17/trackrcnn_dhae_noise'
        if data_type=='gt':
            TRCNN_det = glob.glob(storage+ '/mots_tools/MOT17/gt_folder/' + '*')
            out_path = storage+'/mots_tools/MOT17/MOT17_GT'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        seq_lag = 0
        img_fmt = '.jpg'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    TRCNN_det.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
    #mask and box separator
    for i, file in enumerate(TRCNN_det):
        print(file)
        frame_list = []
        data_loader = public_detection_loader(file,
                                              folders[i+seq_lag],
                                              out_path,
                                              img_fmt=img_fmt,
                                              rgb_dim=rgb_dim,
                                              frame_list=frame_list) # for kitti use +2

        boxs, masks, masks_28, patch = data_loader.load_MRCNNX151_dets(color, dataset, vis=False)

        np.save( os.path.join(out_path,'box0_{}_'.format(rgb_dim) + file.split('/')[-1][:4]+'.npy'),
                 np.array(boxs), allow_pickle=True, fix_imports=True)
        print('saved boxs {}'.format(np.array(boxs).shape))

        np.save(os.path.join(out_path, 'mask0_{}_'.format(rgb_dim) + file.split('/')[-1][:4] + '.npy'),
                np.array(masks), allow_pickle=True,fix_imports=True)
        print('saved masks {}'.format(np.array(masks).shape))

        np.save(os.path.join(out_path, 'patch0_{}_'.format(rgb_dim) + file.split('/')[-1][:4] + '.npy'),
                np.array(patch), allow_pickle=True,fix_imports=True)
        print('saved patch {}'.format(np.array(patch).shape))

        np.save(os.path.join(out_path, 'mask0_' + file.split('/')[-1][:4] + '.npy'),
                np.array(masks_28), allow_pickle=True,fix_imports=True)
        print('saved masks {}'.format(np.array(masks_28).shape))

