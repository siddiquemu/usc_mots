import imageio
import cv2
import os
import sys

codebase = os.path.dirname(sys.argv[0]).split('/utils')[0]
print(codebase)
sys.path.insert(0, codebase)

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.01)
    return

if __name__ == '__main__':
    dataset = 'KITTI'
    img_list1 = []
    img_list2 = []


    result_files = os.path.join(codebase, 'results/{}/tracking_results'.format(dataset))
    for fr in range(200, 300,1):
        if dataset in ['MNIST', 'SPRITE']:
            img1 = cv2.imread(os.path.join(result_files, 'vis/{:06d}.png'.format(fr)))
            img2 = cv2.imread(os.path.join(codebase, 'data/{}_MOT/test/imgs/{:04d}.png'.format(dataset, fr)))
        if dataset in ['KITTI', 'MOT17']:
            seq='0008'
            img1 = cv2.imread(os.path.join(result_files, 'vis/{}/{:04d}.png'.format(seq, fr)))
            img2 = cv2.imread(os.path.join(codebase, 'data/{}/training/image_02/{}/{:06d}.png'.format(dataset, seq, fr)))
        if img1 is not None and img2 is not None:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img_list1.append(img1)
            img_list2.append(img2)

    create_gif(img_list1, os.path.join(result_files, "{}_output.gif".format(dataset)))
    create_gif(img_list2, os.path.join(result_files, "{}_input.gif".format(dataset)))
