# Unsupervised Spatio-temporal Latent Feature Clustering for Multiple-object Tracking and Segmentation
(Accepted by BMVC 2021). Preprint available at [Paper](https://arxiv.org/abs/2007.07175)

Assigning consistent temporal identifiers to multiple moving objects in a video sequence is a challenging problem. A solution to that problem would have immediate ramifications in multiple object tracking and segmentation problems. We propose a strategy that treats the temporal identification task as a spatio-temporal clustering problem. We propose an unsupervised learning approach using a convolutional and fully connected autoencoder, which we call deep heterogeneous autoencoder, to learn discriminative features from segmentation masks and detection bounding boxes. We extract masks and their corresponding bounding boxes from a pretrained semantic segmentation network and train the autoencoders jointly using task-dependent uncertainty weights to generate common latent features. We then construct constraints graphs that encourage associations among objects that satisfy a set of known temporal conditions. The feature vectors and the constraints graphs are then provided to the kmeans clustering algorithm to separate the corresponding data points in the latent space. We evaluate the performance of our method using challenging synthetic and real-world multiple-object video datasets. Our results show that our technique outperforms several state-of-the-art methods.

![model_diagramv1](images/model_diagramv1.PNG)
### Requirements: ###
* Python 3.7 
* Tensorflow-gpu 1.14
* Pytorch 1.1
* Keras 2.3.1
* Pycocotools 2.0

### Installation ###

1. clone this repository and go to root folder
```python
git clone https://github.com/siddiquemu/usc_mots.git
cd usc_mots
```
2. create a conda environment
```python
conda env create -f requirements.yml
```
### Data Preprocessing ###
1. Download the preprocessed data and trained models from [usc-mots-model](https://drive.google.com/file/d/1fP6yCTF_8CUzwjlE7gU1h76_9EwWJaLb/view?usp=sharing) and put the data and model folders in the cloned repo
2. To reproduce the synthetic MNIST-MOT and Sprites-MOT datasets, use [tracking-by-animation](https://github.com/zhen-he/tracking-by-animation.git) or
```shell
python ./utils/gen_mnist.py --test 1
python ./utils/gen_sprite.py --test 1
python ./utils/gen_mnist.py --train 1
python ./utils/gen_sprite.py --train 1
```
3. To reproduce the preprocessed public data for usc-mots, i) Download the publicly available MOTS training and validation datasets and the public detections from [MOTS](https://www.vision.rwth-aachen.de/page/mots) and then run the following lines of code from the root directory
 
```shell
python ./utils/gen_mots.py --test 1 --dataset KITTI
python ./utils/gen_mots.py --train 1 --dataset KITTI
```
### Test ###
1. To test the models

```
python USC_KITTI_MOT17.py --dataset KITTI --MTL 1
python USC_KITTI_MOT17.py --dataset MOT17 --MTL 1
python USC_synthetic.py --dataset MNIST --MTL 1
python USC_synthetic.py --dataset SPRITE --MTL 1
```

### Train ###
2. To train the models from scratch

```
python train_real.py --model_type poseAPP --MTL 1
python train_synthetic.py --model_type MNIST --MTL 1
```

### Evaluation ###

1. Quantitative measures: clone [mots tools](https://github.com/VisualComputingInstitute/mots_tools)

For real data
```
cd results/mots_tools
python mots_eval/eval.py ./KITTI/tracking_results ./KITTI/gt_folder ./KITTI/val.seqmap.txt
python mots_eval/eval.py ./MOT17/tracking_results ./MOT17/gt_folder ./MOT17/val.seqmap.txt
```
For synthetic data
```
 cd results/SPRITE/tracking_results
 python -m motmetrics.apps.eval_motchallenge ./evaluation/ ./
```
2. Qualitative results

### Citing USC_MOTS ###

If you find this work helpful in your research, please cite using the following bibtex
```
@inproceedings{siddiqueBMVC2021_usc_mots,
title={Unsupervised Spatio-temporal Latent Feature Clustering for Multiple-object Tracking and Segmentation}, 
author={Abubakar Siddique and Reza Jalil Mozhdehi and Henry Medeiros},
booktitle={Proceedings of the British Machine Vision Conference},
year={2021},
}
```
