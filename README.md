# USC_MOTS #

Unsupervised Spatio-temporal Latent Feature Clustering for Multiple-object Tracking and Segmentation. Accepted for BMVC 2021. Preprint available at ....

### This repository is for Unsupervised Spatio-temporal Latent Feature Clustering of multiple targets in a video segment ###
![model_diagramv1](images/model_diagramv1.PNG)
### Requirements: ###
* Python 3.7 
* Tensorflow-gpu 1.14
* Keras 2.3.1
* Pycocotools 2.0

### Installation ###

1. clone this repository and go to root folder
```python
git clone https://Siddiquemu@bitbucket.org/Siddiquemu/usc_mots.git
cd usc_mots
```
2. create a conda environment
```python
conda env create -f requirements.yml
```
3. Reproduce the synthetic MNIST-MOT and Sprites-MOT datasets, use [tracking-by-animation](https://github.com/zhen-he/tracking-by-animation.git) or
```shell
python3 ./utils/gen_mnist.py --test 1
python3 ./utils/gen_sprite.py --test 1
```
4. Download the publicly available MOTS training and validation datasets and the public detections from [MOTS](https://www.vision.rwth-aachen.de/page/mots) 
5. download pretrained models
7. To test the models
8. To train the models from scratch

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact
