# Unsupervised Spatio-temporal Latent Feature Clustering for Multiple-object Tracking and Segmentation.
(Accepted by BMVC 2021). Preprint available at ....

Assigning consistent temporal identifiers to multiple moving objects in a video sequence is a challenging problem. A solution to that problem would have immediate ramifications in multiple object tracking and segmentation problems. We propose a strategy that treats the temporal identification task as a spatio-temporal clustering problem. We propose an unsupervised learning approach using a convolutional and fully connected autoencoder, which we call deep heterogeneous autoencoder, to learn discriminative features from segmentation masks and detection bounding boxes. We extract masks and their corresponding bounding boxes from a pretrained semantic segmentation network and train the autoencoders jointly using task-dependent uncertainty weights to generate common latent features. We then construct constraints graphs that encourage associations among objects that satisfy a set of known temporal conditions. The feature vectors and the constraints graphs are then provided to the kmeans clustering algorithm to separate the corresponding data points in the latent space. We evaluate the performance of our method using challenging synthetic and real-world multiple-object video datasets. Our results show that our technique outperforms several state-of-the-art methods.

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
### Data Preprocessing ###
3. Reproduce the synthetic MNIST-MOT and Sprites-MOT datasets, use [tracking-by-animation](https://github.com/zhen-he/tracking-by-animation.git) or
```shell
python3 ./utils/gen_mnist.py --test 1
python3 ./utils/gen_sprite.py --test 1
```
4. Download the publicly available MOTS training and validation datasets and the public detections from [MOTS](https://www.vision.rwth-aachen.de/page/mots) 

### Test ###
5. download pretrained models
7. To test the models

### Train ###
8. To train the models from scratch

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact
