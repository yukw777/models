# Google Object Detection API for RSNA Pneumonia Challenge
This is some of the code I've written to train object detection models using Google Object Detection API for the RSNA Pneumonia Challenge. The best way to get started is to use Docker and follow the steps below.

## How to train a model
These steps use Docker.

1. Install [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker)
1. Download Kaggle data.
1. Run `data.sh` to generate Tensorflow train/eval records from the training data.
1. Download pretrained models from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
1. Run `train.sh` to run the training.

## TODO
- Ability to run the test and eval scripts.
