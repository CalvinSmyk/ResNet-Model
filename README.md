# ResNet-Model


ResNet on CIFAR-10 Dataset

This is a project to train a ResNet model on the CIFAR-10 dataset using PyTorch. The ResNet architecture is a deep neural network that has achieved state-of-the-art results on various computer vision tasks. The CIFAR-10 dataset is a well-known benchmark dataset for image classification.

RequirementsThis project requires Python 3.6 or higher and the following Python packages:

PyTorch, NumPy, argparse
You can install these packages using pip by running the following command:

pip install torch numpy argparse


Usage

To run the project, first, clone the repository to your local machine:

bash
git clone https://github.com/<username>/<repository-name>.git
cd <repository-name>


The repository contains the following files:

DataReader.py: A Python module to load and preprocess the CIFAR-10 dataset.
ImageUtils.py: A Python module with utility functions to read and write images.
Model.py: A Python module that defines the ResNet model architecture.
Train.py: A Python script to train the ResNet model on the CIFAR-10 dataset.
Test.py: A Python script to test the ResNet model on the CIFAR-10 dataset.
Training the ModelTo train the ResNet model on the CIFAR-10 dataset, run the Train.py script with the following command:

css
python Train.py --resnet_version 2 --resnet_size 18 --batch_size 128 --num_classes 10 --save_interval 10 --first_num_filters 16 --weight_decay 2e-4 --modeldir model_v1


The arguments to the script are as follows:

resnet_version: The version of ResNet to use (1 or 2).
resnet_size: The size of ResNet (number of layers).
batch_size: The number of samples per batch during training.
num_classes: The number of classes in the CIFAR-10 dataset.
save_interval: The number of epochs between saving checkpoints.
first_num_filters: The number of filters in the first convolutional layer.
weight_decay: The weight decay rate for L2 regularization.
modeldir: The directory to save the model checkpoints.
The script will train the ResNet model on the CIFAR-10 dataset and save checkpoints of the model every save_interval epochs in the modeldir directory.

Testing the ModelTo test the ResNet model on the CIFAR-10 dataset, run the Test.py script with the following command:

css
python Test.py --modeldir model_v1 --epoch 200


The arguments to the script are as follows:

modeldir: The directory containing the model checkpoints.
epoch: The epoch number of the checkpoint to use for testing.
The script will load the model checkpoint from the modeldir directory and test it on the CIFAR-10 dataset using the specified epoch number.

Results
 
The ResNet model achieved an accuracy of XX% on the CIFAR-10 dataset.

Credits
This project was inspired by the ResNet paper Deep Residual Learning for Image Recognition by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, and the PyTorch tutorial Training a Classifier.
