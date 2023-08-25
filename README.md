# hi-res-simclr
An implementation of SimCLR semi-supervised learning with high-resolution image inputs - part of my 2023 internship with Digital Diagnostics (w/ permission to publish).


Descriptions still in progress:

***All code compiled in Python 3.9

Training Files (All PyTorch Lightning Implementations)

Train_Backbone: Main Contrastive SimCLR Learning training script (For backbone training).
Extracts images from an unlabeled dataset, augments to an image pair, and trains a model. Model is saved to a checkpoint.

FineTune_Backbone: Finetuning script for downstream tasks.
Loads a pretrained model in. Extracts images from labeled training and validation dataset. 
Trains a model, which is saved to a checkpoint. 

Test_Backbone: Script for testing accuracy (confusion matrix, ROC curve) of finetuned model.
Loads a model in. Extracts images from a test dataset, saves an ROC curve and prints out confusion matrix.

BackBoneDRSVM: Script for training an SVM directly on the outputs of a training dataset.
Evaluates results on a test dataset.

In the current code iteration (8/24/23), all training files are set to use HiDenseNet as the model architecture.
HiDenseNet is DenseNet121 with a custom model 448 by 448 or 896 by 896 head in front, which can be found in model_head.py.


Datasets
UnlabeledDataset: Custom unlabeled dataset that returns images in tensors. 
LabeledDataset: Custom labeled dataset that returns "image, label" format.

Model Architecture Files

HiDenseNet: Attaches model_head to DenseNet121. Model head can be set to 448 by 448 input or 896 by 896 based on input.
model_head: Defines functions that create the 448 or 896 model head, since DenseNet121 only handles 224 by 224 well.
model_head based off of https://github.com/johnGettings/Hi-ResNet



Dataset Creation/Preprocessing Testing Files

TransformsTester: Takes an input image path and applies user-provided (size) transformations.  
Excel Augmenter: Reads from an excel file to appropriayely downsample or upsample a dataset.
