# hi-res-simclr
An implementation of SimCLR semi-supervised learning with high-resolution image inputs - part of my 2023 internship with Digital Diagnostics (w/ permission to publish core files and run scripts).


See Train_Backbone.py for main SimCLR training run script and process flow.



***All code compiled in Python 3.9.

# File Descriptions

## Training Files (All PyTorch Lightning Implementations)

### Train_Backbone: Main Contrastive SimCLR Learning training script (For backbone training).
Extracts images from an unlabeled dataset, augments to create an image pair, and trains ML backbone using Contrastive self-supervised learning using a given model architecture. Model is saved to a checkpoint.

### FineTune_Backbone: Finetuning script for downstream tasks.
Loads a pretrained model (ckpt from Train_Backbone) in. Extracts images from labeled training and validation dataset. 
Trains a classification model using training and validation datasets, which is saved to a checkpoint. 

### Test_Backbone: Script for testing accuracy (confusion matrix, ROC curve) of finetuned model.
Loads a finetuned model in, extracts images from a test dataset to forward through the model. Saves an ROC curve and prints out confusion matrix for accuracy results.

### BackBoneDRSVM: Script for training an Support Vector Machine (SVM) directly on the outputs of a pretrined backbone.
Evaluates the trained SVM with a test dataset.

In the current code iteration (8/24/23), all training files are set to use HiDenseNet as the model architecture.
HiDenseNet is DenseNet121 with a custom model 448 by 448 or 896 by 896 head in front, which can be found in model_head.py.

## Datasets

### UnlabeledDataset: 
Custom unlabeled dataset that returns images in tensors. 

### LabeledDataset: 
Custom labeled dataset that returns "image, label" format.


## Model Architecture Files


### HiDenseNet: 
Attaches model_head to DenseNet121. Model head can be set to 448 by 448 input or 896 by 896 based on input.

### model_head: 
Defines functions that create the 448 or 896 model head, since DenseNet121 only handles 224 by 224 well.
model_head based off of https://github.com/johnGettings/Hi-ResNet

## Dataset Creation/Preprocessing Testing Files

### TransformsTester: 
Takes an input image path and applies user-provided (size) transformations.  
### Excel Augmenter: 
Reads from an excel file to appropriayely downsample or upsample a dataset.
