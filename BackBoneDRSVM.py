'''

This is a script for training an SVM directly on the outputs of a training dataset, and evaluating the results on a test dataset.
Functions are located on the top. Run statement and modifications at the bottom.

'''

# import statements below

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD, Adam
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import STL10
import torchvision.models as models
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.multiprocessing import cpu_count
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from sklearn.metrics import confusion_matrix
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.strategies import DDPStrategy
import Image3
import MedianFilter
import IO
import MirrorImage
import math
import time
import traceback
import warnings
from PIL import Image
from PIL import ImageOps
from HiDenseNet import *
from UnlabeledDataset import *
from LabeledDataset import *
from multiprocessing import freeze_support


# Check if val is None. If None, output def_val.
def default(val, def_val):
    return def_val if val is None else val
# Function for seed reproducibility
def reproducibility(config):
    SEED = int(config.seed)
    print('Seed is ' + str(SEED))
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (config.cuda):
        torch.cuda.manual_seed(SEED)


def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)

# From https://github.com/PyTorchLightning/pytorch-lightning/issues/924. Load checkpoint for testing
def weights_update(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f'Checkpoint {checkpoint_path} was loaded')
    return model

# Unlabeled DataLoader (unused in this case)
def get_stl_dataloader(batch_size):
    # Replace with path to your dataset folder.
    unlabeled = LabeledDataset('C:/AlgoInterns/Data/UnlabeledData', transform=True)
    unlabeledLoader = DataLoader(dataset=unlabeled, batch_size=batch_size, num_workers=cpu_count() // 2, shuffle=True)
    return unlabeledLoader
# Train Labeled DataLoader

# Replace '' in these for the path to your dataset folders.


def get_train_dataloader(batch_size, filelist):
    train = LabeledDataset('', labellist = filelist, transform=True)
    unlabeledLoader = DataLoader(dataset=train, batch_size=batch_size, num_workers=cpu_count()//2, shuffle=True)
    return unlabeledLoader
# Val Labeled DataLoader
def get_val_dataloader(batch_size, filelist):
    val = LabeledDataset('', labellist = filelist,  transform=True)
    unlabeledLoader = DataLoader(dataset=val, batch_size=batch_size, num_workers=cpu_count()//2)
    return unlabeledLoader
#Test Labeled DataLoader
def get_test_dataloader(batch_size):
    test = LabeledDataset('', transform=True)
    unlabeledLoader = DataLoader(dataset=test, batch_size=batch_size, num_workers=cpu_count() // 2)
    return unlabeledLoader


import matplotlib.pyplot as plt

# Show a normalized image.
def imshow(img):
    """
    shows an imagenet-normalized image on the screen
    """


    mean = torch.tensor([0.1988, 0.1367, 0.0966], dtype=torch.float32)
    std = torch.tensor([0.1458, 0.1024, 0.0701], dtype=torch.float32)
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    npimg = unnormalize(img).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# For parallel GPU if necessary
def convert_syncbn_model(module):
    module_output = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module_output = nn.SyncBatchNorm(
            module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats
        )
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
    for name, child in module.named_children():
        module_output.add_module(name, convert_syncbn_model(child))
    del module
    return module_output


from pytorch_lightning.callbacks import Callback

class PrintLearningRateCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                print(f"Learning Rate: {param_group['lr']}")
from pytorch_lightning import Trainer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import seaborn as sns

from joblib import dump, load

# Training parameters (non-important for SVM)
class Hparams:
    def __init__(self):
        self.epochs = 200  # number of training epochs
        self.seed = 77777  # randomness seed
        self.cuda = True  # use NVIDIA GPU
        self.img_size = 896  # image shape
        self.save = "./saved_models/"  # save checkpoint
        self.load = True  # load pretrained checkpoint
        self.gradient_accumulation_steps = 7  # gradient accumulation steps
        self.batch_size = 9
        self.lr = 1e-5  # for Adam only
        self.weight_decay = 1e-6
        self.embedding_size = 128  # paper's value is 128
        self.temperature = 0.5  # 0.1 or 0.5
        self.checkpoint_path = './SimCLR_HiDenseNet121_adam_-896.ckpt'  # replace checkpoint path here


if __name__ == '__main__':
    freeze_support()



    '''
    backbone = SimCLR_pl(train_config, model=HiDenseNet.HiDenseNet(size=448, weights='None', classes=128),
                         feat_dim=1024)
    backbone.fc = nn.Identity()
    model = SimCLR_eval(train_config.lr, model=backbone, linear_eval=False)
    checkpoint = torch.load('SimCLR_HiDenseNet_finetunev2__Final.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    test_dataload = get_test_dataloader(8)
    trainer = pl.Trainer(gpus=1)
    result = trainer.test(model, dataloaders= test_dataload)
    print(result)
    '''

    # Makes and saves a list of images with positive labels from data and verifies the number of positive images.
    positives = []
    with pd.ExcelFile('') as xls:
        df = pd.read_excel(xls, 'Sheet1')
        for i in range(18083):
            image_value = df.iloc[i, 2]
            label = df.iloc[i, 3]
            if label == 1:
                positives.append(str(image_value))

    positivesval = []
    with pd.ExcelFile('') as xls:
        df = pd.read_excel(xls, 'Sheet1')
        for i in range(5969):
            image_value = df.iloc[i, 2]
            label = df.iloc[i, 3]
            if label == 1:
                positivesval.append(str(image_value))
    print(len(positivesval))
    print(len(positives))
    print('5x')

    # File save name
    filename = 'SimCLR_HiDenseNet_finetuneequalSVM_'
    #For reproducobility.
    reproducibility(train_config)
    save_name = filename + '_Final.ckpt'

    # load HiDenseNet backbone
    backbone = HiDenseNet.HiDenseNet(size=896, weights='None', classes=128)
    backbone.fc = nn.Identity()
    checkpoint = torch.load('')
    backbone.load_state_dict(checkpoint['model_state_dict'])

    # preprocessing and data loaders
    data_loader = get_train_dataloader(train_config.batch_size, filelist=positives)
    data_loader_test = get_val_dataloader(train_config.batch_size, filelist=positivesval)

    backbone.eval()

    # Extract feature vectors from model.
    def extract_features(dataloader, model):
        features = []
        labels = []

        with torch.no_grad():  # Ensure no gradients are calculated
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.float().cuda()  # If using GPU
                model = model.float().cuda()
                outputs = model(inputs)

                features.append(outputs.cpu().numpy())
                labels.append(targets.cpu().numpy())

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        return features, labels


    # Extract features
    train_features, train_labels = extract_features(data_loader, backbone)
    test_features, test_labels = extract_features(data_loader_test, backbone)

    # Train SVM with balanced class weights
    clf = SVC(kernel='linear', probability=True)
    clf.fit(train_features, train_labels)

    # Dump to save svm to a file.
    dump(clf, '')

    # Evaluate SVM
    test_predictions = clf.predict(test_features)

    # Compute confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)

    # Display the confusion matrix
    print("Confusion Matrix:")
    print(cm)

    # Visualize the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrixbackboneequal.png')
    plt.close()  # Close the figure after saving

    # Compute ROC curve and ROC AUC
    fpr, tpr, _ = roc_curve(test_labels, clf.predict_proba(test_features)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 7))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curvebackboneequal.png')
    plt.close()  # Close the figure after saving
