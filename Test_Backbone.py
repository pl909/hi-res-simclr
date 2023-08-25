
'''


Script for testing accuracy (confusion matrix, ROC curve) of finetuned model.
Loads a model in. Extracts images from a test dataset, saves an ROC curve and prints out confusion matrix.

Alter test_step and end_epoch_test in SimCLR_eval

'''



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
import warnings
import warnings
from Train_Backbone import *


# Disable warnings for the specific module



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




def default(val, def_val):
    return def_val if val is None else val

# reproducibility seed
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

# load weights
# From https://github.com/PyTorchLightning/pytorch-lightning/issues/924
def weights_update(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f'Checkpoint {checkpoint_path} was loaded')
    return model


from PIL import ImageDraw
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

import random


from PIL import ImageDraw
from torch.optim.lr_scheduler import ReduceLROnPlateau


# transform input in all of these dataloaders is not transformes applied (set true to all)

# Unlabeled DataLoader (unused in this case)  see UnlabeledDataset.py for reference.

def get_stl_dataloader(batch_size):
    unlabeled = LabeledDataset('C:/AlgoInterns/Data/UnlabeledData', transform=True)
    unlabeledLoader = DataLoader(dataset=unlabeled, batch_size=batch_size, num_workers=cpu_count() // 2, shuffle=True)
    return unlabeledLoader

# Custom training Labeled DataLoader see LabeledDataset.py for reference.
#labellist = list of image names with positive label
def get_train_dataloader(batch_size, filelist):
    train = LabeledDataset('', labellist = filelist, transform=True)
    unlabeledLoader = DataLoader(dataset=train, batch_size=batch_size, num_workers=cpu_count()//2, shuffle=True, drop_last= True)
    return unlabeledLoader


# Custom validation Labeled DataLoader see LabeledDataset.py for reference.
#labellist = list of image names with positive label

def get_val_dataloader(batch_size, filelist):
    val = LabeledDataset('', labellist = filelist,  transform=True)
    unlabeledLoader = DataLoader(dataset=val, batch_size=batch_size, num_workers=cpu_count()//2)
    return unlabeledLoader

# Custom test Labeled DataLoader see LabeledDataset.py for reference.
#labellist = list of image names with positive label

def get_test_dataloader(batch_size):
    test = LabeledDataset('', transform=True)
    unlabeledLoader = DataLoader(dataset=test, batch_size=batch_size, num_workers=cpu_count() // 2)
    return unlabeledLoader


import matplotlib.pyplot as plt

def imshow(img):
    """
    shows an imagenet-normalized image on the screen
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    npimg = unnormalize(img).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Function to prevent batch normalization layers from being weight decayed.
def define_param_groups(model, weight_decay, optimizer_name):
    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if optimizer_name == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    return param_groups

# Function to plot and save loss curve.

def plot_and_save_loss_curve(loss_values, save_path="loss_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # Save the figure


# For parallel GPU training if needed.
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


# Important - Parameters for training.
class Hparams:
    def __init__(self):
        self.epochs = 200  # number of training epochs
        self.seed = 70000  # randomness seed
        self.cuda = True  # use NVIDIA GPU
        self.img_size = 896  # image size
        self.save = "./saved_models/"  # save checkpoint
        self.load = False  # load pretrained checkpoint
        self.gradient_accumulation_steps = 7 # gradient accumulation steps
        self.batch_size = 18
        self.lr = 0.001  # for Adam only
        self.weight_decay = 1e-6
        self.embedding_size = 128  # paper's value is 128
        self.temperature = 0.5  # 0.1 or 0.5
        self.checkpoint_path = './SimCLR_HiDenseNet121.ckpt'  # replace checkpoint path here

import torch.distributed as dist
import pytorch_lightning as pl
import torch
from torch.optim import SGD
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler
import os
from pytorch_lightning.callbacks import ModelCheckpoint

# Focal Loss implementation. (Use if needed)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(-1), 1)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Main model class for testing.
class SimCLR_eval(pl.LightningModule):
    def __init__(self, lr, model=None, linear_eval=False):
        # initialize lr, num_classes, and linear classification layer.
        super().__init__()
        self.lr = lr
        self.linear_eval = linear_eval
        self.num_classes = 2
        if self.linear_eval:
            model.eval()
        # MLP to add to end of the inout model
        self.mlp = torch.nn.Sequential(

            # applies linear classifier to end of model. Can also add another layer for slightly improved accuracy
            # Make sure this is same as FineTune_Backbone!
            torch.nn.Linear(1024, 2)

        )

        #Combine MLP and model

        self.model = torch.nn.Sequential(
            model, self.mlp
        )
        # Initalize loss, class counts, and labels.
        self.loss = torch.nn.CrossEntropyLoss()
        self.class_counts = {
            'train': [0 for _ in range(self.num_classes)],
            'val': [0 for _ in range(self.num_classes)],
            'test': [0 for _ in range(self.num_classes)]
        }
        self.y_prob = np.array([])
        self.class_correct = [0 for _ in range(self.num_classes)]
        self.class_total = [0 for _ in range(self.num_classes)]

    def forward(self, X):
        return self.model(X)


    # Ignore training - see test step
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float().cuda()
        y = y.long().cuda()
        z = self.forward(x)
        loss = self.loss(z, y)
        self.log('Cross Entropy loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        predicted = z.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        self.log('Train Acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for c in range(self.num_classes):
            self.class_counts['train'][c] += torch.sum(y == c).item()

        return loss
    # Ignore validation step - see test step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float().cuda()
        y = y.long().cuda()
        z = self.forward(x)
        loss = self.loss(z, y)
        self.log('Val CE loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        predicted = z.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        self.log('Val Accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for c in range(self.num_classes):
            self.class_counts['val'][c] += torch.sum(y == c).item()

        return loss

    # Ignore for test step
    def configure_optimizers(self):
        if self.linear_eval:
            print(f"\n\n Attention! Linear evaluation \n")
            optimizer = SGD(self.mlp.parameters(), lr=self.lr, momentum=0.9)
        else:
            optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        return [optimizer]


    # Test step function
    def test_step(self, batch, batch_idx):

        # Get x(image) and y(label)
        x, y = batch
        x = x.float().cuda()
        y = y.long().cuda()
        #print(y)
        # z = output of model with x input
        z = self.forward(x)
        probabilities = F.softmax(z, dim=1)[:, 1]  # Get the probabilities for class 1

        # Gather info for ROC curve
        self.y_prob = np.concatenate((self.y_prob, probabilities.cpu().numpy())) if batch_idx != 0 else probabilities.cpu().numpy()
        #print(z)
        loss = self.loss(z, y)

        predicted = z.argmax(1)

        # Gather info for confusion matrix
        self.y_true = np.concatenate((self.y_true, y.cpu().numpy())) if batch_idx != 0 else y.cpu().numpy()
        self.y_pred = np.concatenate((self.y_pred, predicted.cpu().numpy())) if batch_idx != 0 else predicted.cpu().numpy()

        # Calculate class-wise correct predictions
        for c in range(self.num_classes):
            class_mask = (y == c)
            self.class_correct[c] += torch.sum(predicted[class_mask] == y[class_mask]).item()
            self.class_total[c] += torch.sum(class_mask).item()
        for c in range(self.num_classes):
            self.class_counts['test'][c] += torch.sum(y == c).item()


        return loss


    # Save metrics
    def test_epoch_end(self, outputs):
        # Calculate class-wise accuracy and log it
        fpr, tpr, thresholds = metrics.roc_curve(self.y_true, self.y_prob)
        roc_auc = metrics.auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png")  # Save the plot
        plt.close()

        # Log AUC
        self.log('AUC', roc_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        print(self.class_correct)
        class_accuracies = [self.class_correct[c] / self.class_total[c] for c in range(self.num_classes)]
        cm = confusion_matrix(self.y_true, self.y_pred)
        print("Confusion Matrix:")
        # Print confusion matrix
        print(cm)
        for c, acc in enumerate(class_accuracies):
            self.log(f'Test Accuracy Class {c}', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for set_name, counts in self.class_counts.items():
            print(f"\n {set_name.capitalize()} Set:")
            for c, count in enumerate(counts):
                print(f'Class {c}: {count} images')

def remove_first_occurrence(string, sub_string):
    position = string.find(sub_string)
    if position != -1:
        return string[:position] + string[position+len(sub_string):]
    return string

# Main run script
if __name__ == '__main__':
    freeze_support()
    train_config = Hparams()

    # positivesval is a list of image names that are positive. They are used in LabeledDataset to determine labels.



    # Iterates through excel file, finds image value and labels, and appends positive image names to list.
    positivesval = []
    with pd.ExcelFile('') as xls:
        df = pd.read_excel(xls, 'Sheet1')
        for i in range(5969):
            image_value = df.iloc[i, 2]
            label = df.iloc[i, 3]
            if label == 1:
                positivesval.append(str(image_value))


    backbone = HiDenseNet.HiDenseNet(size=896, weights='None', classes=128)
    backbone.fc = nn.Identity()

    # Load from finetuned model checkpoint.
    checkpoint = torch.load('SimCLR_HiDenseNet_finetune.ckpt')
    model = SimCLR_eval(train_config.lr, model=backbone, linear_eval=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.eval()
    



    # Initiate test dataloader and train it.
    test_dataload = get_test_dataloader(18, positivesval)
    trainer = pl.Trainer(gpus=[2])
    result = trainer.test(model, dataloaders=test_dataload)
    print(result)
