

'''

Finetuning script for downstream tasks.
Loads a pretrained model in. Extracts images from labeled training and validation dataset.
Trains a model, which is saved to a checkpoint.


Important areas of attention:

HParams for training, main run script at the bottom, and SimCLR_eval which is the main training class.


See Train_Backbone for details about SimCLR

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
from Train_Backbone import *




import Image3
import MedianFilter
import IO
import MirrorImage
import math
import time
import traceback
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
    test = LabeledDataset('C:/AlgoInterns/Data/TestSet', transform=True)
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

# Main training file for fine-tuning.
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

            torch.nn.Linear(1024, 2)

        )

        # epoch losses and val_epoch_losses to get a loss graph
        self.epoch_losses = []
        self.val_epoch_losses = []
        self.class_loss_accumulators = {'train': [0.0 for _ in range(self.num_classes)]}

        #Combine MLP and model
        self.model = torch.nn.Sequential(
            model, self.mlp
        )

        # Code for freezing backbone
        '''
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.mlp[0].parameters():
            param.requires_grad = True
        '''

        # Initalize loss, class counts, and labels.
        self.loss = torch.nn.CrossEntropyLoss()

        self.class_counts = {
            'train': [0 for _ in range(self.num_classes)],
            'val': [0 for _ in range(self.num_classes)],
            'test': [0 for _ in range(self.num_classes)]
        }
        self.class_correct = [0 for _ in range(self.num_classes)]
        self.class_total = [0 for _ in range(self.num_classes)]
        self.val_true_labels = []
        self.val_pred_labels = []


    def forward(self, X):
        return self.model(X)

    # training steps

    def training_step(self, batch, batch_idx):

        # Receive image (x) and label (y)
        x, y = batch
        x = x.float().cuda()
        y = y.long().cuda()

        # z = output of x through the model
        z = self.forward(x)
        # loss is calculated
        loss = self.loss(z, y)
        self.log('Cross Entropy loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        # predicted is max value of z
        predicted = z.argmax(1)

        # Remaining code is accuracy metrics or metrics used to make a confusion matrix/loss curve.
        acc = (predicted == y).sum().item() / y.size(0)
        self.log('Train Acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for c in range(self.num_classes):
            class_mask = (y == c)
            class_loss = self.loss(z[class_mask], y[class_mask]).item()
            self.class_loss_accumulators['train'][c] += class_loss * class_mask.sum().item()
            self.class_counts['train'][c] += class_mask.sum().item()

        return {'loss': loss}


    # Appends loss for loss curve
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.val_epoch_losses.append(avg_val_loss.item())


    # Executed on training end. Main purpose is to log metrics.
    def training_epoch_end(self, outputs):



        # Calculates average loss for validation
        avg_val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.val_epoch_losses.append(avg_val_loss.item())

        # Compute and log confusion matrix
        cm = confusion_matrix(self.val_true_labels, self.val_pred_labels)
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.savefig(f'confusion_matrix_epochrb700896_v5{self.current_epoch}.png')
        plt.close()

        # Reset the tracked labels for the next epoch
        self.val_true_labels = []
        self.val_pred_labels = []


        # Calculates training loss
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.epoch_losses.append(avg_train_loss.item())


        # Finds class average loss and logs it
        for c in range(self.num_classes):
            avg_class_loss = self.class_loss_accumulators['train'][c] / (
                        self.class_counts['train'][c] + 1e-6)  # avoid division by zero
            self.log(f'Class {c} Avg Loss', avg_class_loss, prog_bar=True, logger=True)

            # Reset accumulators for the next epoch
        for c in range(self.num_classes):
            self.class_loss_accumulators['train'][c] = 0.0
            self.class_counts['train'][c] = 0

        # Saves loss plot of both validation and training to a file.
        plt.figure(figsize=(10, 7))
        plt.plot(self.epoch_losses, label='Training Loss')
        plt.plot(self.val_epoch_losses, label='Validation Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training & Validation Losses')

        # Saves loss plot to file.
        plt.savefig(f'loss_plot_epoch{self.current_epoch}.png')
        plt.close()
        print(self.epoch_losses)

    # Executed every validation training
    def validation_step(self, batch, batch_idx):

        # Same structure as training_step, and logs val_accuracy at the end.
        x, y = batch
        x = x.float().cuda()
        y = y.long().cuda()
        z = self.forward(x)
        loss = self.loss(z, y)
        self.log('Val CE loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        predicted = z.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)

        self.val_true_labels.extend(y.cpu().numpy().tolist())
        self.val_pred_labels.extend(predicted.cpu().numpy().tolist())

        self.log('Val Accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for c in range(self.num_classes):
            self.class_counts['val'][c] += torch.sum(y == c).item()

        return {'loss': loss}


    # Configure Adam optimizer with ReduceLROnPlateau
    def configure_optimizers(self):
        # Define the optimizer
        optimizer = Adam(self.mlp.parameters(), lr=self.lr, weight_decay=1e-5)

        # Define the ReduceLROnPlateau scheduler
        # Here, I'm assuming you want to reduce the LR if the validation loss plateaus for 10 epochs.
        # You can adjust the parameters as per your requirements.
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=4, verbose=True),
            'monitor': 'Val CE loss',  # This should match the name you've used to log the validation loss.
            'interval': 'epoch',
            'frequency': 1,
        }

        return [optimizer], [scheduler]

        #return [optimizer], [scheduler_warmup, self.scheduler_plateau]


    # Test step can be ignored, unless testing in the same Python file. See Test_Backbone.py


    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float().cuda()
        y = y.long().cuda()
        #print(y)
        z = self.forward(x)
        #print(z)
        loss = self.loss(z, y)

        predicted = z.argmax(1)
        #print(predicted)

        '''
        if y != predicted:
             print(z)
             print(y)
             print(g)
        '''
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

    # Test step can be ignored, unless testing in the same Python file. See Test_Backbone.py
    def test_epoch_end(self, outputs):
        # Calculate class-wise accuracy and log it
        print(self.class_correct)
        class_accuracies = [self.class_correct[c] / self.class_total[c] for c in range(self.num_classes)]
        cm = confusion_matrix(self.y_true, self.y_pred)
        print("Confusion Matrix:")
        print(cm)
        for c, acc in enumerate(class_accuracies):
            self.log(f'Test Accuracy Class {c}', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for set_name, counts in self.class_counts.items():
            print(f"\n {set_name.capitalize()} Set:")
            for c, count in enumerate(counts):
                print(f'Class {c}: {count} images')

from pytorch_lightning.callbacks import Callback

# Print learning rate on epoch end

class PrintLearningRateCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                print(f"Learning Rate: {param_group['lr']}")
from pytorch_lightning import Trainer
import torchvision.models as models


if __name__ == '__main__':
    freeze_support()
    train_config = Hparams()


    # positives and positivesval are lists of image names that are positive. They are used in LabeledDataset to determine labels.

    positives = []

    # Iterates through excel file, finds image value and labels, and appends positive image names to list.
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



    save_model_path = os.path.join(os.getcwd(), "saved_models/")

    # Filename to which the final checkpoint is saved.
    filename = 'SimCLR_HiDenseNet_finetune'
    reproducibility(train_config)
    save_name = filename + '_Final.ckpt'

    # load backbone. Use same architecture as your input to SimCLR_pl in Train_Backbone.py
    backbone = HiDenseNet.HiDenseNet(size=896, weights="Dense121", classes=128)
    # Removes last couple of layers to get same output as feat_dim (1024)
    backbone.fc = nn.Identity()


    # Loads pretrained backbone and takes only the important parts for finetuning.
    model_pl = Train_Backbone.SimCLR_pl(train_config, model=HiDenseNet.HiDenseNet(size=896, weights='None', classes=128))

    model_pl = weights_update(model_pl, "SimCLR_HiDenseNet.ckpt")

    backbone_weights = model_pl.model.backbone

    # Save pretrained backbone without unnecessary parts to a checkpoint file
    torch.save({
        'model_state_dict': backbone_weights.state_dict(),
    }, 'backbone_weights.ckpt')

    # Load the same checkpoint file to backbone
    checkpoint = torch.load('backbone_weights.ckpt')
    backbone.load_state_dict(checkpoint['model_state_dict'])

    # Initializes Fine-tune model
    model = SimCLR_eval(train_config.lr, model=backbone, linear_eval=False)



    # preprocessing and data loaders
    data_loader = get_train_dataloader(train_config.batch_size, filelist=positives)
    data_loader_test = get_val_dataloader(train_config.batch_size, filelist=positivesval)

    # accumulator, callbacks and trainer
    accumulator = GradientAccumulationScheduler(scheduling={0: 7})

    checkpoint_callback = ModelCheckpoint(filename=filename, dirpath=save_model_path, save_last=True, save_top_k=2,
                                          monitor='Val Accuracy_epoch', mode='max')

    trainer = Trainer(callbacks=[accumulator, checkpoint_callback, PrintLearningRateCallback()],
                      gpus=[1],
                      max_epochs=train_config.epochs)

    # Trains and saves to checkpoint.
    trainer.fit(model, data_loader, data_loader_test)
    trainer.save_checkpoint(save_name)
