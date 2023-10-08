'''

Main Contrastive SimCLR Learning training script (For backbone training).
Extracts images from an unlabeled dataset, augments to an image pair, and trains a model. Model is saved to a checkpoint.


See Unlabeled Dataset to edit dataset.

Key places to alter:

HParams, Augment Class, and Main code at the bottom.

Training Logic:

Trainer trains a SimCLR_pl class, which uses a user-defined model architecture as an input (HiDenseNet, in this case).
SimCLR calls a Projection head, which removes the final classification layer and adds a MLP end to 128 dim.
SimCLR calls Contrastive Loss, which is optimized. Final model is saved.


'''

# import statements

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

# Implementation of SimCLR Loss. The overall goal of the loss is to maximize similarity of positive(same image) pairs and maximize disimilarity of negative pairs.
class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        # initialize batch temperature, and an eye mask
        self.batch_size = batch_size
        self.temperature = temperature
        # Diagonal 1s and everything else 0
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        # Cosine similarity (dot product) for loss functions
    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]

        # Important step of normalizing vectors

        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        # Move mask to the GPU
        self.mask = device_as(self.mask, similarity_matrix)



        #

        # Compute the positive pairs from the similarity matrix.
        # sim_ij represents the similarity between representations of augmented views of the same image.
        sim_ij = torch.diag(similarity_matrix, batch_size)

        # sim_ji is the same as sim_ij but from the opposite side.
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        # Concatenate the positive pairs to get all positive similarities.
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        # Scale the positives by the temperature and exponentiate to compute the nominator of the loss.
        nominator = torch.exp(positives / self.temperature)

        # Mask out the positives from the similarity matrix so we only get negatives.
        # Scale the negatives by the temperature and exponentiate to compute the denominator of the loss.
        denominator = self.mask * torch.exp(similarity_matrix / self.temperature)

        # Add a small constant to the denominator to ensure numerical stability.
        denominator += 1e-8

        # Check if the entire denominator is zero (this should not happen in practice).
        if (denominator == 0).all():
            print('Becomes Nan Fail')

        # Sum the denominator across all negatives for each positive.
        denominator_sum = torch.sum(denominator, dim=1) + 1e-9

        # Compute the negative log likelihood for each positive.
        all_losses = -torch.log(nominator / denominator_sum)

        # Compute the mean loss over all positives.
        loss = torch.sum(all_losses) / (2 * self.batch_size)

        return loss


import random

# Custom random center crop
def random_center_crop(img, min_scale=0.5, max_scale=1.0, img_size=896):
    scale = random.uniform(min_scale, max_scale)
    new_size = int(img_size * scale)
    return T.CenterCrop(new_size)(img)


from PIL import ImageDraw
from torch.optim.lr_scheduler import ReduceLROnPlateau



# Augmentations applied to images. This class should and will be consistently altered
class Augment:
    """
    A stochastic data augmentation module
    Transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, img_size, s=1):

        # Code from here to the draw.ellipse is about making an 896 by 896 dimension circular mask to make images consistent.
        width, height = 896, 896
        corrected_radius = (448 ** 2 + 248 ** 2) ** 0.5

        # Recreating the inverted mask with the corrected radius
        self.inverted_mask_corrected = Image.new('L', (width, height), 'black')  # 'L' mode for grayscale
        draw = ImageDraw.Draw(self.inverted_mask_corrected)
        self.background = Image.new('RGB', (896, 896), 'black')
        # Drawing a white filled circle onto the black image using the calculated center and corrected radius
        draw.ellipse([(width / 2 - corrected_radius, height / 2 - corrected_radius),
                      (width / 2 + corrected_radius, height / 2 + corrected_radius)], fill='white')

        # end of aforementioned mask.

        self.to_pil = T.ToPILImage()


        # stacked augmentations (applied in order)
        self.train_transform = T.Compose([
            T.RandomApply([
                lambda img: random_center_crop(img, min_scale=0.7, max_scale=1.0, img_size=896),
            ], p=0.25),
            T.Resize(896),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)], p=0.5),
            T.RandomApply([
                T.RandomRotation(degrees=15),
                lambda img: random_center_crop(img, min_scale=0.5, max_scale=0.7, img_size=896),
            ], p=0.25),
            T.Resize(896),
            # T.RandomApply([T.RandomRotation(degrees=15)], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=int(896 * 0.05) // 2 * 2 + 1, sigma=(0.1))], p=0.25),
        ])
        self.normalize = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.1988, 0.1367, 0.0966], std=[0.1458, 0.1024, 0.0701])])
        # Found mean and std for normalization from averaging 1000 Images mean: tensor([0.1988, 0.1367, 0.0966]) std: tensor([0.1458, 0.1024, 0.0701])

    #Input: A tensor of size batch x image size, which is then transformed w/ mask and stacked back into a batch. Output: Two augmented versions of the input w/ same dimensions.
    def __call__(self, x):
        
        # This code will be applied to each image x. (Creates two augmented versions of x)
        x = [self.to_pil(img) for img in x]

        # Transformations applied, then mask is applied.
        x1 = [self.normalize(Image.composite(self.train_transform(img), self.background, self.inverted_mask_corrected))
              for img in x]
        x2 = [self.normalize(Image.composite(self.train_transform(img), self.background, self.inverted_mask_corrected))
              for img in x]

        # Finally, stack the list of tensors back into a batch
        x1 = torch.stack(x1)
        x2 = torch.stack(x2)

        return x1, x2

#Custom unlabeled dataset.(see UnlabeledDataset.py)

def get_stl_dataloader(batch_size, split="unlabeled"):
    unlabeled = UnlabeledDataset('C:/AlgoInterns/Data/Images', transform=True)
    unlabeledLoader = DataLoader(dataset=unlabeled, batch_size=batch_size, num_workers=cpu_count() // 2, shuffle=True,
                                 drop_last=True)
    return unlabeledLoader


import matplotlib.pyplot as plt

# Denormalizes image to show.
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

# Removes Final Dense layer and adds a MLP to project to feature embeddings.
class AddProjection(nn.Module):
    def __init__(self, config, model=None, mlp_dim=1024):
        super(AddProjection, self).__init__()

        # Initializes end embedding size, backbone architecture, and MLP.
        embedding_size = config.embedding_size
        self.backbone = default(model, HiDenseNet.HiDenseNet(size=896, weights='None', classes=128))
        mlp_dim = default(mlp_dim, 1024)  # Set mlp_dim to the output dimension of the penultimate layer in HiDenseNet
        print('Dim MLP input:', mlp_dim)
        self.backbone.FinalDense = nn.Identity()  # Replace the final layer with an Identity layer to preserve the features

        # add mlp projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x, return_embedding=False):
        print(x.size)
        embedding = self.backbone(x)
        if return_embedding:
            return embedding
        return self.projection(embedding)


# Below function: remove weight decay from batch normalization layers. In the case of using the LARS optimizer, you also need to remove weight decay from biases

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

# Function to plot and save a loss curve.
def plot_and_save_loss_curve(loss_values, save_path="loss_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # Save the figure

# Main SimCLR_pl class, which the model trains using.

class SimCLR_pl(pl.LightningModule):
    def __init__(self, config, model=None, feat_dim=1024):
        super().__init__()
        # Load config, set up augmentations, projection mlp head, and contrastive loss.
        self.config = config
        self.augment = Augment(config.img_size)
        self.model = AddProjection(config, model=model, mlp_dim=feat_dim)
        self.loss = ContrastiveLoss(config.batch_size, temperature=self.config.temperature)
        # Attributes to store the loss values and accumulation
        self.epoch_loss_values = []
        self.epoch_loss_accumulator = 0.0
        self.batch_count = 0

    def forward(self, X):
        return self.model(X)

    def get_device(self):
        return next(self.parameters()).device

    # Main training step.
    def training_step(self, batch, batch_idx):

        # custom drop last
        if batch.size(0) != self.config.batch_size:
            return

        # Augments entire batch, transfers to GPU, calculates loss and deletes to save GPU memory.
        x = batch
        x1, x2 = self.augment(x)
        x1 = x1.to(self.get_device())
        z1 = self.model(x1)
        del x1
        x2 = x2.to(self.get_device())
        z2 = self.model(x2)
        del x2
        loss = self.loss(z1, z2)
        del z1, z2

        # Accumulating loss
        self.epoch_loss_accumulator += loss.item()
        self.batch_count += 1

        # Contrastive loss logger
        self.log('Contrastive loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    # Function to plot loss curve on training epoch end.
    def on_train_epoch_end(self):
        # Store average loss for the epoch
        average_epoch_loss = self.epoch_loss_accumulator / self.batch_count
        self.epoch_loss_values.append(average_epoch_loss)
        self.log('avg_epoch_loss', average_epoch_loss, prog_bar=True, logger=True)
        # Plot and save the loss curve
        plot_and_save_loss_curve(self.epoch_loss_values, save_path=f"loss_curve_epochAdam896_{self.current_epoch}.png")

        # Reset the loss accumulator and batch count for the next epoch
        self.epoch_loss_accumulator = 0.0
        self.batch_count = 0


    # Configure Adam optimizer with Cosine Annealing LR.
    '''

    def configure_optimizers(self):
        max_epochs = int(self.config.epochs)
        param_groups = define_param_groups(self.model, self.config.weight_decay, 'adam')
        lr = self.config.lr
        optimizer = Adam(param_groups, lr=lr, weight_decay=self.config.weight_decay)

        print(f'Optimizer Adam, '
              f'Learning Rate {lr}, '
              f'Effective batch size {self.config.batch_size * self.config.gradient_accumulation_steps}')

        scheduler_warmup = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=max_epochs,
                                                         warmup_start_lr=0.0)

        return [optimizer], [scheduler_warmup]
    '''
    # Configure Adam optimizer with Reduce LR on Plateau
    def configure_optimizers(self):
        max_epochs = int(self.config.epochs)

        # important step to remove weight decay from Batch Norm layers

        param_groups = define_param_groups(self.model, self.config.weight_decay, 'adam')
        lr = self.config.lr
        optimizer = Adam(param_groups, lr=lr, weight_decay=self.config.weight_decay)

        print(f'Optimizer Adam, '
              f'Learning Rate {lr}, '
              f'Effective batch size {self.config.batch_size * self.config.gradient_accumulation_steps}')

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=3, verbose=True),
            'monitor': 'avg_epoch_loss',  # This should match the name you've used to log the validation loss.
            'interval': 'epoch',
            'frequency': 1,
        }

        # Return optimizer and schedulers with their update configurations
        return [optimizer], [scheduler]

# convert syncbn in the case of parallel training
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

# Important: Parameters to alter for training.
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
        self.lr = 1e-5  # learning rate for Adam only
        self.weight_decay = 1e-6
        self.embedding_size = 128  # paper's value is 128. This value determines the final feature embedding size
        self.temperature = 0.5  # 0.1 or 0.5 customary
        self.checkpoint_path = './SimCLR_HiDenseNet121_adam_-896.ckpt'  # replace checkpoint path here

# Main run script.
if __name__ == '__main__':

    freeze_support()

    # dist.init_process_group(backend='gloo', init_method='tcp://localhost:23456', rank=0, world_size=1)

    # Prints # available GPUs
    available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])


    # saved_models/ is the save folder of the checkpoints
    save_model_path = os.path.join(os.getcwd(), "saved_models/")
    print('available_gpus:', available_gpus)

    # file name of the saved end checkpoint
    filename = 'SimCLR_HiDenseNet121_adam896_'

    # Resume training from checkpoint or from scratch.
    resume_from_checkpoint = False

    # Initialize Parameters (important)
    train_config = Hparams()

    # Reproducibility and establish DataLoader
    reproducibility(train_config)
    save_name = filename + '.ckpt'
    data_loader = get_stl_dataloader(train_config.batch_size)

    # Initialize SimCLR class with HiDenseNet Structure (896 input in this case)
    model = SimCLR_pl(train_config, model=HiDenseNet.HiDenseNet(size=896, weights='None', classes=128), feat_dim=1024)

    # model = convert_syncbn_model(model)

    # Gradient accumulator increases training speed, increases batch size by compiling gradients from multiple batches
    accumulator = GradientAccumulationScheduler(scheduling={0: train_config.gradient_accumulation_steps})

    # Save checkpoint every 2 epochs.
    checkpoint_callback = ModelCheckpoint(filename=filename, dirpath=save_model_path, every_n_epochs=2, save_last=True,
                                          save_top_k=2, monitor='Contrastive loss_epoch', mode='min')

    # ddp = DDPStrategy(process_group_backend='mpi', parallel_devices= [torch.cuda.device(i) for i in range(torch.cuda.device_count())])


    # Trainer initialization
    if resume_from_checkpoint:
        trainer = Trainer(callbacks=[accumulator, checkpoint_callback], accelerator='cuda', num_nodes=1, gpus=1,
                          max_epochs=train_config.epochs, resume_from_checkpoint=train_config.checkpoint_path)
    else:
        trainer = Trainer(callbacks=[accumulator, checkpoint_callback], accelerator='cuda', num_nodes=1, gpus=[0],
                          max_epochs=train_config.epochs)

    # Train and save model.

    trainer.fit(model, data_loader)
    trainer.save_checkpoint(save_name)
