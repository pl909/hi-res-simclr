import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

import model_head
import HiDenseNet
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import pickle

# This python file is a custom labeled dataset used for FineTune_Backbone and Test_Backbone.
# Set transform = true.

class LabeledDataset(Dataset):
    def __init__(self, root_dir, labellist, transform = False):

        # Initializes root_dir, which is the folder w/ images
        self.root_dir = root_dir
        # Peeks into folder to see list of images
        self.file_list = os.listdir(root_dir)

        self.transform = transform

        self.count = 0

        # List of positive images
        self.labellist = labellist


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        file_path = os.path.join(self.root_dir, filename)
        

        # Code from this to 'try' is about initalizing a black corner mask for 896 by 896 images for consistency
        width, height = 896, 896
        corrected_radius = (448 ** 2 + 248 ** 2) ** 0.5

        # Recreating the inverted mask with the corrected radius
        inverted_mask_corrected = Image.new('L', (width, height), 'black')  # 'L' mode for grayscale
        draw = ImageDraw.Draw(inverted_mask_corrected)
        background = Image.new('RGB', (896, 896), 'black')
        # Drawing a white filled circle onto the black image using the calculated center and corrected radius
        draw.ellipse([(width / 2 - corrected_radius, height / 2 - corrected_radius),
                      (width / 2 + corrected_radius, height / 2 + corrected_radius)], fill='white')


        # Image preprocessing
        try:
            # ... your existing logic ...

            # Custom cropping for images to make it 896 by 896. Can be altered.
            im = Image.open(file_path)
            border = (220, 80, 220, 80)  # left, top, right, bottom
            im = ImageOps.crop(im, border)
            sqrWidth = np.ceil(np.sqrt(im.size[0] * im.size[1])).astype(int)
            im_resize = im.resize((sqrWidth, sqrWidth))
            image = im_resize.resize((896, 896))
            # Applied black corner mask
            image = Image.composite(image, background, inverted_mask_corrected)
            # ... your existing logic ...
        # in case of truncated image
        except OSError:
            print(f"Corrupted image at index {filename}. ")
            return torch.zeros(3, 896, 896)  # or some default data



        # Code below finds labels using labellist. If statements are for my case (I used upsampled dataset) and can be altered.

        label = 0
        imgname = filename[:-4]
        if imgname.endswith('copy'):
            imgname = imgname[:-4]
        
        if any(i==imgname for i in self.labellist):
            label = 1



        # for a possible pickle file as labels.
        '''
        label = 0
        with open(pickle_path, 'rb') as file:
            loaded_dict = pickle.load(file)
            label = loaded_dict[filename]
            if label == 0 or label == 1:
                label = 0
            if label == 2 or label == 3:
                label = 1
            if label == 4:
                label = 2
        '''

        # normalize similar to way image is trained.
        normalize = T.Compose([T.ToTensor(), T.Normalize(mean=[0.1988, 0.1367, 0.0966], std=[0.1458, 0.1024, 0.0701])])
        image = normalize(image)
        return image, label







