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
import pydicom


'''
Custom unlabeled dataset that returns images in tensors. 

'''

class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform = False):

        # Initializes root directory (folder containing images)
        self.root_dir = root_dir
        # List of files.
        self.file_list = os.listdir(root_dir)
        self.transform = transform
        self.count = 0




    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        file_path = os.path.join(self.root_dir, filename)
        # Try cropping images to 896 by 896 and returning tensor
        try:
            # ... your existing logic ...
            im = Image.open(file_path)
            border = (220, 80, 220, 80)  # left, top, right, bottom
            im = ImageOps.crop(im, border)
            sqrWidth = np.ceil(np.sqrt(im.size[0] * im.size[1])).astype(int)
            im_resize = im.resize((sqrWidth, sqrWidth))
            image = im_resize.resize((896, 896))
            tensor = T.ToTensor()
            image = tensor(image)
            return image
            # ... your existing logic ...
        # catch error for corrupted files.
        except OSError:
            print(f"Corrupted image at index {filename}. ")
            return torch.zeros(3, 896, 896)  # or some default data
        
    
        

def dicom_to_pil_image_rgb(filename):
    # Load the DICOM file
    dicom_data = pydicom.dcmread(filename)

    # Ensure the pixel data is RGB
    if len(dicom_data.pixel_array.shape) != 3 or dicom_data.pixel_array.shape[2] != 3:
        raise ValueError("The provided DICOM file does not contain RGB data.")

    # Extract the RGB pixel data
    image_data = dicom_data.pixel_array

    # The DICOM might store RGB data as int16, so we need to normalize to uint8 if necessary
    if image_data.dtype != np.uint8:
        # Normalize each channel
        for channel in range(3):
            image_data[:, :, channel] = ((image_data[:, :, channel] - image_data[:, :, channel].min()) /
                                         (image_data[:, :, channel].max() - image_data[:, :,
                                                                            channel].min()) * 255).astype(np.uint8)

    # Convert to PIL Image
    pil_image = Image.fromarray(image_data)

    return pil_image

