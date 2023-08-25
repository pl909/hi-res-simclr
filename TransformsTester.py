
'''

Takes an input image path and applies user-provided (size) transformations.

'''


# Not Working, import statements need to be added
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
from PIL import Image, ImageDraw




import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

# Assuming you're using ImageFolder to load your data
dataset = ImageFolder("C:/AlgoInterns\Data/normalizeSet", transform=ToTensor())

# Compute mean and std for normalization
mean = torch.stack([img.mean(1).mean(1) for img, _ in dataset]).mean(0)
std = torch.stack([img.view(img.size(0), -1).std(1) for img, _ in dataset]).mean(0)

print(mean, std)


# For dicom images
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


# Code for creating a black mask for consistency
width, height = 896, 896
corrected_radius = (448**2 + 248**2)**0.5

# Recreating the inverted mask with the corrected radius
inverted_mask_corrected = Image.new('L', (width, height), 'black')  # 'L' mode for grayscale
draw = ImageDraw.Draw(inverted_mask_corrected)
background = Image.new('RGB', (896,896), 'black')
# Drawing a white filled circle onto the black image using the calculated center and corrected radius
draw.ellipse([(width/2 - corrected_radius, height/2 - corrected_radius),
              (width/2 + corrected_radius, height/2 + corrected_radius)], fill='white')



path = ''
file_path = ''


# In the code below, I tested all different types of crops for different images.
if len(path) == 17:
    if path.endswith('dcm'):
        im = dicom_to_pil_image_rgb(file_path)
    else:
        im = Image.open(file_path)
    border = (165, 165, 165, 165)  # left, top, right, bottom
    im = ImageOps.crop(im, border)
    sqrWidth = np.ceil(np.sqrt(im.size[0] * im.size[1])).astype(int)
    im_resize = im.resize((sqrWidth, sqrWidth))
    image = im_resize.resize((896, 896))
    image = Image.composite(image, background, inverted_mask_corrected)
    plt.imshow(image)
    plt.show()
elif path.startswith('000'):
    im = Image.open(file_path)
    border = (430, 200, 400, 200)  # left, top, right, bottom
    im = ImageOps.crop(im, border)
    sqrWidth = np.ceil(np.sqrt(im.size[0] * im.size[1])).astype(int)
    im_resize = im.resize((sqrWidth, sqrWidth))
    image = im_resize.resize((896, 896))
    image = Image.composite(image, background, inverted_mask_corrected)
    plt.imshow(image)
    plt.show()
else:
    im = Image.open(file_path)
    border = (220, 80, 220, 80)  # left, top, right, bottom
    im = ImageOps.crop(im, border)
    sqrWidth = np.ceil(np.sqrt(im.size[0] * im.size[1])).astype(int)
    im_resize = im.resize((sqrWidth, sqrWidth))
    image = im_resize.resize((896, 896))
    image = Image.composite(image, background, inverted_mask_corrected)

    #image = Image.composite(image, Image.new('RGB', image.size, 'black'), draw.convert('L'))
    plt.imshow(image)
    plt.show()