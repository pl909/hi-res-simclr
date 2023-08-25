# This Python file is to change datasets (upsample, downsample, and augment based on an excel file of image names and labels).

# Use for unbalanced datasets.


# Take an excel file and save an extra flipped version of the positive images to a dataset.
'''
import pandas as pd
from PIL import Image

xlsx = pd.ExcelFile('')

df = pd.read_excel(xlsx, 'Sheet1')

for i in range(18083):
    image_value = df.iloc[i, 2]
    label = df.iloc[i,3]
    if label == 1:
        with Image.open(''+ str(image_value) + '.jpg') as img:
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_img.save(''+ str(image_value) + 'copy.jpg')
'''

# Take an excel file and save an equal amount of negative and positive images to a dataset.

'''
import pandas as pd
from PIL import Image
import torchvision.transforms as T

# put in excel file path
xlsx = pd.ExcelFile('')

df = pd.read_excel(xlsx, 'Sheet1')

count = 0

for i in range(18083):
    image_value = df.iloc[i, 2]
    label = df.iloc[i,3]
    if label == 0:
        count = count + 1
        if count <= 9380:
            with Image.open('' + str(image_value) + '.jpg') as img:
                img.save('' + str(image_value)  + '.jpg')
    if label == 1 :
        with Image.open(''+ str(image_value) + '.jpg') as img:
            img.save(''+ str(image_value)  + '.jpg')
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_img.save('' + str(image_value) + 'copy.jpg')

'''

import pandas as pd
from PIL import Image
import torchvision.transforms as T


# This main code below augments the positive images to handle unbalanced datasets.


# Read excel file.

xlsx = pd.ExcelFile('')

df = pd.read_excel(xlsx, 'Sheet1')

count = 0


# Set up augmentations

color_jitter = T.ColorJitter(
            0.8 , 0.8 , 0.8 , 0.2
        )
        # 10% of the image
blur = T.GaussianBlur((3, 3), (0.1, 2.0))

train_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.4),
            T.RandomApply([blur], p=0.2),
            # imagenet stats
])

# Iterates through excel file, sets a limit on amount of negative images and saves 5x augmented images.

for i in range(18083):
    image_value = df.iloc[i, 2]
    label = df.iloc[i,3]
    if label == 0:
        count = count + 1
        if count <= 9380:
            with Image.open('' + str(image_value) + '.jpg') as img:
                img.save('' + str(image_value)  + '.jpg')
    if label == 1 :
        with Image.open(''+ str(image_value) + '.jpg') as img:
            for i in range(5):
                img = train_transform(img)
                img.save('' + str(image_value) + 'copy' + str(i) + '.jpg')

