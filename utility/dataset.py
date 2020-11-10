import os
import glob
import math
import numpy as np
from PIL import Image
from .readPatient import PatientData
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2


def RandomTransform(img, mask):
    finalImg = np.zeros((img.shape[0], 192, 192, 1))
    finalMask = np.zeros((img.shape[0], 192, 192, 2))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    for imgNo in range(img.shape[0]):
        heart = img[imgNo, :, :, 0]
        heart = clahe.apply(heart)
        heart = Image.fromarray(heart)
        reqMask1 = Image.fromarray(mask[imgNo, :, :, 0])
        reqMask2 = Image.fromarray(mask[imgNo, :, :, 1])

        # Random rotate
        if random.random()>0.5:
            angle = random.randint(-180, 180) 
            heart = TF.rotate(heart, angle)
            reqMask1 = TF.rotate(reqMask1, angle)
            reqMask2 = TF.rotate(reqMask2, angle)

        # Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(heart, output_size=(192, 192))
        heart = TF.crop(heart, i, j, h, w)
        reqMask1 = TF.crop(reqMask1, i, j, h, w)
        reqMask2 = TF.crop(reqMask2, i, j, h, w)

        heart = np.array(heart)
        reqMask1 = np.array(reqMask1)
        reqMask2 = np.array(reqMask2)
        finalImg[imgNo, :, :, 0] = heart
        finalMask[imgNo, :, :, 0] = reqMask1
        finalMask[imgNo, :, :, 1] = reqMask2

    return finalImg, finalMask


def load_images(dataDir):
    globSearch = os.path.join(dataDir, "patient*")
    patientdirs = sorted(glob.glob(globSearch))
    if len(patientdirs)==0:
        raise Exception(f"No patient directory at {dataDir}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    images = []
    innerMask = []
    outerMask = []
    for patientDir in patientdirs:
        p = PatientData(patientDir)
        images.extend(p.images)
        innerMask.extend(p.endocardium_masks)
        outerMask.extend(p.epicardium_masks)

    images = np.asarray(images)[:, :, :, None]
    masks = np.asarray(innerMask) + np.asarray(outerMask)
    masksChannel = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 2))
    for i in range(masksChannel.shape[0]):
        masksChannel[i, :, :, 0] = masks[i, :, :]==1
        masksChannel[i, :, :, 1] = masks[i, :, :]==2
        images[i, :, :, 0] = clahe.apply(images[i, :, :, 0])

    # plt.imshow(masksChannel[3, :, :, 0])
    # plt.show()
    # plt.imshow(masksChannel[3, :, :, 1])
    # plt.show()
    
    return images, masksChannel


class DataLoader:
    def __init__(self, dataDir, batchSize, shuffle=True):
        self.dataDir = dataDir
        self.batchSize = batchSize
        self.shuffle = shuffle

        self.images, self.masks = load_images(self.dataDir)

    def __len__(self):
        return math.ceil(self.images.shape[0]/self.batchSize)

    def __getitem__(self, index):
        start = index*self.batchSize
        end = min(self.images.shape[0], (index+1)*self.batchSize)

        images = self.images[start:end, :, :, :]
        masks = self.masks[start:end, :, :, :]
        img, msk = RandomTransform(images, masks)

        return img, msk
