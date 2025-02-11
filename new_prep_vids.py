#!/usr/bin/env python
# coding: utf-8

# In[14]:


from typing import Tuple
import argparse
from glob import glob
import os
import random
import cv2
import numpy
import pickle



RAIN_KERNEL = numpy.array([
    [0, 0, 0, 0, 0.05, 0.1, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.05, 0.1, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.09, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.09, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.08, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.08, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.07, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.07, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.06, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.06, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.05, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.05, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.04, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.04, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.03, 0.015, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.03, 0.015, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.01, 0.02, 0.01, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.01, 0.02, 0.01, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.005, 0, 0, 0, 0]])


RAIN_KERNEL_SMALL = numpy.array([
    [0, 0, 0, 0, 0.05, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.05, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.015, 0, 0, 0, 0]])


RAIN_KERNEL_BIG = numpy.array([
    [0, 0, 0, 0, 0.05, 0.1, 0.2, 0.1, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.05, 0.1, 0.2, 0.1, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.09, 0.18, 0.09, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.09, 0.18, 0.09, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.08, 0.16, 0.08, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.08, 0.16, 0.08, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.07, 0.14, 0.07, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.07, 0.14, 0.07, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.06, 0.12, 0.06, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.06, 0.12, 0.06, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.05, 0.1, 0.05, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.05, 0.1, 0.05, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.04, 0.08, 0.04, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.04, 0.08, 0.04, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.03, 0.06, 0.03, 0.015, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.03, 0.06, 0.03, 0.015, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.01, 0.02, 0.04, 0.02, 0.01, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.01, 0.02, 0.04, 0.02, 0.01, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.02, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.02, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.02, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.02, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.0025, 0.005, 0.01, 0.005, 0.0025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.0025, 0.005, 0.01, 0.005, 0.0025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.0025, 0.005, 0.01, 0.005, 0.0025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.0025, 0.005, 0.01, 0.005, 0.0025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.00125, 0.0025, 0.005, 0.0025, 0.00125, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.00125, 0.0025, 0.005, 0.0025, 0.00125, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.00125, 0.0025, 0.005, 0.0025, 0.00125, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.00125, 0.0025, 0.005, 0.0025, 0.00125, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.000625, 0.00125, 0.0025, 0.00125, 0.000625, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.000625, 0.00125, 0.0025, 0.00125, 0.000625, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.000625, 0.00125, 0.0025, 0.00125, 0.000625, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.000625, 0.00125, 0.0025, 0.00125, 0.000625, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.000625, 0.00125, 0.0025, 0.00125, 0.000625, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.000625, 0.00125, 0.0025, 0.00125, 0.000625, 0, 0, 0, 0]])


SNOW_KERNEL = numpy.array([
    [0.1, 0.2, 0.3, 0.2, 0.1],
    [0.2, 0.3, 0.4, 0.3, 0.2],
    [0.3, 0.4, 0.5, 0.4, 0.3],
    [0.2, 0.3, 0.4, 0.3, 0.2],
    [0.1, 0.2, 0.3, 0.2, 0.1]])


SNOW_KERNEL_SMALL = numpy.array([
    [0.1, 0.2, 0.2, 0.1],
    [0.2, 0.3, 0.3, 0.2],
    [0.3, 0.4, 0.4, 0.3],
    [0.2, 0.3 ,0.3, 0.2],
    [0.1, 0.2, 0.2, 0.1]])


SNOW_KERNEL_BIG = numpy.array([
    [0.1, 0.2, 0.3, 0.6, 0.3, 0.2, 0.1],
    [0.2, 0.3, 0.4, 0.8, 0.4, 0.3, 0.2],
    [0.3, 0.4, 0.5, 1.0, 0.5, 0.4, 0.3],
    [0.2, 0.3, 0.4, 0.8, 0.4, 0.3, 0.2],
    [0.1, 0.2, 0.3, 0.6, 0.3, 0.2, 0.1]])


# In[2]:


def update_mask(mask: numpy.ndarray,
                rain_level: float,
                rain_speed: int) -> numpy.ndarray:
    """Generate a new rain mask given the previous one.
    
    Args:
        mask - previous rain mask image.
        rain_level - amount of rain at this time instance.
        rain_speed - how fast the rain should fall.
        
    Returns:
        The new rain mask.
    """
    noise = numpy.random.rand(rain_speed, mask.shape[1])
    new_drops = numpy.zeros((rain_speed, mask.shape[1]), dtype=numpy.uint8)
    new_drops[noise < rain_level] = 255
    new_drops[noise >= rain_level] = 0
    mask[rain_speed:, :] = mask[:-rain_speed, :]
    mask[:rain_speed, :] = new_drops
    return mask


# In[3]:


def apply_mask(img: numpy.ndarray,
               mask: numpy.ndarray,
               type: str) -> numpy.ndarray:
    """Given a rain or snow mask, apply it to the image.
    
    Args:
        img - image to add the mask to.
        mask - single channel mask of rain or snow.
        type - the type of mask to apply
        
    Returns:
        3-channel color image with the mask applied.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    if type == "rain":
        mask = cv2.filter2D(mask, -1, 1.5 * RAIN_KERNEL)
    elif type == "snow":
        mask = cv2.filter2D(mask, -1, SNOW_KERNEL)
    elif type == "small_snow":
        mask = cv2.filter2D(mask, -1, SNOW_KERNEL_SMALL)
    elif type == "big_snow":
        mask = cv2.filter2D(mask, -1, SNOW_KERNEL_BIG)
    elif type == 'small_rain':
        mask = cv2.filter2D(mask, -1, 1.5 * RAIN_KERNEL_SMALL)
    elif type == 'big_rain':
        mask = cv2.filter2D(mask, -1, 1.5 * RAIN_KERNEL_BIG)
    mask = cv2.merge((mask, mask, mask))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGBA)
    mask[:, :, 3] = 128
    img = cv2.add(img, mask)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


# In[4]:


def adjust_brightness(img: numpy.ndarray, level: float) -> numpy.ndarray:
    """Adjust the brightness of an image.
    
    Args:
        img - input image.
        level - how much to change brightness (negative numbers are darker
            and positive numbers are brighter).
            
    Return:
        Image with adjusted brightness.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img[:, :, 1] = cv2.add(img[:, :, 1], level*25.5)
    return cv2.cvtColor(img, cv2.COLOR_HLS2BGR)


# In[5]:


def folder2vid(folder: str, output_d: Tuple[int]) -> None:
    """Create a video using all the pictures in a folder, where the frames
    appear in the order they are read from the folder (usually alphabetical).
    
    Args:
        folder - path to folder to convert.
        output_d - (width x height) desired dimensions of output video.
    """
    dirname, vidname = os.path.split(os.path.normpath(folder))
    vidname = vidname + '.avi'
    writer = cv2.VideoWriter(
        os.path.join(dirname, vidname),
        cv2.VideoWriter_fourcc(*'XVID'),
        20.0,
        output_d)
    for file in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_COLOR)
        img = cv2.resize(img, output_d)
        writer.write(img)
    writer.release()


# In[6]:


def process_folder(folder: str,
                   rain_level: float,
                   rain_speed: int,
                   snow_level: float,
                   snow_speed: int,
                   brightness_level: float,
                   min_level: float,
                   mode: int,
                   output_d: Tuple[int]) -> None:
    """Add OOD with a given rain level and brightness level to every image
    file in a folder of images.

    Args:
        folder - path to folder of images.
        rain_level - amount of rain to add in (bigger = more).
        rain_speed - how fast the rain falls.
        bightness_level - change in image brightness (negative = darker,
            positive = lighter).
        mode - divide the files into 4 segments of equal length:
            1: [OOD, OOD, OOD, OOD]
            2: [ID,  ID,  OOD, OOD]
            3: [ID,  OOD, OOD, ID ]
            4: Select random intensity on the interval (min_level, max_level)
            5: Ramp intensity for all OOD frames
        output_d - (width x height) desired dimensions of output images.
    """
    rain_mask = numpy.zeros(output_d[::-1], dtype=numpy.uint8)
    rain_mask_small = numpy.zeros(output_d[::-1], dtype=numpy.uint8)
    rain_mask_big = numpy.zeros(output_d[::-1], dtype=numpy.uint8)
    imagelist = sorted(glob(folder + "/*.png"))
    frame_count = len(imagelist)
    ood_start = 0
    ood_stop = frame_count
    if mode == 2:
        ood_start = int(frame_count * 0.5)
    elif mode == 3:
        ood_start = int(frame_count * 0.25)
        ood_stop = int(frame_count * 0.75)
    elif mode == 4:
        orig_rain = rain_level
        orig_brightness = brightness_level
        rain_speed = output_d[1]

    elif mode == 6:
        # with rain
        currrnt_dir = os.getcwd()
        imagelist = glob(os.path.join(folder, "*/*.png"))
        random.seed(0)
        random.shuffle(imagelist)
        rain_cap = rain_level
        brightness_cap = brightness_level
        for i in range(-4, 6, 1):
            rain_level = rain_cap * i / 5 if i > 0 else 0.0
            brightness_level = brightness_cap * i / 5 if i > 0 else 0.0
            print(f"rain: {rain_level}, brightness: {brightness_level}")
            start = int(len(imagelist) * (i + 4) / 10)
            end = int(len(imagelist) * (i + 5) / 10)
            for file in imagelist[start:end]:
                new_filename = file.replace("train", f"train_rain{rain_cap}_brightness{brightness_cap}_m{mode}")
                new_dir = os.path.dirname(new_filename)
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir)
                img = cv2.imread(file, cv2.IMREAD_COLOR)
                img = cv2.resize(img, output_d)
                if rain_level > 0.0:
                    rain_mask = update_mask(rain_mask, rain_level, rain_speed)
                    img = apply_mask(img, rain_mask, 'rain')
                    rain_mask_small = update_mask(
                        rain_mask_small,
                        rain_level,
                        rain_speed)
                    img = apply_mask(img, rain_mask_small, 'small_rain')
                    rain_mask_big = update_mask(
                        rain_mask_big,
                        rain_level,
                        rain_speed)
                    img = apply_mask(img, rain_mask_big, 'big_rain')
                if snow_level > 0.0:
                    snow_mask = update_mask(rain_mask, snow_level, snow_speed)
                    img = apply_mask(img, snow_mask, 'snow')
                    snow_mask_small = update_mask(
                        rain_mask_small,
                        snow_level,
                        snow_speed)
                    img = apply_mask(img, snow_mask_small, 'small_snow')
                    snow_mask_big = update_mask(
                        rain_mask_big,
                        snow_level,
                        snow_speed)
                    img = apply_mask(img, snow_mask_big, 'big_snow')
                if brightness_level != 0.0:
                    img = adjust_brightness(img, brightness_level)
                cv2.imwrite(new_filename, img)
        return
    elif mode == 7:
        currrnt_dir = os.getcwd()
        imagelist = glob(os.path.join(folder, "*/*.png"))
        rain_cap = rain_level
        brightness_cap = brightness_level
        for i in range(1, 11):
            rain_level = rain_cap * i / 10
            brightness_level = brightness_cap * i / 10
            print(f"rain: {rain_level}, brightness: {brightness_level}")
            for file in imagelist:
                new_filename = file.replace("val", f"val_rain{rain_level}_brightness{brightness_level}_m{mode}")
                new_dir = os.path.dirname(new_filename)
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir)
                img = cv2.imread(file, cv2.IMREAD_COLOR)
                img = cv2.resize(img, output_d)
                if rain_level > 0.0:
                    rain_mask = update_mask(rain_mask, rain_level, rain_speed)
                    img = apply_mask(img, rain_mask, 'rain')
                    rain_mask_small = update_mask(
                        rain_mask_small,
                        rain_level,
                        rain_speed)
                    img = apply_mask(img, rain_mask_small, 'small_rain')
                    rain_mask_big = update_mask(
                        rain_mask_big,
                        rain_level,
                        rain_speed)
                    img = apply_mask(img, rain_mask_big, 'big_rain')
                if snow_level > 0.0:
                    snow_mask = update_mask(rain_mask, snow_level, snow_speed)
                    img = apply_mask(img, snow_mask, 'snow')
                    snow_mask_small = update_mask(
                        rain_mask_small,
                        snow_level,
                        snow_speed)
                    img = apply_mask(img, snow_mask_small, 'small_snow')
                    snow_mask_big = update_mask(
                        rain_mask_big,
                        snow_level,
                        snow_speed)
                    img = apply_mask(img, snow_mask_big, 'big_snow')
                if brightness_level != 0.0:
                    img = adjust_brightness(img, brightness_level)
                cv2.imwrite(new_filename, img)
        return
    
    elif mode == 8:
        # with rain
        labels = {}
        currrnt_dir = os.getcwd()
        imagelist = glob(os.path.join(folder, "*/*.png"))
        random.seed(0)
        random.shuffle(imagelist)
        rain_cap = rain_level
        brightness_cap = brightness_level
        for i in range(20):
            rain_level = rain_cap * (i-4) / 5 if 5 <= i < 10 else 0.0
            brightness_level = brightness_cap * (i-15) / 5 if 10 <= i < 15 else (brightness_cap * (i-14) / 5 if 15 <= i < 20 else 0.0)
            print(f"rain: {rain_level}, brightness: {brightness_level}")
            start = int(len(imagelist) * (i) / 20)
            end = int(len(imagelist) * (i+1) / 20)
            for file in imagelist[start:end]:
                labels[file] = 'rain' if rain_level != 0 else ('none' if brightness_level == 0 else('dark' if brightness_level < 0 else 'bright'))
                assert "train" in file
                new_filename = file.replace("train", f"train_rain{rain_cap}_brightness{brightness_cap}_m{mode}")
                new_dir = os.path.dirname(new_filename)
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir)
                img = cv2.imread(file, cv2.IMREAD_COLOR)
                img = cv2.resize(img, output_d)
                if rain_level > 0.0:
                    rain_mask = update_mask(rain_mask, rain_level, rain_speed)
                    img = apply_mask(img, rain_mask, 'rain')
                    rain_mask_small = update_mask(
                        rain_mask_small,
                        rain_level,
                        rain_speed)
                    img = apply_mask(img, rain_mask_small, 'small_rain')
                    rain_mask_big = update_mask(
                        rain_mask_big,
                        rain_level,
                        rain_speed)
                    img = apply_mask(img, rain_mask_big, 'big_rain')
                if snow_level > 0.0:
                    snow_mask = update_mask(rain_mask, snow_level, snow_speed)
                    img = apply_mask(img, snow_mask, 'snow')
                    snow_mask_small = update_mask(
                        rain_mask_small,
                        snow_level,
                        snow_speed)
                    img = apply_mask(img, snow_mask_small, 'small_snow')
                    snow_mask_big = update_mask(
                        rain_mask_big,
                        snow_level,
                        snow_speed)
                    img = apply_mask(img, snow_mask_big, 'big_snow')
                if brightness_level != 0.0:
                    img = adjust_brightness(img, brightness_level)
                cv2.imwrite(new_filename, img)
        
        with open('image_labels.pickle', 'wb') as handle:
            pickle.dump(labels, handle)
        return
    

    elif mode == 9:
        # with rain
        labels = {}
        currrnt_dir = os.getcwd()
        imagelist = glob(os.path.join(folder, "*/*.png"))
        random.seed(0)
        random.shuffle(imagelist)
        rain_cap = rain_level
        brightness_cap = brightness_level
        for i in range(20):
            rain_level = rain_cap * (i-4) / 5 if 5 <= i < 10 else 0.0
            brightness_level = brightness_cap * (i-15) / 5 if 10 <= i < 15 else (brightness_cap * (i-14) / 5 if 15 <= i < 20 else 0.0)
            print(f"rain: {rain_level}, brightness: {brightness_level}")
            start = int(len(imagelist) * (i) / 20)
            end = int(len(imagelist) * (i+1) / 20)
            for file in imagelist[start:end]:
                assert "val" in file
                labels[file] = 'rain' if rain_level != 0 else ('none' if brightness_level == 0 else('dark' if brightness_level < 0 else 'bright'))
                new_filename = file.replace("val", f"val_rain{rain_cap}_brightness{brightness_cap}_m{mode}")
                new_dir = os.path.dirname(new_filename)
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir)
                img = cv2.imread(file, cv2.IMREAD_COLOR)
                img = cv2.resize(img, output_d)
                if rain_level > 0.0:
                    rain_mask = update_mask(rain_mask, rain_level, rain_speed)
                    img = apply_mask(img, rain_mask, 'rain')
                    rain_mask_small = update_mask(
                        rain_mask_small,
                        rain_level,
                        rain_speed)
                    img = apply_mask(img, rain_mask_small, 'small_rain')
                    rain_mask_big = update_mask(
                        rain_mask_big,
                        rain_level,
                        rain_speed)
                    img = apply_mask(img, rain_mask_big, 'big_rain')
                if snow_level > 0.0:
                    snow_mask = update_mask(rain_mask, snow_level, snow_speed)
                    img = apply_mask(img, snow_mask, 'snow')
                    snow_mask_small = update_mask(
                        rain_mask_small,
                        snow_level,
                        snow_speed)
                    img = apply_mask(img, snow_mask_small, 'small_snow')
                    snow_mask_big = update_mask(
                        rain_mask_big,
                        snow_level,
                        snow_speed)
                    img = apply_mask(img, snow_mask_big, 'big_snow')
                if brightness_level != 0.0:
                    img = adjust_brightness(img, brightness_level)
                cv2.imwrite(new_filename, img)
        
        with open('image_labels_val.pickle', 'wb') as handle:
            pickle.dump(labels, handle)
        return 

    count = 0
    currrnt_dir = os.getcwd()
    new_dir_name = f"{folder}_rain{rain_level}_brightness{brightness_level}_snow{snow_level}_m{mode}"
    new_dir = os.path.join(currrnt_dir, new_dir_name)
    os.mkdir(new_dir)
    for file in imagelist:
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, output_d)
        if count >= ood_start and count <= ood_stop:
            if rain_level > 0.0:
                if mode == 4:
                    rain_level = random.uniform(min_level, orig_rain)
                rain_mask = update_mask(rain_mask, rain_level, rain_speed)
                img = apply_mask(img, rain_mask, 'rain')
                rain_mask_small = update_mask(
                    rain_mask_small,
                    rain_level,
                    rain_speed)
                img = apply_mask(img, rain_mask_small, 'small_rain')
                rain_mask_big = update_mask(
                    rain_mask_big,
                    rain_level,
                    rain_speed)
                img = apply_mask(img, rain_mask_big, 'big_rain')
            if snow_level > 0.0:
                if mode == 4:
                    snow_level = random.uniform(min_level, orig_rain)
                snow_mask = update_mask(rain_mask, snow_level, snow_speed)
                img = apply_mask(img, snow_mask, 'snow')
                snow_mask_small = update_mask(
                    rain_mask_small,
                    snow_level,
                    snow_speed)
                img = apply_mask(img, snow_mask_small, 'small_snow')
                snow_mask_big = update_mask(
                    rain_mask_big,
                    snow_level,
                    snow_speed)
                img = apply_mask(img, snow_mask_big, 'big_snow')
            if brightness_level != 0.0:
                if mode == 4:
                    brightness_level = random.uniform(
                        min_level,
                        orig_brightness)
                img = adjust_brightness(img, brightness_level)
        image_name = os.path.split(file)[-1]
        new_file = new_dir + '/' + image_name    
        cv2.imwrite(new_file, img)
        count += 1


# In[13]:


def process_video(path: str,
                  rain_level: float,
                  rain_speed: int,
                  brightness_level: float,
                  min_level: float,
                  mode: int,
                  output_d: Tuple[int]) -> None:
    """Add OOD with a given rain level and brightness level to a video.

    Args:
        path - location of video.
        rain_level - amount of rain to add in (bigger = more).
        rain_speed - how fast the rain falls.
        bightness_level - change in image brightness (negative = darker,
            positive = lighter).
        mode - divide the video into 4 segments of equal length:
            1: [OOD, OOD, OOD, OOD]
            2: [ID,  ID,  OOD, OOD]
            3: [ID,  OOD, OOD, ID ]
            4: Randomly select intensity from (min_level, max_level)
            5: Ramp intensity for every OOD frame
        output_d - (width x height) desired dimensions of output video.
    """
    rain_mask = numpy.zeros(output_d[::-1], dtype=numpy.uint8)
    rain_mask_small = numpy.zeros(output_d[::-1], dtype=numpy.uint8)
    rain_mask_big = numpy.zeros(output_d[::-1], dtype=numpy.uint8)
    reader = cv2.VideoCapture(path)
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        f'{path[:-4]}_rain{rain_level}_brightness{brightness_level}'
        f'_m{mode}.avi',
        cv2.VideoWriter_fourcc(*'XVID'),
        20.0,
        output_d)
    ood_start = 0
    ood_stop = frame_count
    if mode == 2:
        ood_start = int(frame_count * 0.5)
        mode = 5
    if mode == 3:
        ood_start = int(frame_count * 0.25)
        ood_stop = int(frame_count * 0.75)
        mode = 5
    if mode == 4 or mode == 5 or mode == 6 or mode == 7:
        orig_rain = rain_level
        orig_brightness = brightness_level
        rain_speed = output_d[1]
    count = 0
    while reader.isOpened():
        ret, frame = reader.read()
        if not ret:
            break
        frame = cv2.resize(frame, output_d)
        if count >= ood_start and count <= ood_stop:
            if rain_level > 0.0:
                if mode == 4:
                    rain_level = random.uniform(min_level, orig_rain)
                if mode == 5 or mode == 6 or mode == 7:
                    rain_level = ((rain_level + orig_rain / 3) % orig_rain)                         + min_level
                rain_mask = update_mask(rain_mask, rain_level, rain_speed)
                frame = apply_mask(frame, rain_mask, 'rain')
                rain_mask_small = update_mask(
                    rain_mask_small,
                    rain_level,
                    rain_speed)
                frame = apply_mask(frame, rain_mask_small, 'small_rain')
                rain_mask_big = update_mask(
                    rain_mask_big,
                    rain_level,
                    rain_speed)
                frame = apply_mask(frame, rain_mask_big, 'big_rain')
            if brightness_level != 0.0:
                if mode == 4:
                    brightness_level = random.uniform(
                        min_level,
                        orig_brightness)
                if mode == 5 or mode == 6 or mode == 7:
                    brightness_level = ((brightness_level + orig_brightness                                          / 3) % orig_brightness) + min_level
                frame = adjust_brightness(frame, brightness_level)
        writer.write(frame)
        count += 1
    reader.release()
    writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Add OOD Rain to image/video.')
    parser.add_argument(
        '--input',
        help='Input video or folder of files. OOD will be added according to '
             'the "mode" argument.')
    parser.add_argument(
        '--rain_level',
        default=0.0,
        type=float,
        help='Rain level to add.')
    parser.add_argument(
        '--rain_speed',
        default=10,
        type=int,
        help='Speed of rainfall.')
    parser.add_argument(
        '--snow_level',
        default=0.0,
        type=float,
        help='Snow level to add.')
    parser.add_argument(
        '--snow_speed',
        default=10,
        type=int,
        help='Speed of snowfall.')
    parser.add_argument(
        '--brightness_level',
        default=0.0,
        type=float,
        help='Adjust brightness level.')
    parser.add_argument(
        '--min_level',
        default=0.0,
        type=float,
        help='Minimum level for generative factor (mode 4 only)')
    parser.add_argument(
        '--folder2vid',
        action='store_true',
        help='Don\'t add OOD, just convert folder to video.')
    parser.add_argument(
        '--output_width',
        default=640,
        type=int,
        help='Width of output images in pixels.')
    parser.add_argument(
        '--output_height',
        default=480,
        type=int,
        help='Height of output image in pixels.')
    parser.add_argument(
        '--mode',
        default=1,
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        help='1: whole video or folder becomes OOD; 2: 1st half of video or '
             'folder remains ID, 2nd half becomes OOD; 3: 1st quarter of '
             'video or folder remains ID, quarters 2 and 3 become OOD, 4th '
             'quarter remains ID, 4: choose a rondom rain/brightness level '
             'between min_level and maximum for each frame in video, 5: ramp '
             'intensity from min_level to max_level for entire video. 6: custom')
    args = parser.parse_args()

    output_shape = (args.output_width, args.output_height)
    if os.path.isdir(args.input):
        if args.folder2vid:
            folder2vid(args.input, output_shape)
        else:
            process_folder(
                args.input,
                args.rain_level,
                args.rain_speed,
                args.snow_level,
                args.snow_speed,
                args.brightness_level,
                args.min_level,
                args.mode,
                output_shape)
    else:
        process_video(
            args.input,
            args.rain_level,
            args.rain_speed,
            args.brightness_level,
            args.min_level,
            args.mode,
            output_shape)






