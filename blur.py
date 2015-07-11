#!/usr/bin/python
"""Set of functions to blur an entire image that replicates a lens blur."""
import cv2
import numpy as np
import os
import shutil


def make_more_vivid(image):
    """Modify the saturation and value of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    saturation = np.array(saturation * 1.2, dtype=np.uint16)
    saturation = np.array(np.clip(saturation, 0, 255), dtype=np.uint8)

    value = np.array(value * 1.1, dtype=np.uint16)
    value = np.array(np.clip(value, 0, 255), dtype=np.uint8)

    return cv2.cvtColor(cv2.merge((hue, saturation, value)), cv2.COLOR_HSV2BGR)


def read_image(input_dir):
    """Read in an image and provide the image itself, name, and extension."""
    for photo in os.listdir(input_dir):
        print photo,
        name, ext = os.path.splitext(photo)
        image = cv2.imread(input_dir + '/' + photo)
        yield image, name, ext


def clean_folder(directory):
    """Clean out the given directory."""
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def process(image):
    """Given an image process it using the process to replicate a lens blur."""
    print '...bluring image',
    image = make_more_vivid(image)
    image = cv2.bilateralFilter(image, 9, 150, 150)
    image = cv2.blur(image, (15, 15))

    return image


def main():
    """Given the images in a directory blur each of them."""
    input_dir = 'images/original'
    output_dir = 'images/blur'

    clean_folder(output_dir)
    for image, name, ext in read_image(input_dir):
        output = process(image)
        cv2.imwrite(output_dir + '/' + name + ext, output)
        print '...[DONE]'

if __name__ == "__main__":
    main()
