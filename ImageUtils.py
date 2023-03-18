import numpy as np
import random
import cv2
import matplotlib.pyplot as plt


""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:

        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        img_width = 40
        img_height = 40
        image = cv2.resize(image,dsize=(img_height,img_height), interpolation=cv2.INTER_CUBIC)
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        x = random.randint(0,image.shape[1] - 32)
        y = random.randint(0,image.shape[0] - 32)
        img = image[y:y+32,x:x+32]
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        if random.choice([0,1]):
            img = np.flip(img,0)
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    mean = np.mean(img)
    std_of_img = img.std(axis=(0,1,2))
    img = img - mean
    img = img / std_of_img
    ### YOUR CODE HERE

    return img