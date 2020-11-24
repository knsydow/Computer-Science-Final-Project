#!/usr/bin/env python3
"""! @brief TensorFlow demonstration of training off of your own images.
Most of this is from the following links :
- https://www.tensorflow.org/tutorials/load_data/images
- https://www.tensorflow.org/tutorials
@author Kristin and Jessica
@date 2020 November
@copyright MIT
"""

# Includes
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import pathlib
import argparse
import datetime
import matplotlib.pyplot as plt
import doctest
import pdb

# Global Variables


# Functions
def get_file_list(data_dir, file_type='jpg', model='', labels=[]):
    """! Creates a list of the files of file_type in data_dir
    @param data_dir The directory to count in
    @param file_type A string containing the extension of the
    files to count (jpg is default)
    @param model The trained model 
    @param labels The list of labels for the classes

    @returns A list of all the files of type file_type in data_dir
    """
    data_dir = pathlib.Path(data_dir)
    img_list = list(data_dir.glob(f'*.{file_type}'))
    # return img_list

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    class_correct = []
    for image_name in img_list:
        # Replace this with the path to your image
        image = Image.open(image_name)

        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then,
        # cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # display the resized image

        # Normalize the image
        normalized_image_array = (image_array[:, :, 0:3].astype(np.float32) /
                                  127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)

        book_name = labels[prediction.argmax()]
        print(book_name)

        if book_name.lower() in image_name.name.lower():
            class_correct.append(1)
        else:
            class_correct.append(0)
            print(f'{prediction}, {image_name}')
            image.show()
    return sum(class_correct)/len(class_correct)


def get_labels(file_name):
    """! Insert the labels for the names of the classes

    @param file_name name of file that comes into the function

    @return list of class names
    """
    class_names = []
    try:
        with open(file_name, 'r') as f_in:
            for line in f_in:
                [idx, class_name] = line.strip().split(' ')
                print(f'{idx}, {class_name}')
                class_names.append(class_name)
        return class_names
    except FileNotFoundError:
        print('file not found')
        return None


def main():
    """! Main function
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doctest', action='store_true',
                        help='Pass this flag to run doctest on the script')
    parser.add_argument('--test', required=True,
                        help='Directory where test images are located')
    parser.add_argument('--labels', required=True,
                        help='Label categories')
    args = parser.parse_args()  # parse the arguments from the commandline
    start_time = datetime.datetime.now()  # save the script start time

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5')

    if(args.doctest):
        doctest.testmod(verbose=True)  # run the tests in verbose mode

    print("-------------------")
    labels = get_labels(args.labels)
    success = get_file_list(args.test, 'png', model, labels)
    print(f'Accuracy was {success*100}%')
    end_time = datetime.datetime.now()    # save the script end time
    print(f'{__file__} took {end_time - start_time} s to complete')


# This runs if the file is run as a script vs included as a module
if __name__ == '__main__':
    main()
