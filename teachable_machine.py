import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import pathlib
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('converted_keras/keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# need a loop for it to do multiple pictures
def get_file_list(data_dir, file_type='jpg'):
    """! Creates a list of the files of file_type in data_dir
    @param data_dir The directory to count in
    @param file_type A string containing the extension of the files to count (jpg is default)

    @returns A list of all the files of type file_type in data_dir
    """
    data_dir = pathlib.Path(data_dir)
    img_list = list(data_dir.glob(f'*.{file_type}'))
    # return img_list

    for image_name in img_list:
        # Replace this with the path to your image
        image = Image.open(image_name)

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # display the resized image
        image.show()

        # Normalize the image
        normalized_image_array = (image_array[:,:,0:3].astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        print(prediction)
get_file_list('Test', 'png')