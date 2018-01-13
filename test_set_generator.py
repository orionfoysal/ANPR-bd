from skimage.io import imread
from skimage import util
import numpy as np
import os
import pandas as pd

i = False
imageArray = np.asarray([[]])
for root, dirs, files in os.walk("./data/original/"):

    for filename in files:

        fileInitial = filename.split("_")[0]

        image = imread(root + filename, as_grey=True)
        
        image = util.invert(image)

        ## MNIST dataset style. Make the range between 0 to 255

        image = image * 255

        image = image.flatten()

        ## Converted data type to string. As numpy array doesn't support different types of data.##
        ## fileInitial aka level parameters are of string type

        image = image.astype(str)

        image = np.insert(image, 0, fileInitial)

        image = np.asarray([image])

        if i == False:
            imageArray = image
            i = True
        else:
            imageArray = np.concatenate((imageArray ,image))


## Use Panda DataFrame to save into csv file ##
pdImageArray = pd.DataFrame(imageArray)

pdImageArray.columns = ['pix'+str(i-1) for i in range(785)]
pdImageArray.rename(columns={'pix-1': 'Labels'}, inplace=True)
pdImageArray.to_csv("./data/original/original.csv")