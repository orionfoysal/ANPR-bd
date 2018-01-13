from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import util
import PIL.ImageOps
import os

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
    rescale=None,
    preprocessing_function=None)


filenames = []
numberOfSample = 100

for root, dirs, files in os.walk("./data/original/"):
    print(dirs)
    print(root)
    for filename in files:
        filenames.append(filename)
        #print(filename)
        fileInitial = filename.split(".")[0]
        #print(fileInitial)
        img = load_img(root + filename)

        img = PIL.ImageOps.invert(img)

        x = img_to_array(img)

        x = x.reshape((1,) + x.shape)

        i = 0

        for batch in datagen.flow(x, batch_size=1, save_to_dir='./data/created', save_prefix=fileInitial, save_format='png'):
            i+=1
            if i>numberOfSample:
                break