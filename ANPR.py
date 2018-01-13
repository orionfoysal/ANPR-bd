import numpy as np
import os
from sklearn.externals import joblib
from skimage.io import imread
from skimage import measure, util
from skimage.transform import resize
from skimage.filters import threshold_otsu
from sklearn import neighbors
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import cv2


PLATE_MIN_AREA = 5000
PLATE_MAX_HEIGHT_COEFF = 0.4
PLATE_MAX_WEIDTH_COEFF = 0.7
PLATE_MIN_HEIGHT_COEFF = 0.08
PLATE_MIN_WEIDTH_COEFF = 0.15

CHAR_MAX_AREA = 5000
CHAR_MAX_HEIGHT_COEFF = 0.7
CHAR_MAX_WEIDTH_COEFF = 0.7
CHAR_MIN_HEIGHT_COEFF = 0.05
CHAR_MIN_WEIDTH_COEFF = 0.05

dateTime = datetime.now().strftime("%H_%M_%S_%b_%d_%Y")
sample_image_path = "./Photos/a.jpg"

def imageCapture():
    # Capture Image in grayscale mode. If no camera connceted then load from sample images
    cam = cv2.VideoCapture(0)
    camConnected, img = cam.read()
    

    

    if camConnected:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ######### display the image on window ##############

        cv2.namedWindow("ANPR Test", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ANPR Test", image)
        cv2.waitKey(0)
        cv2.destroyWindow("ANPR Test")
        cv2.imwrite("test_image.jpg", image)
    
    else:
        image = imread(sample_image_path, as_grey=True)


    

    img = cv2.imread('photos/a.jpg',0)
    gray_car_image = img

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(gray_car_image, cmap="gray")
    threshold_value = threshold_otsu(gray_car_image)
    binary_image = gray_car_image > threshold_value #+ 50

    binary_image = cv2.adaptiveThreshold(gray_car_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)/255

    print(binary_image)
    ax2.imshow(binary_image, cmap="gray")
    plt.show()

    # get the image pixels values between 0 and 255. makes more senses to human. we can scale them down
    # during training and prediction
    # gray_image = image * 255

    # threshold_value = threshold_otsu(gray_image)

    # # threshold_value = 220

    # binary_image = gray_image > threshold_value

    # # plot the binary and grayscale images 
    # fig, (ax1, ax2) = plt.subplots(1,2)
    # ax1.imshow(gray_image, cmap="gray")
    # ax2.imshow(binary_image, cmap="gray")
    # plt.show()

    return gray_car_image, binary_image



def cca(gray_image, binary_image):

    # this gets all the connected regions and groups them together. [Connected Component Analysis]
    label_image = measure.label(binary_image)

    # getting the maximum width, height and minimum width and height that a license plate can be. 
    # This helps to narrow down our area of interests
    plate_dimensions = (PLATE_MIN_HEIGHT_COEFF * label_image.shape[0], PLATE_MAX_HEIGHT_COEFF * label_image.shape[0], PLATE_MIN_WEIDTH_COEFF * label_image.shape[1], PLATE_MAX_WEIDTH_COEFF * label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions

    plate_objects_cordinates = []
    plate_like_objects = []

    fig, (ax1) = plt.subplots(1)
    ax1.imshow(gray_image, cmap="gray")
    ##################################################
    oka =[]

    # regionprops creates a list of properties of all the labelled regions
    for region in regionprops(label_image):
        if region.area < PLATE_MIN_AREA:
            #if the region is so small then it's likely not a license plate
            continue

        # the bounding box coordinates

        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col

        #######################################################
        oka.append(region.bbox)

        # ensuring that the region identified satisfies the condition of a typical license plate
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            plate_like_objects.append(binary_image[min_row:max_row,
                                    min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col,
                                                max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
        # let's draw a red rectangle over those regions

    plt.show()
    #################################################################################
    print(oka)

    return plate_like_objects




def segmentation(plate_like_objects):
    # The invert was done so as to convert the black pixel to white pixel and vice 
    print("start")
    print(plate_like_objects[0])
    print("Finish")
    license_plate = np.array(plate_like_objects[0])
    license_plate = license_plate ==1

    license_plate = np.invert(license_plate)

    # license_plate = np.invert(plate_like_objects[0])
    # license_plate = plate_like_objects[0]

    #############################################################################
    
    # im = imread(sample_image_path, as_grey=True)
    # crop_im = im[208:383, 471:2596]
    # crop_im = im[774:1024, 1236:1917]

    # fig, ax2 = plt.subplots(1)
    # ax2.imshow(crop_im)
    # plt.show()



    ############################################################################

    labelled_plate = measure.label(license_plate)

    fig, ax1 = plt.subplots(1)
    ax1.imshow(license_plate, cmap="gray")
    # the next two lines is based on the assumptions that the width of
    # a license plate should be between 5% and 15% of the license plate,
    # and height should be between 35% and 60%
    # this will eliminate some

    # character_dimensions = (0.35*license_plate.shape[0], 0.60*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
    character_dimensions = (CHAR_MIN_HEIGHT_COEFF * license_plate.shape[0], CHAR_MAX_HEIGHT_COEFF * license_plate.shape[0], CHAR_MIN_WEIDTH_COEFF * license_plate.shape[1], CHAR_MAX_WEIDTH_COEFF * license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    counter=0
    column_list = []
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]

            # draw a red bordered rectangle over the character.
            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                        linewidth=2, fill=False)
            ax1.add_patch(rect_border)

            # resize the characters to 20X20 and then append each character into the characters list
            resized_char = resize(roi, (28, 28))
            characters.append(resized_char)

            # plt.figure()
            # plt.imshow(resized_char,cmap='gray')

            # plt.imsave('./data/'+str(x0)+'.png', resized_char, cmap='gray')

            # this is just to keep track of the arrangement of the characters
            column_list.append(x0)

    plt.show()

    return characters, column_list


### Flatten the array, append in a single array and save them in a temporary file

def charImageToArray(characters):
    charArray = []
    for character in characters:
        character = util.invert(character)
        character = character * 255
        
        plt.figure()
        plt.imshow(character, cmap='gray')

        charArray.append(character.flatten())
    # np.savetxt(dateTime+".csv", charArray, delimiter=",", fmt = "%s")
    
    # plt.show()

    return charArray




def predict(characters, column_list):

    #load the model

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(current_dir, 'models/knn/knn.pkl')
    model = joblib.load(model_dir)

    classification_result = []
    plate = ''

    for each_character in characters:
        each_character = each_character.reshape(1, -1)
        result = model.predict(each_character)
        classification_result.append(result)
        plate = plate + str(result)
    
    print(plate)

    # print(classification_result)

    # plate_string = ''
    # for eachPredict in classification_result:
    #     plate_string += eachPredict[0]

    # print(plate_string)

    # it's possible the characters are wrongly arranged
    # since that's a possibility, the column_list will be
    # used to sort the letters in the right order

    # column_list_copy = column_list[:]
    # column_list.sort()
    # rightplate_string = ''
    # for each in column_list:
    #     rightplate_string += plate_string[column_list_copy.index(each)]

    # print(rightplate_string)

    # return rightplate_string



gray_image, binary_image = imageCapture()

plate_like_objects = cca(gray_image, binary_image)

characters, column_list = segmentation(plate_like_objects)


charArray = charImageToArray(characters)

predict(charArray, column_list)
