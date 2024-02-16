
import cv2
import json
from imutils import paths

# loading a json file with annotations of dog and cat images
folder_path = "Image Classifier/dog and cat images and annotation/"
jsonFile = open(folder_path + "_annotations.json")
annotations = json.load(jsonFile)

image_paths = list(paths.list_images(folder_path))
train_images = []
train_labels = []
class_object = annotations['labels']

for i, path in enumerate(image_paths):

    # read, convert to grayscale and resize to reduce the processing time
    image = cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32))
    flat_image = image.flatten()

    #label image using the annotations
    temp_label = annotations["annotations"][path[len(folder_path):]][0]['label']
    label = class_object.index(temp_label)
    
    #Append flattened image and label
    train_images.append(flat_image)
    train_labels.append(label)
    print('Loaded...', 'Image', str(i+1), 'is a', temp_label)

