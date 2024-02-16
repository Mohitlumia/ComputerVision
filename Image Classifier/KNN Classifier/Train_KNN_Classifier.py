
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



# opencv only identify array of float32 for training
train_images = np.array(train_images).astype('float32')

# and array of integers of shape (lable size, 1)
train_labels = np.array(train_labels)
train_labels = train_labels.astype(int)
train_labels = train_labels.reshape((train_labels.size,1))


# split samples and labels for training and testing
test_size = 0.2
train_samples = train_images[:800]
train_labels = train_labels[:800]
test_samples = train_images[800:]
test_labels = train_labels[800:]

# KNN works on the majority votes from k nearest sample
# k is a hyper-perameter which can be found with max accuracy in Validation Table

# to train the KNN model we will us cv2.ml.KNearest_create() from cv2

knn = cv2.ml.KNearest_create()
knn.train(train_samples, cv2.ml.ROW_SAMPLE, train_labels)

## get different values of K
k_values = [1, 2, 3, 4, 5]
k_result = []
for k in k_values:
    ret,result,neighbours,dist = knn.findNearest(test_samples,k=k)
    k_result.append(result)
flattened = []
for res in k_result:
    flat_result = [item for sublist in res for item in sublist]
    flattened.append(flat_result)

