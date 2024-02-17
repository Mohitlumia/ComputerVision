
import cv2
import json
from imutils import paths
import numpy as np

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

########################################################################

from sklearn.model_selection import train_test_split

# split samples and labels for training and testing
test_size = 0.2
train_samples, test_samples, train_labels, test_labels = train_test_split(
    train_images, train_labels, test_size=test_size, random_state=0)

########################################################################

# KNN works on the majority votes from k nearest sample
# k is a hyper-perameter which can be found with max accuracy in Validation Table
# to train the KNN model we will us cv2.ml.KNearest_create() from cv2
# it may take a while depending on how large the dataset is

from datetime import datetime

start_datetime = datetime.now()

knn = cv2.ml.KNearest_create()
knn.train(train_samples, cv2.ml.ROW_SAMPLE, train_labels)

# get different values of K
k_values = [1, 2, 3, 4, 5]
k_result = []
for k in k_values:
    ret,result,neighbours,dist = knn.findNearest(test_samples,k=k)
    k_result.append(result)
flattened = []
for res in k_result:
    flat_result = [item for sublist in res for item in sublist]
    flattened.append(flat_result)

end_datetime = datetime.now()
print('Training Duration: ' + str(end_datetime-start_datetime))

########################################################################

from sklearn.metrics import confusion_matrix

# create an empty list to save accuracy and the cofusion matrix
accuracy_res = []
con_matrix = []

# we will use a loop because we have multiple value of k
for k_res in k_result:
    label_names = [0, 1]
    cmx = confusion_matrix(test_labels, k_res, labels=label_names)
    con_matrix.append(cmx)
    # get values for when we predict accurately
    matches = k_res==test_labels
    correct = np.count_nonzero(matches)
    # calculate accuracy
    accuracy = correct*100.0/result.size
    accuracy_res.append(accuracy)

# stor accuracy for later when we create the graph
res_accuracy = {k_values[i]: accuracy_res[i] for i in range(len(k_values))}
list_res = sorted(res_accuracy.items())

#######################################################################

# get k with max accuracy in validation data
k_best = max(list_res,key=lambda item:item[1])[0]

# and at the end save the KNN model to a file
knn.save('Image Classifier/KNN Classifier/knn_samples.yml')
