
import json
import random
import cv2

# loading a json file with annotations of dog and cat images
jsonFile = open("Image Classifier/dog and cat images and annotation/_annotations.json")
annotations = json.load(jsonFile)

# randomly choosing an image
random_filename = 'Image Classifier/dog and cat images and annotation/' + random.choice(list(annotations["annotations"].keys()))

# reading and resizing to reduce the processing time
image = cv2.imread(random_filename)
image = cv2.resize(image, (32, 32))

# and now show the image
cv2.imshow("Example Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()