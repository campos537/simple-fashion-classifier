import cv2
import sys
import os
from classifier.fashion_classifier import FashionClassifier


def main(model_path, img_path):
    fashion_model = FashionClassifier(model_path, 1.0, (32,32), (), True)
    
    if os.path.isdir(img_path):
        for img in os.listdir(img_path):
            image = cv2.imread(img_path+"/"+img)
            print(img, " ", fashion_model.predict(image))
    else:
        image = cv2.imread(img_path)
        image_name = img_path.split("/")[len(img_path.split("/"))-1]
        print(image_name, " ",fashion_model.predict(image))
    
if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print("usage: python main.py path/to/model path/to/imageorfolder")
        exit(0)
    main(sys.argv[1], sys.argv[2])
