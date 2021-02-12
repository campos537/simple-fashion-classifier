import cv2
import sys
import os
from classifier.fashion_classifier import FashionClassifier


def main(model_path, img_path):
    fashion_model = FashionClassifier(model_path, 1.0, (32, 32), (), True)

    avg_time = 0.0
    if os.path.isdir(img_path):
        images = os.listdir(img_path)
        folder_size = len(images)
        total_time = 0.0
        for img in images:
            image = cv2.imread(img_path+"/"+img)
            t1 = cv2.TickMeter()
            t1.start()
            print(img, " ", fashion_model.predict(image))
            t1.stop()
            total_time = total_time + t1.getTimeMilli()
        avg_time = total_time / folder_size
    else:
        t1 = cv2.TickMeter()
        t1.start()
        image = cv2.imread(img_path)
        t1.stop()
        avg_time = t1.getTimeMilli()
        image_name = img_path.split("/")[len(img_path.split("/"))-1]
        print(image_name, " ", fashion_model.predict(image))
    print("\n AVG TIME PER IMAGE: %.2f" % avg_time + "ms")


if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print("usage: python main.py path/to/model path/to/imageorfolder")
        exit(0)
    main(sys.argv[1], sys.argv[2])
