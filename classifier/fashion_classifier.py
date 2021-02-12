import cv2
import numpy as np

labels = {0: "angle-boot", 1: "bag", 2: "coat", 3: "dress", 4: "pullover",
          5: "sandal", 6: "shirt", 7: "sneaker", 8: "t-shirt", 9: "trouser"}


class FashionClassifier:

    def __init__(self, model_path, scalefactor=1.0, size=(), mean=(), swapRB=False, crop=False, ddepth=cv2.CV_32F):

        self.net = cv2.dnn.readNetFromONNX(model_path)

        self.net.setPreferableBackend(0)
        self.net.setPreferableTarget(0)

        self.scalefactor = scalefactor
        self.size = size
        self.mean = mean
        self.swapRB = swapRB
        self.crop = crop
        self.ddepth = ddepth

    def process_output(self, out):
        out = out.flatten()
        classId = np.argmax(out)
        return labels[classId]

    def predict(self, img):
        blob = cv2.dnn.blobFromImage(
            img, self.scalefactor, self.size, self.mean, self.swapRB, self.crop, self.ddepth)
        self.net.setInput(blob)
        out = self.net.forward()
        return self.process_output(out)
