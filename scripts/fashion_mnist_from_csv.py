import pandas as pd
import numpy as np
import os
import cv2

labels = {0: "t-shirt", 1: "trouser", 2: "pullover", 3: "dress", 4: "coat",
          5: "sandal", 6: "shirt", 7: "sneaker", 8: "bag", 9: "angle-boot"}

# Do some processing on the image and write in the folder


def save_image(img, output_path):
    final_image = np.array(img)
    final_image = final_image.reshape((28, 28, 1))
    final_image = np.uint8(final_image)
    cv2.imwrite(output_path, final_image)


def main():
    # Iterate over the data folder
    for set_ in os.listdir("data/"):
        # Read the csv file found and get the DataFrame with Pandas
        if ".csv" in set_:
            set_type = set_.split('_')[1][:-4]
            set_path = "data/" + set_
            if not os.path.isdir("data/"+set_type):
                os.mkdir("data/"+set_type)
            fashion_set = pd.read_csv(set_path)

            count = 0
            # Iterate over the DataFrame to get each flat image and label
            for img in fashion_set.iterrows():
                flat_img = img[1][1:]
                label = labels[img[1][0]]
                output_path = "data/"+set_type+"/"+label
                img_path = output_path + "/" + \
                    label + "_" + str(count) + ".jpg"
                if not os.path.isdir(output_path):
                    os.mkdir(output_path)
                save_image(flat_img, img_path)
                count += 1


if __name__ == '__main__':
    main()
