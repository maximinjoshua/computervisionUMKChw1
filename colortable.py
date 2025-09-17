import glob
import cv2
import os
import random
import numpy as np

def colortable():
    
    classes_used = ["airplane"]
    images_per_class = 40
    color_table_entry_count = 64

    dataset_root = r"dataset\dataset"
    # print(os.path.join(dataset_root, "*"))

    # classes_list = glob.glob(os.path.join(dataset_root, "*"))
    # print(classes_list, "classeslist")

    # loop through each class folder

    color_sampler_array = np.zeros((len(classes_used)*images_per_class*color_table_entry_count, 3))
    offset_start = 0

    for image_class in classes_used:

        all_image_class_paths = glob.glob(os.path.join(dataset_root, image_class, "*"))
        sampled_files = random.sample(all_image_class_paths, 40)

        # loop through each image
        for image_file in sampled_files:
            # print(image_file, "image file")

            # read the image
            image = cv2.imread(image_file)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_image_reshaped = hsv_image.reshape(-1, 3)
            print(hsv_image_reshaped.shape)

            rng = np.random.default_rng()
            sampled_hsv_colors = rng.choice(hsv_image_reshaped, size = 64, axis=0, replace=False)

            print(sampled_hsv_colors.shape, "sampled hsv colors")
            # print(sampled_hsv_colors)

            # print(hsv_image.shape)

            break

colortable()

    