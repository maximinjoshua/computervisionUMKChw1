import glob
import cv2
import os
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def colortable(color_table_entry_count):
    classes_used = ["airplane", "airport", "beach", "bridge", "church", "cloud", "desert", "forest", "freeway", "golf_course"]
    images_per_class = 40
    dataset_root = r"dataset\dataset"

    color_sampler_array = np.zeros((len(classes_used)*images_per_class*color_table_entry_count, 3))
    offset_start = 0

    for class_index in range(len(classes_used)):
        all_image_class_paths = glob.glob(os.path.join(dataset_root, classes_used[class_index], "*"))
        sampled_files = random.sample(all_image_class_paths, images_per_class)

        for image_file in sampled_files:
            image = cv2.imread(image_file)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_image_reshaped = hsv_image.reshape(-1, 3)

            rng = np.random.default_rng()
            sampled_hsv_colors = rng.choice(hsv_image_reshaped, size=color_table_entry_count, axis=0, replace=False)

            color_sampler_array[offset_start: offset_start + color_table_entry_count] = sampled_hsv_colors
            offset_start += color_table_entry_count

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(np.float32(color_sampler_array), color_table_entry_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # rows, cols = 8, 8
        # fig, ax = plt.subplots(figsize=(cols, rows))

        # table_img = np.zeros((rows*50, cols*50, 3), dtype=np.uint8)

        # for idx, color in enumerate(centers):
        #     if idx >= rows * cols:
        #         break
        #     r, c = divmod(idx, cols)
        #     table_img[r*50:(r+1)*50, c*50:(c+1)*50] = color

        # ax.imshow(table_img)
        # ax.set_title("Color Table (Centroids)")
        # ax.axis("off")
        # plt.show()

    return centers

def histogram(centers):
    classes_used = ["airplane", "airport", "beach", "bridge", "church", "cloud", "desert", "forest", "freeway", "golf_course"]
    images_per_class = 40
    dataset_root = r"dataset\dataset"

    histogram_array = np.zeros((len(classes_used)*images_per_class, color_table_entry_count))
    output_feature_array = np.zeros(len(classes_used)*images_per_class)

    classes_array = np.array(classes_used).reshape(-1, 1)
    encoder = LabelEncoder()
    label_encoded_output = encoder.fit_transform(classes_array)

    offset = 0
    for class_index in range(len(classes_used)):
        all_image_class_paths = glob.glob(os.path.join(dataset_root, classes_used[class_index], "*"))
        sampled_files = random.sample(all_image_class_paths, images_per_class)

        for image_file in sampled_files:
            image = cv2.imread(image_file)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_image_reshaped = hsv_image.reshape(-1, 3)

            distances = np.linalg.norm(hsv_image_reshaped[:, None, :] - centers[None, :, :], axis=2)
            nearest_color_idx = np.argmin(distances, axis=1)

            histogram_counts, _ = np.histogram(nearest_color_idx, bins=np.arange(len(centers)+1))
            histogram_array[offset] = histogram_counts
            output_feature_array[offset] = label_encoded_output[class_index]
            offset += 1

            # Visualization of histograms
            # Show the average histogram per class
            plt.figure(figsize=(12, 6))
            for class_index, class_name in enumerate(classes_used):
                class_histograms = histogram_array[class_index*images_per_class:(class_index+1)*images_per_class]
                mean_hist = np.mean(class_histograms, axis=0)
                plt.plot(mean_hist, label=class_name)

            plt.title("Average Histogram per Class")
            plt.xlabel("Color Cluster Index")
            plt.ylabel("Frequency")
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.show()

    return histogram_array, output_feature_array, encoder

# Calculate the color table
color_table_entry_count = 128
centers = colortable(color_table_entry_count)
histogram_array, output_feature_array, encoder = histogram(centers)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    histogram_array, output_feature_array, test_size=0.2, random_state=42, stratify=output_feature_array
)

# Train KNN
knn = cv2.ml.KNearest_create()
knn.train(np.float32(X_train), cv2.ml.ROW_SAMPLE, np.float32(y_train))

# Predict on test set
ret, results, neighbours, dist = knn.findNearest(np.float32(X_test), k=3)

# Show prediction for each test image
print("Predictions per test image:")
for i in range(len(X_test)):
    predicted_label = int(results[i][0])
    true_label = int(y_test[i])
    predicted_class_name = encoder.inverse_transform([predicted_label])[0]
    true_class_name = encoder.inverse_transform([true_label])[0]
    correct = "CORRECT" if predicted_label == true_label else "WRONG"
    print(f"Image {i+1}: Predicted = {predicted_class_name}, True = {true_class_name} --> {correct}")

# Compute overall accuracy
accuracy = np.mean(results.flatten() == y_test) * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")
