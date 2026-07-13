import numpy as np
import os
import cv2


def calculate_lacunarity(matrix, box_size):
    """Calculate lacunarity for a given box size"""
    counts = []
    for i in range(matrix.shape[0] - box_size + 1):
        for j in range(matrix.shape[1] - box_size + 1):
            window = matrix[i:i+box_size, j:j+box_size]
            zero_count = np.sum(window == 0)
            counts.append(zero_count)

    counts = np.array(counts)
    if len(counts) == 0:
        return np.nan

    M1 = np.mean(counts)
    M2 = np.mean(counts**2)

    if M1 == 0:
        return np.inf  # Handle division by zero for p=0 case

    return M2 / (M1**2)


dataset_path = r"C:\Users\Lenovo\Documents\INNO-dir\Tumors\Nodular_tumor_CA\test_materials\test_imgs"

image_files = sorted([f for f in os.listdir(dataset_path)])
box_size = 600
results = dict()

for file_name in image_files:
    img_path = os.path.join(dataset_path, file_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.uint32)
    lacunarity = calculate_lacunarity(img, box_size)

    print(f"Файл: {file_name}")
    print("Lacunarity =", lacunarity)

    #show_image(img_path) #вывод изображения

for c in results:
    print(f"{c}:", *results.get(c))

