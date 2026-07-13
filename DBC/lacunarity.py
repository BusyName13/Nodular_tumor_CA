import numpy as np
import os
import cv2


def calculate_lacunarity(matrix, box_size):
    """Calculate lacunarity for a given box size"""
    masses = []
    for i in range(matrix.shape[0] - box_size + 1):
        for j in range(matrix.shape[1] - box_size + 1):
            window = matrix[i:i+box_size, j:j+box_size]
            window_mass = np.sum(window)
            masses.append(window_mass)

    masses = np.array(masses)
    if len(masses) == 0:
        return np.nan

    M1 = np.mean(masses)
    M2 = np.mean(masses**2)

    if M1 == 0:
        return "not stated"  # Handle division by zero for p=0 case

    return M2 / (M1**2)


dataset_path = r"C:\Users\Lenovo\Documents\INNO-dir\Tumors\Nodular_tumor_CA\FBM_png"

image_files = sorted([f for f in os.listdir(dataset_path)])
box_size = 10
results = dict()

for file_name in image_files:
    img_path = os.path.join(dataset_path, file_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float64)
    lacunarity = calculate_lacunarity(img, box_size)

    print(f"Файл: {file_name}")
    print("Lacunarity =", lacunarity)

    #show_image(img_path) #вывод изображения

for c in results:
    print(f"{c}:", *results.get(c))

