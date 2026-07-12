import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt


def show_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # для правильных цветов в matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(file_name)
    plt.axis("off")
    plt.show()

def show_MNK(dots, k, b):
    x = dots[:, 0]
    y = dots[:, 1]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="blue", label="Данные")

    x_line = np.linspace(min(x), max(x), 100)
    y_line = k * x_line + b

    plt.plot(x_line, y_line, color="red",
             label=f"y = {k:.3f}x + {b:.3f}")

    plt.xlabel("log2(1/r)")
    plt.ylabel("log2(N(r))")
    plt.title("MNK")
    plt.grid(True)
    plt.legend()

    plt.show()

def DBC(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    img_size = min(img.shape)
    img = img[0:img_size, 0:img_size] #обрезали до квадрата
    if min(img.shape) != max(img.shape): print("!!! Изображение было обрезано до квадратного.")
    s = 2
    s_max = img_size // 2 + (img_size % 2)
    G = 256 #количество оттенков

    dots = []
    while s <= s_max: #для каждого размера s ячейки
        curr_img = np.pad(img, ((0, img_size % s), (0, img_size % s)), constant_values=0) #дополнение нулями, надо найти лучшую модификациюю этого момента
        curr_size = curr_img.shape[0]
        h = math.ceil((s * G) / img_size)
        N_r = 0 #суммарное количество boxes для масштаба r
        for i in range(s, curr_size + 1, s): #для каждой ячейки размера s
            for j in range(s, curr_size + 1, s):
                box = curr_img[i - s:i, j - s:j]
                g_max, g_min = box.max(), box.min()
                k = math.ceil(g_min / h)
                l = math.ceil(g_max / h)
                n_r = l - k + 1 #количество boxes, занимаемое в высоту
                N_r += n_r
        dots.append((math.log2(img_size//s), math.log2(N_r)))
        s += 1

    dots = np.array(dots)

    FD, b = np.polyfit(dots[:, 0], dots[:, 1],1) #МНК
    #show_MNK(dots, FD, b) #вывод графика mnk для dbc
    return FD


dataset_path = r"C:\Users\Lenovo\Documents\INNO-dir\Tumors\Nodular_tumor_CA\test_imgs"

image_files = sorted([f for f in os.listdir(dataset_path)])
results = dict()

for file_name in image_files:
    img_path = os.path.join(dataset_path, file_name)

    fd = DBC(img_path)
    print(f"Файл: {file_name}")
    print("DBC =", fd)
    results.setdefault(file_name[5:8], []).append(float(f"{float(fd):.4f}"))

    #show_image(img_path) #вывод изображения

for c in results:
    print(f"{c}:", *results.get(c))