import cv2
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide root Tkinter window
Tk().withdraw()

# Open file dialog
image_path = askopenfilename(
    title="Select Medical Image",
    filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
)

if not image_path:
    print("No file selected!")
    exit()

# Read image in grayscale
img = cv2.imread(image_path, 0)

if img is None:
    print("Image not found!")
    exit()

# 1️⃣ Median Filter
median = cv2.medianBlur(img, 3)

# 2️⃣ CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(median)

# 3️⃣ Sharpening
kernel = np.array([[0,-1,0],
                   [-1,5,-1],
                   [0,-1,0]])

sharpened = cv2.filter2D(clahe_img, -1, kernel)

# Display Input vs Output
titles = ['Input Image', 'Enhanced Output']
images = [img, sharpened]

plt.figure(figsize=(10,5))

for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()