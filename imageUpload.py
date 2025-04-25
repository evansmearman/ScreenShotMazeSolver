import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import matplotlib.pyplot as plt




def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Maze Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    return file_path



