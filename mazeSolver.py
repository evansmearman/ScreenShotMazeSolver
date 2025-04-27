import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology
from skimage import measure

from imageUpload import select_image




def crop_excess(binary, padding=5):
    # Find coordinates of all white pixels
    white_pixels = np.argwhere(binary == 255)

    if white_pixels.size == 0:
        print("No white pixels found, skipping cropping.")
        return binary  # Nothing to crop

    # Get min and max coords
    (y_min, x_min) = white_pixels.min(axis=0)
    (y_max, x_max) = white_pixels.max(axis=0)

    # Add a small padding (optional)
    y_min = max(y_min - padding, 0)
    x_min = max(x_min - padding, 0)
    y_max = min(y_max + padding, binary.shape[0] - 1)
    x_max = min(x_max + padding, binary.shape[1] - 1)

    # Crop
    cropped = binary[y_min:y_max+1, x_min:x_max+1]

    print(f"Cropping to box: ({y_min}, {x_min}) -> ({y_max}, {x_max})")

    return cropped


def needs_edge_mask(binary, margin=5, white_threshold=0.05):
    height, width = binary.shape
    
    # Extract margins
    top = binary[0:margin, :]
    bottom = binary[-margin:, :]
    left = binary[:, 0:margin]
    right = binary[:, -margin:]

    # Count white pixels (path) in margins
    white_pixels = (
        np.sum(top == 255) +
        np.sum(bottom == 255) +
        np.sum(left == 255) +
        np.sum(right == 255)
    )

    # Total number of edge pixels
    total_edge_pixels = (2 * margin * width) + (2 * margin * height)

    # Calculate white pixel ratio
    white_ratio = white_pixels / total_edge_pixels

    print(f"White ratio on edges: {white_ratio:.3f}")

    # If too much white on the edges, we need to mask
    return white_ratio > white_threshold

def find_entrance(image_path):
    # Load image in grayscale

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image.")
        return None

    img_color = cv2.imread(image_path)
    if img_color is None:
        print("Failed to load color image.")
        return None
    # Binarize the image (invert so paths are white)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                            cv2.THRESH_BINARY_INV, 11, 2)

   # Decide if we need to block edges
    # if needs_edge_mask(binary):
    #     print("Cropping excess from edges...")
    #     binary = crop_excess(binary)
    # else:
    #     print("Edges look clean, no cropping needed.")

    height, width = binary.shape

    # binary[0:10, :] = 1
    # binary[-10:, :] = 1
    # binary[:, 0:10] = 1
    # binary[:, -10:] = 1

    print("No clear entrance found on border. Showing binary image:")
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Maze (Debug View)')
    plt.axis('off')
    plt.show()

    
    labels = measure.label(binary > 0)
    regions = measure.regionprops(labels)

    # Find largest region
    largest_region = max(regions, key=lambda r: r.area)

    # Create a new mask
    binary_cleaned = np.zeros_like(binary)
    for coord in largest_region.coords:
        binary_cleaned[coord[0], coord[1]] = 255


     # Optionally: try to find largest external contour for fallback
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
            # Find contours again on the cleaned image

    #     biggest = max(contours, key=cv2.contourArea)

    #     # Approximate the contour to a polygon
    #     epsilon = 0.02 * cv2.arcLength(biggest, True)
    #     approx = cv2.approxPolyDP(biggest, epsilon, True)

    #     # If it finds 4 corners (good for rectangles), great
    #     if len(approx) == 4:
    #             print("Found 4 corners!")
    #             corners = [tuple(pt[0]) for pt in approx]
    #             print("Corners:", corners)

    #             # Optional: draw the corners on the original image
    #             for corner in corners:
    #                 cv2.circle(img_color, corner, 10, (0, 0, 255), -1)

    #             cv2.imshow('Corners Found', img_color)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #     else:
    #         print(f"Detected {len(approx)} points, not exactly 4. Try adjusting epsilon.")
    # else:
    #     print("No contours found.")

        drawn = cv2.drawContours(np.copy(binary), contours, 0, (255, 255, 255), 5)
        cv2.imshow('Fallback Contours', drawn)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
    
    dc = cv2.drawContours(binary, contours, 1, (0,0,0) , 5)
    cv2.medianBlur(dc, 3)
    cv2.imshow('Contours Part 2', dc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Scan top and bottom borders
    # ke = int(min(img.shape[:2]) / 50)  # Make kernel relative to maze size
    # if ke % 2 == 0: ke += 1  # Make sure it's odd

    ke = 19
    kernel = np.ones((ke, ke), np.uint8)


# Dilation
    dilation = cv2.dilate(np.copy(binary), kernel, iterations=1)

    # REMOVE SMALL OBJECTS in DILATION!
    dilation_bool = dilation > 0
    dilation_cleaned_bool = skimage.morphology.remove_small_objects(dilation_bool, min_size=500)
    dilation_cleaned = (dilation_cleaned_bool * 255).astype(np.uint8)

    # (Optional) Show cleaned dilation
    cv2.namedWindow('Cleaned Dilation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cleaned Dilation', 800, 800)
    cv2.imshow('Cleaned Dilation', dilation_cleaned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    # Remove small objects (small white blobs)
    


    
     # Erosion
    erosion = cv2.erode(dilation_cleaned, kernel, iterations=1)
    cv2.imshow('Erosion', erosion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert closed image to boolean
    closed_bool = erosion > 0

    # Remove small objects (small white blobs)
    cleaned_bool = skimage.morphology.remove_small_objects(closed_bool, min_size=150)
    
    cleaned = (cleaned_bool * 255).astype(np.uint8)

    cv2.imshow('Test', cleaned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # closed = cv2.medianBlur(closed, 3)
    # cv2.imshow('Closed', closed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

        
    # blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

    # Difference between dilation and erosion
    diff = cv2.absdiff(dilation_cleaned, cleaned)
    cv2.imshow('Difference', diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Create mask and apply to color image
    mask_inv = cv2.bitwise_not(diff)
    cv2.imshow('Inverted Mask', mask_inv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply mask to color channels
    res = cv2.bitwise_and(img_color, img_color, mask=mask_inv)
    cv2.imshow('Solved Maze (Masked)', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Optional: Skeletonize (makes paths thin, optional depending on your needs)
    # skeleton = skimage.morphology.skeletonize(closed // 255)  # Convert to 0/1
    # skeleton = (skeleton * 255).astype(np.uint8)

    # cv2.imshow('Skeletonized Maze', skeleton)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Create inverted mask (walls are white, paths are black)
    # mask_inv = cv2.bitwise_not(skeleton)

    # cv2.imshow('Inverted Mask', mask_inv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Apply mask to color image
    # res = cv2.bitwise_and(img_color, img_color, mask=mask_inv)

    # cv2.imshow('Masked Maze Solution', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



if __name__ == "__main__":
    path = select_image()
    if path:
       find_entrance(path)
