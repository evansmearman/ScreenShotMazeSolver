import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_maze(img_path) -> np.ndarray:
    # Load image in color
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image could not be loaded. Check the path and file format.")

    # Convert to grayscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to black and white
    _, black_white_img = cv2.threshold(grey_img, 127, 255, cv2.THRESH_BINARY)

    # Find rows with black pixels (value 0)
    zero_rows = np.where(black_white_img == 0)[0]
    if zero_rows.size == 0:
        raise ValueError("No black pixels found in image. Check if it's a maze image with visible walls.")

    top_line_index = zero_rows[0]
    bottom_line_index = zero_rows[-1] + 1

    # Find columns with black pixels in the top row
    zero_cols = np.where(black_white_img[top_line_index] == 0)[0]
    if zero_cols.size == 0:
        raise ValueError("No horizontal black line found at the top. Check maze image structure.")

    l_line_index = zero_cols[0]
    r_line_index = zero_cols[-1] + 1

    # Crop the image to the bounding box of the maze
    cropped_black_white_img = black_white_img[top_line_index:bottom_line_index, l_line_index:r_line_index]

    # Display the cropped maze
    plt.imshow(cropped_black_white_img, cmap='gray')
    plt.title('Cropped Maze Image')
    plt.axis('off')
    plt.show()
    
    return cropped_black_white_img



def convert_img(image: np.ndarray) -> np.ndarray | tuple | int:
    num_rows_img = len(image)
    num_cols_img = len(image[0])
    total_px_count = num_rows_img * num_cols_img

    # scaling factor thresholds
    SF_1_THRESH = 10_000  # ~ 100x100
    SF_2_THRESH = 40_000  # ~ 200x200
    SF_3_THRESH = 100_000 # ~ 330x330 
    SF_4_THRESH = 160_000 # ~ 400x400

    if total_px_count < SF_1_THRESH:
        scaling_factor = 1
    elif total_px_count < SF_2_THRESH:
        scaling_factor = 2
    elif total_px_count < SF_3_THRESH:
        scaling_factor = 3
    elif total_px_count < SF_4_THRESH:
        scaling_factor = 4
    else:
        scaling_factor = 5

    resized_image = cv2.resize(image, (num_rows_img//scaling_factor, num_cols_img//scaling_factor))

    num_rows_resized, num_cols_resized = resized_image.shape
    grid = []
    for r in range(num_rows_resized):
        grid.append([])
        for c in range(num_cols_resized):
            grid[r].append(0 if resized_image[r][c] == 255 else 1)

    grid = np.asarray(grid)

    print("HIT", grid)

    return grid, scaling_factor