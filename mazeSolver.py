import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import matplotlib.pyplot as plt

from imageUpload import select_image
from imageToGrid import process_maze
from imageToGrid import convert_img






def _get_index_of_nonconsecutive_occurrence(index_list):
    list_len = len(index_list)
    
    if list_len < 2:
        return

    for i in range(1, list_len):
        if index_list[i-1] == index_list[i]-1:
            continue
        return i


# completes node when given a predetermined half of it, and zeros array
def _get_single_node_value(node, zeros):
    gap_middle_index = (zeros[0] + zeros[-1])//2
    
    if node[0] == -1:
        node[0] = gap_middle_index 
        return node
    node.append(gap_middle_index)
    return node	

def find_maze_entrances(maze_array: np.ndarray, edge_scan_width: int = 3) -> list:
    maze_array[0][0], maze_array[0][-1], maze_array[-1][0], maze_array[-1][-1] = 1, 1, 1, 1

    top_pixels = maze_array[0]
    bottom_pixels = maze_array[-1]
    left_pixels = np.array([maze_array[i][0] for i in range(len(maze_array))])
    right_pixels = np.array([maze_array[i][-1] for i in range(len(maze_array))])
    last_row_index = len(maze_array) - 1
    start_node, end_node = None, None

    determined_values = {
        id(top_pixels): [0],
        id(bottom_pixels): [last_row_index],
        id(left_pixels): [-1, 0],
        id(right_pixels): [-1, last_row_index]
    }

    for pixels in (top_pixels, bottom_pixels, left_pixels, right_pixels):
        zeros = np.where(pixels == 0)[0]
        if len(zeros) == 0:
            continue

        nonc_index = _get_index_of_nonconsecutive_occurrence(zeros)

        if nonc_index is None:
            if start_node is None:
                start_node = _get_single_node_value(determined_values[id(pixels)], zeros)
                continue
            end_node = _get_single_node_value(determined_values[id(pixels)], zeros)
            break

        start_node = determined_values[id(pixels)].copy()
        end_node = start_node.copy()

        if start_node[0] == -1:
            start_node[0] = (zeros[0] + zeros[:nonc_index][-1]) // 2
            end_node[0] = (zeros[nonc_index] + zeros[-1]) // 2
            continue

        start_node.append((zeros[0] + zeros[:nonc_index][-1]) // 2)
        end_node.append((zeros[nonc_index] + zeros[-1]) // 2)
        break

    return tuple(start_node), tuple(end_node)



def visualize_entrances(maze_array, entrances):
    display_img = np.copy(maze_array).astype(float)
    for y, x in entrances:
        display_img[y, x] = 0.5  # mark entrance in gray for visibility

    plt.imshow(display_img, cmap='gray')
    plt.title("Maze with Detected Entrances")
    plt.axis('off')
    plt.show()


def draw_path_on_grid(grid: np.ndarray, path: list):
    grid_with_path = np.copy(grid)
    for r, c in path:
        grid_with_path[r, c] = 0.5  # use a different value for visualization

    import matplotlib.pyplot as plt
    plt.imshow(grid_with_path, cmap='gray')
    plt.title("Maze with BFS Path")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    path = select_image()
    if path:
        maze_array = process_maze(path)
        grid, res = convert_img(maze_array)


        entrances = find_maze_entrances(grid)


        draw_path_on_grid(grid)

        print("Found entrances at:", entrances)
