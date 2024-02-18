import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from tile_images_config import ARGS
from matplotlib import pyplot as plt

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Config
TILE_SIZE = 1024
BLACK_THRESHOLD = 0.01 # Used to trim black
WHITE_THRESHOLD = 0.97 # Used to trim white
INPUT_DIR = ARGS["data_read_path"]
OUTPUT_DIR = ARGS["data_write_path"]

def split_images_into_tiles(image, tile_size):
    # Get the dimensions of the input image batch
    input_height, input_width, channels = image.shape
    # Calculate the number of tiles in the height and width directions
    num_tiles_height = input_height // tile_size
    num_tiles_width = input_width // tile_size

    # Initialize the output batch
    output_image = np.zeros((num_tiles_height * num_tiles_width, tile_size, tile_size, channels))

    # Split each image in the batch into tiles
    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            h_start = i * tile_size
            h_end = (i + 1) * tile_size
            w_start = j * tile_size
            w_end = (j + 1) * tile_size
            tile = image[h_start:h_end, w_start:w_end, :]
            output_image[i * num_tiles_width + j] = tile

    return output_image

def isAllBlack(image):
    img = np.float32(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.max(img_gray) < BLACK_THRESHOLD


def trim(im):
    """
    Converts the image to grayscale using cv2, then computes binary matrix
    of the pixels that are above a certrain threshold, then takes out the first
    row where a certain percentage of the pixels are above the threshold will
    be the first clip point. Same idea for col, max row, max col.
    """

    img = (np.array(im))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_sums = np.sum(img_gray, axis=1)
    col_sums = np.sum(img_gray, axis=0)
    rows = np.where(np.logical_or(row_sums < img.shape[1] * BLACK_THRESHOLD,
                                  row_sums > img.shape[1] * WHITE_THRESHOLD))[0]
    cols = np.where(np.logical_or(col_sums < img.shape[0] * BLACK_THRESHOLD,
                                  col_sums > img.shape[0] * WHITE_THRESHOLD))[0]
    im_crop = np.delete(img, rows, axis=0)
    im_crop = np.delete(im_crop, cols, axis=1)
    return im_crop

def get_file_path(image_id, folder_name="train_thumbnails"):
    if os.path.exists(os.path.join(INPUT_DIR, folder_name, f"{image_id}_thumbnail.png")):
        return os.path.join(INPUT_DIR, folder_name, f"{image_id}_thumbnail.png")
    else:
        return "NO_THUMBNAIL"
        #return f"{INPUT_DIR}/{image_id}.png"

def get_desired_size(image):
    width, height, _ = image.shape

    num_tiles_height = height // TILE_SIZE
    desired_height = num_tiles_height * TILE_SIZE if num_tiles_height > 0 else TILE_SIZE

    num_tiles_width = width // TILE_SIZE
    desired_width = num_tiles_width * TILE_SIZE if num_tiles_width > 0 else TILE_SIZE
    
    return cv2.resize(image, (desired_width, desired_height))

if __name__ == "__main__":
    traindf = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    traindf['file_path'] = traindf['image_id'].apply(get_file_path)
    tiles_df = pd.DataFrame()

    not_found_thumbnails = 0

    for _, row in tqdm(traindf.iterrows()):
        file_path = row['file_path']
        if file_path == "NO_THUMBNAIL":
            not_found_thumbnails+=1
            continue
        if not os.path.exists(os.path.join(OUTPUT_DIR, "train_thumbnails", str(row['image_id']))):
            os.makedirs(os.path.join(OUTPUT_DIR, "train_thumbnails", str(row['image_id'])))
        original_image = plt.imread(file_path)
        trimmed_image = trim(original_image)
        # Resize to closest multiple of TILE_SIZE
        pre_processed_image = get_desired_size(trimmed_image)
        tiles = split_images_into_tiles(pre_processed_image, TILE_SIZE)
        # Save images
        tile_index = 0
        for tile in tiles:
            if not isAllBlack(tile):
                file_path = os.path.join(OUTPUT_DIR, "train_thumbnails", str(row['image_id']), f"{str(tile_index)}.png")
                tiles_df = pd.concat([tiles_df, pd.DataFrame({
                    "image_id": str(row['image_id']) + f'_{str(tile_index)}',
                    "file_path": file_path
                }, index=[0])], ignore_index=True)

                plt.imsave(file_path, tile)
                tile_index += 1
    tiles_df.to_csv(os.path.join(OUTPUT_DIR, "train_tiles.csv"))
    print(f"Could not find {not_found_thumbnails} thumbnails")


