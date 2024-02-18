import pandas as pd
from dotenv import dotenv_values
env_config = dotenv_values("../.env")

# Function to construct the new file path
def update_file_path(row):
    # Extract id and tile_number from image_id
    id, tile_number = row['image_id'].split('_')
    
    # Construct the new file path
    file_path = f'{env_config["DATA_DIR"]}/train_thumbnails/{id}/{tile_number}.png'
    
    return file_path
if __name__ == "__main__":
    # Assuming your DataFrame is named traindf
    traindf = pd.read_csv(f'{env_config["DATA_DIR"]}/train_tiles.csv')

    # Apply the function to update the file_path column
    traindf['file_path'] = traindf.apply(update_file_path, axis=1)

    # Save the updated DataFrame to a new CSV file
    traindf.to_csv(f'{env_config["DATA_DIR"]}/train_tiles_updated.csv', index=False)
    print(traindf)
