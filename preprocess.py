import os
from pathlib import Path
from PIL import Image
import torch


def create_puzzle_image(puzzles, image_path):
    rows, cols = 6, 6
    assert len(puzzles) == rows * cols
    # Define the dimensions of your images
    image_width, image_height = 56, 56

    new_image = Image.new("RGB", (image_width * cols, image_height * rows))

    for i in range(len(puzzles)):
        image = Image.open(puzzles[i]).convert("RGB")
        row, col = int(i / cols), int(i % cols)
        new_image.paste(image, (col * image_width, row * image_height))

    new_image.save(image_path, "JPEG")
    print(f"Puzzle image file created: {image_path}")


# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")  # Use CPU

print("device: ", device)

folders = ["validation", "test"]

for folder in folders:
    # Define the directory where your images are located
    data_dir = f"data/{folder}"
    print("data_dir: ", data_dir)
    path = Path(data_dir)
    if not path.is_dir():
        print(f"Folder {path} doesn't exist - skipping ...", flush=True)
        continue

    dest_dir = f"data/cs/{folder}"
    print("dest_dir: ", dest_dir)

    path = Path(dest_dir)
    if path.is_dir():
        print(f"Folder {path} exists - skipping ...", flush=True)
        continue

    print(f"Folder {path} does not exist - creating a new one ...", flush=True)
    os.makedirs(dest_dir, exist_ok=True)

    # Define the total number of data points and images per data point
    images_per_data_point = 36
    total_data_points = int(len(os.listdir(data_dir)) / images_per_data_point)
    print("total_data_points: ", total_data_points)

    # Loop through each data point
    for data_point_index in range(total_data_points):
        puzzles = []
        # Loop through each image within a data point
        for image_index in range(images_per_data_point):
            # Construct the image file path
            image_filename = f"{data_point_index}_{image_index}.jpg"
            image_path = os.path.join(data_dir, image_filename)

            # Check if the image file exists
            if os.path.exists(image_path):
                puzzles.append(image_path)
            else:
                print(f"file {image_path} not found ...")
                break

        image_filename = f"{data_point_index}.jpg"
        image_path = os.path.join(dest_dir, image_filename)
        create_puzzle_image(puzzles, image_path)

        break
