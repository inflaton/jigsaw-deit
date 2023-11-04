import gc
import os
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
import argparse
import random
import warnings
import models_jigsaw
from PIL import Image
from datasets import build_transform
from torchvision import datasets
from timm.utils import accuracy
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class JigsawTestDataset(Dataset):
    def __init__(self, test_image_folder, transform=None):
        self.test_image_folder = test_image_folder
        self.transform = transform
        self.image_paths = []
        self.load_images()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def load_images(self):
        num_images = 5000
        for k in range(num_images):
            image_filename = f"{k:d}.jpg"
            image_path = os.path.join(self.test_image_folder, image_filename)

            # Check if the image file exists
            if os.path.exists(image_path):
                self.image_paths.append(image_path)

        print(
            f"total images found in folder {self.test_image_folder}: {len(self.image_paths)}"
        )


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="Model name",
    default="jigsaw_base_patch56_336",
)
parser.add_argument(
    "-c",
    "--checkpoint",
    type=str,
    help="Checkpoint to load",
    default="data/checkpoints/best_checkpoint-h2-e-100-min-lr-1-e6.pth",
)
parser.add_argument("-b", "--batch", type=int, help="Batch size", default=32)
parser.add_argument(
    "-i", "--input_size", type=int, help="Input image size", default=336
)
parser.add_argument(
    "-t",
    "--test_image_folder",
    type=str,
    help="Test image folder",
    default="data/cs/test",
)
parser.add_argument(
    "-v",
    "--val_image_folder",
    type=str,
    help="Validation image folder",
    default="data/cs/val",
)
parser.add_argument(
    "-n",
    "--train_image_folder",
    type=str,
    help="Validation image folder",
    default="data/cs/train",
)
parser.add_argument(
    "-r",
    "--result_filename",
    type=str,
    help="Result file name",
    default="data/test.txt",
)

# Parse the arguments
args = parser.parse_args()

model = args.model
batch_size = args.batch
checkpoint = args.checkpoint
test_image_folder = args.test_image_folder
val_image_folder = args.val_image_folder
train_image_folder = args.train_image_folder
result_filename = args.result_filename
num_classes = 50

print(
    "model: ",
    model,
    "\ncheckpoint: ",
    checkpoint,
    "\ntrain_image_folder: ",
    train_image_folder,
    "\nval_image_folder: ",
    val_image_folder,
    "\ntest_image_folder: ",
    test_image_folder,
    "\nbatch: ",
    batch_size,
)

transform = build_transform(False, args)


RANDOM_SEED = 193

# initialising seed for reproducibility
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
seeded_generator = torch.Generator().manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

start_time = time.time()

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")  # Use CPU

print(f"device: {device}")

# initialise model instance
model = models_jigsaw.jigsaw_base_patch56_336(
    mask_ratio=0.0,
    use_jigsaw=True,
    pretrained=False,
    num_classes=num_classes,
    drop_rate=0.0,
    drop_path_rate=0.1,
)
loaded_checkpoint = torch.load(checkpoint, map_location="cpu")
checkpoint_model = loaded_checkpoint["model"]
model.load_state_dict(checkpoint_model)
print("loaded checkpoint:", checkpoint, flush=True)

# transfer over to gpu
model = model.to(device)
model.eval()

with torch.no_grad():
    model_result = []
    all_labels = []

    val_set = datasets.ImageFolder(val_image_folder, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    for my_images, my_labels in val_loader:
        all_labels.extend(my_labels.numpy())

        my_images = my_images.to(device, non_blocking=True)
        my_labels = my_labels.to(device, non_blocking=True, dtype=torch.int64)

        output = model(my_im=my_images)
        model_result.extend(output.sup.cpu().numpy())

    acc1_cls = accuracy(
        torch.from_numpy(np.array(model_result)),
        torch.from_numpy(np.array(all_labels)),
        topk=(1,),
    )[0]

    print(f"acc1_cls: {acc1_cls:.3f}")

    mappings = [0] * num_classes
    for i in range(num_classes):
        model_result = []
        test_image_folder = f"{train_image_folder}/{i}"

        test_set = JigsawTestDataset(
            test_image_folder=test_image_folder, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

        for inputs in test_loader:
            inputs = inputs.to(device)
            model_batch_result = model(my_im=inputs)
            model_result.extend(model_batch_result.sup.cpu().numpy())

        pred = [np.argmax(i) for i in model_result]

        average_pred = np.mean(pred)
        print(f"mean predictions for class {i}: {average_pred:.3f}")

        mappings[int(average_pred)] = i

    print(f"mappings: {mappings}")

    for i in range(num_classes):
        model_result = []
        test_image_folder = f"{train_image_folder}/{i}"

        test_set = JigsawTestDataset(
            test_image_folder=test_image_folder, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

        for inputs in test_loader:
            inputs = inputs.to(device)
            model_batch_result = model(my_im=inputs)
            model_result.extend(model_batch_result.sup.cpu().numpy())

        pred = [mappings[np.argmax(i)] for i in model_result]

        average_pred = np.mean(pred)
        print(f"mean predictions for class {i}: {average_pred:.3f}")

# Calculate time elapsed
end_time = time.time()
time_difference = end_time - start_time
hours, rest = divmod(time_difference, 3600)
minutes, seconds = divmod(rest, 60)
print(
    "Validation is completed in {} hours, {} minutes, {:.3f} seconds".format(
        hours, minutes, seconds
    )
)
