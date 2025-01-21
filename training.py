from configparser import ConfigParser
from _utils import *

config = ConfigParser()
config.read("config.ini")
enable_crop = eval(config["TRAINING"]["crop"])
enable_augmentations = eval(config["TRAINING"]["apply_augmentation"])
augmentations_value = eval(config["TRAINING"]["number_of_augmentations"])

if enable_crop:
    crop_images("images", "images")

if enable_augmentations:
    augment_images_in_folders(image_folder="images", num_augmented_images_per_image=augmentations_value)

rename_files_in_directory("images")

generate_embeddings("images", "embeddings")