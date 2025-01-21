import os
import cv2
import torch
import numpy as np
import random
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps
from torchvision import transforms, datasets
from face_features_extraction import *
import torch
import joblib

def crop_images(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    mtcnn = MTCNN(select_largest=False, keep_all=True, device='cpu')

    for folder in os.listdir(input_path):
        class_path = os.path.join(input_path, folder)
        if os.path.isdir(class_path):
            class_output_folder = os.path.join(output_path, class_path)
            if not os.path.exists(class_output_folder):
                os.makedirs(class_output_folder)
            
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                img = cv2.imread(image_path)

                faces,_ = mtcnn.detect(img)
                if faces is not None:
                    selected_face = None
                    min_dist_to_center = float('inf')
                    largest_area = 0
                    center_x, center_y = img.shape[1] // 2, img.shape[0]//2

                    for i, face in enumerate(faces):
                        x1, y1, x2, y2 = map(int, face)
                        face_width = x2 - x1
                        face_height = y2-y1
                        face_area = face_width*face_height

                        face_center_x = (x1 + x2) // 2
                        face_center_y = (y1 + y2) // 2

                        dist_to_center = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)

                        if face_area > largest_area or (face_area == largest_area and dist_to_center < min_dist_to_center):
                            largest_area = face_area
                            selected_face = (x1, y1, x2, y2)
                            min_dist_to_center = dist_to_center
                    
                    if selected_face is not None:
                        x1, y1, x2, y2 = selected_face
                        cropped_face = img[y1:y2, x1:x2]
                        output_filename = os.path.join(class_output_folder, image_file)
                        cv2.imwrite(output_filename, cropped_face)


def histogram_equalization(image):
    img_np = np.array(image)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_y_cr_cb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(img_y_cr_cb)
        y_eq = cv2.equalizeHist(y)
        img_y_cr_cb = cv2.merge((y_eq, cr, cb))
        return Image.fromarray(cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2RGB))
    else:
        return ImageOps.equalize(image)

def gamma_correction(image, gamma_value=1.5):
    return transforms.functional.adjust_gamma(image, gamma=gamma_value, gain=1)

def apply_augmentations(image, num_augmentations=5):
    augmentations = []

    available_augmentations = [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=45),
        lambda img: histogram_equalization(img),
        lambda img: gamma_correction(img, gamma_value=1.5)
    ]

    for _ in range(num_augmentations):
        aug_list = random.sample(available_augmentations, k=random.randint(1, len(available_augmentations)))

        augmented_image = image
        for aug in aug_list:
            augmented_image = aug(augmented_image)
        augmentations.append(augmented_image)
    
    return augmentations

def augment_images_in_folders(image_folder, num_augmented_images_per_image=10):
    for class_folder in os.listdir(image_folder):
        class_path = os.path.join(image_folder, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                img = Image.open(image_path).convert("RGB")

                augmentations = apply_augmentations(img, num_augmentations=num_augmented_images_per_image)

                for i, aug_img in enumerate(augmentations):
                    augmented_image_filename = f"{os.path.splitext(image_file)[0]}_aug_{i+1}.jpg"
                    augmented_image_path = os.path.join(class_path, augmented_image_filename)
                    aug_img.save(augmented_image_path)

def rename_files_in_directory(directory):
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)

        if os.path.isdir(folder_path):
            file_index = 1

            files = sorted(os.listdir(folder_path))

            temp_names = []
            for filename in files:
                file_path = os.path.join(folder_path, filename)


                if os.path.isfile(file_path):
                    temp_filename = f"temp_{file_index}"
                    temp_file_path = os.path.join(folder_path, temp_filename)
    
                    os.rename(file_path, temp_file_path)
                    temp_names.append((temp_file_path, file_index))
                    file_index += 1

            file_index = 1


            for temp_file_path, idx in temp_names:
                file_extension = os.path.splitext(temp_file_path)[1]
                new_filename = f"{folder_name}_{idx}{file_extension}.JPG"
                new_file_path = os.path.join(folder_path, new_filename)
                os.rename(temp_file_path, new_file_path)

def normalise_string(string):
    return string.lower().replace(' ', '_')

def normalise_dict_keys(dictionary):
    new_dict = dict()
    for key in dictionary.keys():
        new_dict[normalise_string(key)] = dictionary[key]
    return new_dict

def dataset_to_embeddings(dataset, model):
    transform = transforms.Compose([
        ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])
    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        _, embedding = model(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            continue
        if embedding.shape[0] > 1:
            embedding = embedding[0,:]
        embeddings.append(embedding.flatten())
        labels.append(label)
    return np.stack(embeddings), labels

def generate_embeddings(input_folder, output_folder):
    torch.set_grad_enabled(False)

    features_model = FacialFeaturesExtractor()
    dataset = datasets.ImageFolder(input_folder)
    embeddings, labels = dataset_to_embeddings(dataset, features_model)

    dataset.class_to_idx = normalise_dict_keys(dataset.class_to_idx)
    idx_to_class = {v:k for k,v in dataset.class_to_idx.items()}
    labels = list(map(lambda idx: idx_to_class[idx], labels))
    
    np.savetxt(output_folder + os.path.sep + 'embeddings.txt', embeddings)
    np.savetxt(output_folder + os.path.sep + 'labels.txt', np.array(labels, dtype=str).reshape(-1, 1), fmt="%s")
    joblib.dump(dataset.class_to_idx, output_folder + os.path.sep + 'class_to_idx.pkl')