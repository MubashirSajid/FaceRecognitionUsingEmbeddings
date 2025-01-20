import os
import cv2
import torch
import numpy as np
import random
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps
from torchvision import transforms

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
