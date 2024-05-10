import random
import shutil
import os
from PIL import Image
import PIL
import numpy as np
import cv2

# Mafa Obtain Train Tests
def mafa_label_obtain_train(label_df):
    """
    Obtain training data labels from the MAFA dataset.

    Args:
        label_df (list): List of label data.

    Returns:
        list: List of tuples containing image filenames and corresponding labels.
    """
    label_list = []

    for arr in label_df:
        for label_data in arr:
            image_filename = label_data[1][0]
            occluder_type = label_data[2][0][12]
            occluder_degree = label_data[2][0][13]

            if occluder_type == -1 or (occluder_type == 3 and occluder_degree > 0):
                continue

            contains_mask = 1 if occluder_type in [1, 2] else 0
            label_list.append((image_filename, contains_mask))

    return label_list

# Mafa Obtain Tests
def mafa_label_obtain_test(label_df):
    """
    Obtain test data labels from the MAFA dataset.

    Args:
        label_df (list): List of label data.

    Returns:
        list: List of tuples containing image filenames and corresponding labels.
    """
    label_list = []

    for label in label_df:
        for data in label:
            image_file_name = data[0][0]
            masked_value = data[1][0][4]

            if masked_value != 1 or masked_value != 1.0:
                label_list.append((f'/Users/josh-v/Downloads/images/{image_file_name}', 0))
            else:
                label_list.append((f'/Users/josh-v/Downloads/images/{image_file_name}', 1))

    random.shuffle(label_list)

    return label_list

# Widerface Test Obtain
def widerface_obtain_test(directory):
    """
    Obtain test data labels from the Widerface dataset.

    Args:
        directory (str): Directory containing the test images.

    Returns:
        list: List of tuples containing image filenames and corresponding labels.
    """
    label_list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            label_list.append((file_path, 0))

    random.shuffle(label_list)

    return label_list

# Seperation of Masked and Unmasked Images
def copy_files_to_folder(file_list, source_folder, masked_face_folder, unmasked_face_folder):
    """
    Copy files from source folder to destination folders based on their labels.

    Args:
        file_list (list): List of tuples containing filenames and labels.
        source_folder (str): Path to the source folder.
        masked_face_folder (str): Path to the folder for masked face images.
        unmasked_face_folder (str): Path to the folder for unmasked face images.

    Returns:
        None
    """
    if not os.path.exists(source_folder):
        print('Source Folder does not Exist')
    elif not os.path.exists(masked_face_folder) or not os.path.exists(unmasked_face_folder):
        print('Destination Folder does not Exist')
    else:
        for file_name, masked_value in file_list:
            source_path = os.path.join(source_folder, file_name)
            if masked_value == 1:
                destination_path = os.path.join(masked_face_folder, file_name)
            elif masked_value == 0:
                destination_path = os.path.join(unmasked_face_folder, file_name)
            shutil.copy(source_path, destination_path)

# Wider Face Unmasked Image Obtain
def wider_train_get_unmasked(directory):
    """
    Obtain unmasked images from the Widerface dataset.

    Args:
        directory (str): Directory containing the unmasked images.

    Returns:
        list: List of paths to unmasked image files.
    """
    image_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))

    return image_files

# Push WiderFace Unmasked to Unmasked Folder
def widerface_to_unmasked_folder(destination_directory, widerface_changed_file_names):
    """
    Copy unmasked images to the destination directory.

    Args:
        destination_directory (str): Path to the destination directory.
        widerface_changed_file_names (list): List of paths to unmasked image files.

    Returns:
        None
    """
    for image_file in widerface_changed_file_names:
        filename = os.path.basename(image_file)
        shutil.copy(image_file, os.path.join(destination_directory, filename))

# Pull Images from Unmasked, Masked Folder, Resize, RGB Conversion and Append to Numpy Array
def load_images_from_folder_resize_train(folder, limit=None, shuffle=True):
    """
    Load and preprocess images from a folder for training.

    Args:
        folder (str): Path to the folder containing the images.
        limit (int, optional): Maximum number of images to load. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the images. Defaults to True.

    Returns:
        list: List of preprocessed image arrays.
    """
    training_images = []
    count = 0

    try:
        filenames = os.listdir(folder)
        if shuffle:
            random.shuffle(filenames)
        for filename in filenames:
            if limit is not None and count >= limit:
                break
            try:
                img = Image.open(os.path.join(folder, filename))
                img = img.resize((224, 224))
                img = img.convert('RGB')
                training_images.append(np.array(img))
                count += 1
            except (FileNotFoundError, PIL.UnidentifiedImageError) as e:
                print(f"Error loading image {filename}: {e}")
    except Exception as e:
        print(f"Error loading images: {e}")
        
    return training_images

# Test Images resizing, conversion, appended to array
def test_image_resize_preprocess(test_image_paths, limit=None):
    """
    Load and preprocess test images.

    Args:
        test_image_paths (list): List of paths to test images.
        limit (int, optional): Maximum number of images to load. Defaults to None.

    Returns:
        list: List of preprocessed test image arrays.
    """
    test_images = []
    count = 0

    for path in test_image_paths:
        if limit is not None and count >= limit:
            break
        try:
            img = Image.open(path)
            img = img.resize((224, 224))
            img = img.convert('RGB')
            test_images.append(np.array(img))
            count += 1
        except (FileNotFoundError, PIL.UnidentifiedImageError) as e:
            print(f"Error loading image {path}: {e}")

    return test_images

# Denoise Training Images
def denoise_images(train_images):
    """
    Denoise training images using Gaussian blur.

    Args:
        train_images (list): List of input training images.

    Returns:
        list: List of denoised training images.
    """
    denoised_images = []
    
    for image in train_images:
        denoised_image = cv2.GaussianBlur(image, (3, 3), 0)
        denoised_images.append(denoised_image)
        
    return denoised_images
