import numpy as np
import matplotlib.pyplot as plt

def printRandomImages(denoised_train_images, train_labels):
    """
    Prints 5 random images along with their corresponding labels.

    Parameters:
        denoised_train_images (numpy.ndarray): Array containing the denoised training images.
        train_labels (numpy.ndarray): Array containing the labels corresponding to the training images.
    """
    # Generate 5 random indices
    random_indices = np.random.choice(len(denoised_train_images), 5, replace=False)

    # Plot 5 random images
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, idx in enumerate(random_indices):
        axes[i].imshow(denoised_train_images[idx], cmap='gray')  # Assuming grayscale images
        axes[i].set_title(f'Label: {train_labels[idx]}')
        axes[i].axis('off')  # Hide axis
    plt.show()
    
    
def printZeroLabelledImages(denoised_train_images, train_labels):
    """
    Prints 5 random images with label 0.

    Parameters:
        denoised_train_images (numpy.ndarray): Array containing the denoised training images.
        train_labels (numpy.ndarray): Array containing the labels corresponding to the training images.
    """
    indices_label_0 = np.where(train_labels == 0)[0]
    num_samples_to_plot = 5
    random_indices_label_0 = np.random.choice(indices_label_0, num_samples_to_plot, replace=False)

    fig, axes = plt.subplots(1, num_samples_to_plot, figsize=(15, 3))
    for i, idx in enumerate(random_indices_label_0):
        axes[i].imshow(denoised_train_images[idx], cmap='gray')  # Assuming grayscale images
        axes[i].set_title(f'Label: 0')
    plt.show()

def printOneLabelledImages(denoised_train_images, train_labels):
    """
    Prints 5 random images with label 1.

    Parameters:
        denoised_train_images (numpy.ndarray): Array containing the denoised training images.
        train_labels (numpy.ndarray): Array containing the labels corresponding to the training images.
    """
    indices_label_1 = np.where(train_labels == 1)[0]
    num_samples_to_plot = 5  
    random_indices_label_1 = np.random.choice(indices_label_1, num_samples_to_plot, replace=False)

    fig, axes = plt.subplots(1, num_samples_to_plot, figsize=(15, 3))
    for i, idx in enumerate(random_indices_label_1):
        axes[i].imshow(denoised_train_images[idx], cmap='gray')  # Assuming grayscale images
        axes[i].set_title(f'Label: 1')
    plt.show()
    
def getSamples(denoised_train_images, train_labels):
    """
    Samples 1000 images for each class (labels 0 and 1) from the provided data.

    Parameters:
        denoised_train_images (numpy.ndarray): Array containing the denoised training images.
        train_labels (numpy.ndarray): Array containing the labels corresponding to the training images.

    Returns:
        numpy.ndarray: Array containing the sampled images.
        numpy.ndarray: Array containing the corresponding labels for the sampled images.
    """
    indices_label_0 = np.where(train_labels == 0)[0]
    sample_indices_label_0 = np.random.choice(indices_label_0, size=1000, replace=False)
    indices_label_1 = np.where(train_labels == 1)[0]
    sample_indices_label_1 = np.random.choice(indices_label_1, size=1000, replace=False)
    
    sample_train_label_0 = denoised_train_images[sample_indices_label_0]
    sample_train_label_1 = denoised_train_images[sample_indices_label_1]
    sample_labels_label_0 = train_labels[sample_indices_label_0]
    sample_labels_label_1 = train_labels[sample_indices_label_1]

    sample_train = np.concatenate([sample_train_label_0, sample_train_label_1], axis=0)
    sample_labels = np.concatenate([sample_labels_label_0, sample_labels_label_1], axis=0)

    random_indices = np.random.permutation(len(sample_train))
    sample_train = sample_train[random_indices]
    sample_labels = sample_labels[random_indices]
    
    sample_train = sample_train.astype(np.float32)
    sample_train /= 255.0
    
    return sample_train, sample_labels

def printTrainingLabelImages(sample_train, sample_labels):
    """
    Prints 5 random samples from the provided sample_train array along with their corresponding labels.

    Parameters:
        sample_train (numpy.ndarray): Array containing the sampled training images.
        sample_labels (numpy.ndarray): Array containing the labels corresponding to the sampled training images.
    """
    num_samples_to_plot = 5  
    random_indices = np.random.choice(len(sample_train), num_samples_to_plot, replace=False)

    fig, axes = plt.subplots(1, num_samples_to_plot, figsize=(15, 3))
    for i, idx in enumerate(random_indices):
        axes[i].imshow(sample_train[idx], cmap='gray')  # Assuming grayscale images
        axes[i].set_title(f'Label: {sample_labels[idx]}')
        axes[i].axis('off')
    plt.show()
