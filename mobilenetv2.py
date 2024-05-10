import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import regularizers

def mobilenetv2():
    """
    Creates and returns a MobileNetV2-based model for image classification.

    Returns:
        tensorflow.keras.Model: MobileNetV2-based model.
    """
    image_size = (224, 224)
    channels = 3
    image_shape = (image_size[0], image_size[1], channels)

    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=image_shape
    )

    # Mark the base model as non-trainable
    base_model.trainable = False

    # L2 Regularisation is applied to penalize large weight values to prevent overfitting
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(rate=0.5),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  
        Dropout(rate=0.5),              
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),   
        Dropout(rate=0.5),               
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),   
        Dropout(rate=0.5),              
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),   
        Dropout(rate=0.5),
        Dense(1, activation='sigmoid')
    ])
    
    return model


def compile_mobile(model):
    """
    Compiles the provided model for binary classification task.

    Parameters:
        model (tensorflow.keras.Model): Model to be compiled.
    """
    # Compile the model
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()
    
    return model


if __name__ == '__main__':
    model = mobilenetv2()
    model = compile_mobile(model=model)
