import tensorflow as tf
from tensorflow.keras import layers, regularizers

def make_model(input_shape, num_classes):
    """
    Constructs a custom convolutional neural network model.

    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of classes for classification.

    Returns:
        tensorflow.keras.Sequential: Custom convolutional neural network model.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(64, 3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, 3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, 3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(512, 3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(728, 3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
    ])

    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        model.add(layers.Dense(num_classes, activation=None))

    return model


def compile_custom_model(model):
    """
    Compiles the provided custom model for binary classification task.

    Parameters:
        model (tensorflow.keras.Sequential): Custom model to be compiled.

    Returns:
        tensorflow.keras.Sequential: Compiled custom model.
    """
    # Compile the model
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()
    
    return model


if __name__ == '__main__':
    image_size = (224, 224)  # Update with your image size  
    model = make_model(input_shape=image_size + (3,), num_classes=2)  # Update num_classes if necessary
    model = compile_custom_model(model=model)
