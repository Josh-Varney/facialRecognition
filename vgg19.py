from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


def get_vgg19():
    """
    Creates a VGG19-based model for image classification.

    Returns:
        tensorflow.keras.Sequential: VGG19-based model.
    """
    vgg_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # the weights of layers will be updated. common practice when limited data
    for layer in vgg_base.layers:
        layer.trainable = False
        
    model = Sequential()
    model.add(vgg_base)
    # Add a GlobalAveragePooling2D layer instead of Flatten
    model.add(GlobalAveragePooling2D())

    # Dense layers for classification with batch normalization and dropout
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Adding dropout for regularization

    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Adding dropout for regularization

    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Adding dropout for regularization

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def compile_vgg19(model):
    """
    Compiles the provided VGG19-based model for binary classification task.

    Parameters:
        model (tensorflow.keras.Sequential): Model to be compiled.
        
    Returns:
        model (tensorflow.keras.Sequential): Compiled model
    """
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


if __name__ == '__main__':
    model = get_vgg19()
    model = compile_vgg19(model=model)