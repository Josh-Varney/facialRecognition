from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

def plotTrainingHistory(history):
    """
    Plots the training and validation accuracy over epochs during model training.

    Parameters:
        history (object): The training history object returned by the `fit()` method of a Keras model.
    """
    plt.figure(figsize=(8, 8))
    epochs_range = range(1, 21)  # For 20 Epochs
    plt.plot(epochs_range, history.history['accuracy'], label="Training Accuracy")
    plt.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
    plt.axis(ymin=0.4, ymax=1)
    plt.grid()
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])
    plt.show()
    
def getModelStatistics(model, X_test, y_test):
    """
    Evaluates the trained model on the test dataset and calculates various statistics.

    Parameters:
        model (object): The trained Keras model.
        X_test (array-like): Input features of the test dataset.
        y_test (array-like): True labels of the test dataset.
    """
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test Accuracy =', accuracy)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Convert predicted probabilities to binary predictions (0 or 1)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate confusion matrix components
    TP = sum((y_pred_binary == 1) & (y_test == 1)) 
    TN = sum((y_pred_binary == 0) & (y_test == 0))
    FP = sum((y_pred_binary == 1) & (y_test == 0))
    FN = sum((y_pred_binary == 0) & (y_test == 1))

    # Overall confusion matrix components
    TP_total = np.sum(TP)
    TN_total = np.sum(TN)
    FP_total = np.sum(FP)
    FN_total = np.sum(FN)

    print('Overall True Positives (TP) =', TP_total)
    print('Overall True Negatives (TN) =', TN_total)
    print('Overall False Positives (FP) =', FP_total)
    print('Overall False Negatives (FN) =', FN_total)

    # Calculate metrics
    accuracy = (TP_total + TN_total) / (TP_total + TN_total + FP_total + FN_total)
    precision = TP_total / (TP_total + FP_total)
    recall = TP_total / (TP_total + FN_total)
    specificity = TN_total / (TN_total + FP_total)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('Specificity:', specificity)
    print('F1 Score:', f1_score)

def printTestCalculations(X_test, y_test, model):
    """
    Randomly selects a few samples from the test dataset, makes predictions, and visualizes the predictions.

    Parameters:
        X_test (array-like): Input features of the test dataset.
        y_test (array-like): True labels of the test dataset.
        model (object): The trained Keras model.
    """
    num_samples = 5
    sample_indices = random.sample(range(len(X_test)), num_samples)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))  

    for i, index in enumerate(sample_indices):
        image = X_test[index]
        true_label = y_test[index]

        predicted_probability = model.predict(np.expand_dims(image, axis=0))[0]  
        predicted_label = 1 if predicted_probability > 0.5 else 0

        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f"True Label: {true_label}\nPredicted Label: {predicted_label} (Probability: {predicted_probability:.4f})", fontsize=10)

    plt.subplots_adjust(wspace=0.3)  
    plt.tight_layout()
    plt.show()

def printConfusionMatrix(model, y_pred, y_test):
    """
    Computes and plots the confusion matrix based on the model's predictions and true labels of the test dataset.

    Parameters:
        model (object): The trained Keras model.
        y_pred (array-like): Predicted labels or probabilities returned by the model.
        X_test (array-like): Input features of the test dataset.
    """
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def plotPredictionRecall(y_test, y_pred):
    """
    Computes and plots the precision-recall curve based on the model's predictions and true labels of the test dataset.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.show()
    
if __name__ == '__main__':
    pass