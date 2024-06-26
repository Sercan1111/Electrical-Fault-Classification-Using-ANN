# Electrical Fault Detection Classification

## Overview
This project involves the classification of electrical faults using two separate datasets. The first dataset is used for binary classification, while the second dataset is used for multi-class classification. The project utilizes Python libraries such as pandas, scikit-learn, TensorFlow, and Keras Tuner for preprocessing, modeling, and hyperparameter optimization.

## Dataset
- Binary Classification Dataset: This dataset is used to classify whether a fault has occurred or not.
- Multi-Class Classification Dataset: This dataset categorizes faults into different types: LG Fault, LLG Fault, LLL Fault, and LLLG Fault.

## Preprocessing Steps
1. Data Loading:
2. Data Cleaning:
3. Feature Scaling:
4. Fault Type Classification:
5. Train-Test-Validation Split:


## Visualization
- Correlation Matrix: Visualize the correlation between features using a heatmap.
- Feature Importance: Plot the importance of features using Random Forest for both binary and multi-class datasets.
- Evaualate the model's performance 
## Classification Model
-------Binary Classification
-Model Architecture:
Sequential model with multiple convolutional and dense layers.
Uses 'relu' activation for hidden layers and 'sigmoid' activation for the output layer.
Training:
-Compiled with 'adam' optimizer and binary cross-entropy loss.
-Trained using TensorFlow and Keras.
Hyperparameter Tuning:
-Utilized Keras Tuner for optimizing hyperparameters like the number of layers, units per layer, and learning rate.
------Multi-Class Classification
-Model Architecture:
Sequential model with multiple dense layers, batch normalization, and dropout for regularization.
Uses 'relu' activation for hidden layers and 'sigmoid' activation for the output layer.
-Training:
Compiled with 'adam' optimizer and binary cross-entropy loss.
Trained using TensorFlow and Keras with EarlyStopping and ReduceLROnPlateau callbacks.
-Hyperparameter Tuning:
Utilized Keras Tuner for optimizing hyperparameters.

## Files in the Repository
-electrical_fault_detection.py: The main Python script with all the preprocessing, modeling, and evaluation steps.
-requirements.txt: Lists all the dependencies to run the script.
-detect_dataset.csv: The binary classification dataset.
-classData.csv: The multi-class classification dataset.

## Results
--------Binary Classification
-The model achieved high accuracy in detecting whether a fault occurred.
-Visualized the training and validation loss and accuracy over epochs.
--------Multi-Class Classification
-The model was able to classify different fault types with significant accuracy.
-Visualized the training and validation loss and accuracy over epochs.
## Future Work
- Explore more complex models and architectures for improved performance.
- Implement additional data augmentation techniques.
- Integrate the models into a real-time fault detection system.
## Contributions
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.
