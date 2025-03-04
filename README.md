# Driver Activity Recognition using LSTM

## Introduction
This project implements a **Long Short-Term Memory (LSTM)** model for recognizing driver activities using sensor-based data. The goal is to classify different activities performed by the driver using a sequence of recorded features. The dataset consists of various driving-related actions, and an LSTM neural network is trained to recognize patterns in these sequences.

## Algorithm: Long Short-Term Memory (LSTM)
LSTM is a type of **Recurrent Neural Network (RNN)** designed to process sequential data efficiently. It is particularly useful for time-series and sequence-related tasks as it retains long-term dependencies using **memory cells** and **gates**.

### How It Works:
1. **Input Layer**: The input consists of sensor-based numerical features recorded from driver activity.
2. **Data Preprocessing**:
   - Standardization is applied to normalize the feature values.
   - Labels are encoded into numerical values and converted into one-hot vectors.
   - The data is reshaped to fit the LSTM input format (samples, timesteps, features).
3. **LSTM Layers**:
   - A primary LSTM layer with 50 neurons extracts temporal dependencies.
   - A Dense layer with 50 neurons and ReLU activation enhances feature extraction.
   - A final Dense layer uses a softmax activation function to classify the activities.
4. **Training**: The model is trained using the **Adam optimizer** and **categorical cross-entropy loss function**.
5. **Evaluation**: The model is evaluated using accuracy, precision, recall, and F1-score.

## Performance Metrics
Due to a **limited dataset**, the model exhibits a lower accuracy:
- **Train Accuracy**: 44%
- **Test Accuracy**: 19%
- **Precision**: 20%
- **Recall**: 19%
- **F1-Score**: 19%

### Explanation:
- The model performs well on the training set but generalizes poorly to the test set.
- This could be due to **insufficient data** or an **imbalanced dataset**.
- Further improvements can be made by **augmenting the dataset**, **tuning hyperparameters**, and **using more complex LSTM architectures**.

## Installation & Usage
1. Install dependencies:
   ```bash
   pip install numpy tensorflow scikit-learn
   ```
2. Load the dataset (`driver_activity_train.csv` and `driver_activity_test.csv`).
3. Run the Python script in a Jupyter Notebook or Colab.

## Future Improvements
- Collect more data to improve model generalization.
- Experiment with **Bi-directional LSTMs** and **GRUs**.
- Fine-tune hyperparameters and optimize the model architecture.

This project provides a foundation for driver activity recognition using deep learning. Further enhancements can improve accuracy and robustness for real-world applications!
