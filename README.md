# 🧠 CIFAR-10 Classification with TensorFlow Subclassed API

This project demonstrates the use of **TensorFlow's Keras subclassing API** to build and train a custom Convolutional Neural Network (CNN) for image classification on the **CIFAR-10 dataset**.

---

## 📌 Project Overview

The goal is to create a deep learning model capable of classifying 32x32 color images across 10 categories using a custom-defined architecture. Instead of using high-level Keras sequential or functional APIs, this project adopts a **subclassed approach** via `tf.keras.Model`, allowing full control over the forward pass and custom training behavior.

---

## 🧠 Model Architecture

The model is a moderately deep CNN designed for small images, with the following structure:

- **Convolutional Layers**: Two convolutional layers are used to extract spatial features from the input images. The first layer uses 32 filters, and the second uses 64 filters, both with 3x3 kernels and ReLU activation.
  
- **Pooling Layers**: Each convolutional layer is followed by a **MaxPooling layer** that reduces spatial dimensions, helping in generalization and reducing overfitting.

- **Flatten Layer**: Converts the 2D feature maps into a 1D vector to feed into the dense layers.

- **Fully Connected Layer**: A dense layer with 128 neurons and ReLU activation is used to learn higher-level representations.

- **Dropout Layer**: A dropout layer with a 50% rate is applied during training to reduce overfitting.

- **Output Layer**: A final dense layer with 10 output units (one for each CIFAR-10 class), without an activation function (logits are used for loss calculation).

This architecture balances performance and efficiency, suitable for quick experiments and educational purposes.

---

## 🏋️ Training Setup

- **Loss Function**: Sparse Categorical Crossentropy (from logits), appropriate for multi-class classification.
- **Optimizer**: Adam optimizer is used for adaptive learning and efficient convergence.
- **Metrics**: Training and testing accuracy are tracked using TensorFlow’s built-in accuracy metric.
- **Data Pipeline**: The CIFAR-10 dataset is loaded using `tf.keras.datasets`, batched, and shuffled using `tf.data.Dataset` for efficient loading during training.

---

## 📊 Results

After 10 epochs of training, the model typically reaches:

- **Training Accuracy**: 0.7006
- **Test Accuracy**: 0.7018 

Performance may vary slightly based on hyperparameters such as learning rate and batch size.

---

## 📂 Repository Structure

├── model.py # Contains CIFAR10Model definition using subclassed API
├── train.py # Custom training loop using tf.GradientTape
├── utils.py # Optional utility functions for plotting, data prep
├── requirements.txt # Python dependencies
└── README.md # This file

---

## 📚 Dataset

The **CIFAR-10** dataset is a benchmark in computer vision consisting of 60,000 32x32 color images across 10 classes. It is automatically downloaded using TensorFlow’s dataset loader.

---

## ⚙️ Requirements

- Python
- TensorFlow
- NumPy
- Matplotlib (optional, for visualization)

Install all dependencies using:

```bash
pip install -r requirements.txt

```markdown
---

## 🙌 Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Creating Custom Layers with the Keras Subclassing API – Tirendaz Academy](https://tirendazacademy.medium.com/creating-custom-layers-with-the-keras-subclassing-api-c84f6ee4bdaf)
- [Sequential vs Functional vs Subclassing API in TensorFlow – The Deep Hub](https://medium.com/thedeephub/sequential-vs-functional-vs-subclassing-api-in-tensorflow-8bfcfe91859d)

