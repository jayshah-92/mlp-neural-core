# MLP Neural Core 

*A multi-class neural network classifier for handwritten digit recognition constructed with TensorFlow and Keras.*

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8+-yellow)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.0+-orange)

MLP Neural Core is an interactive deep learning implementation designed to classify handwritten digits (0–9). By utilising a three-layer Multilayer Perceptron (MLP) architecture, it processes grayscale handwritten digit images and predicts the corresponding numerical class. It features both a custom-implemented, numerically stable Softmax activation and Keras' sequential modeling pipelines.

---

## 📑 Table of Contents

- [MLP Neural Core](#mlp-neural-core)
  - [Project Description](#project-description)
  - [Features](#features)
  - [Technology Stack](#technology-stack)
  - [Demo and Preview](#demo-and-preview)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [Testing](#testing)
  - [License](#license)
  - [FAQs](#faqs)
  - [Contact Information](#contact-information)

---

## 📝 Project Description

Recognizing handwritten text is a foundational computer vision problem with widespread applications, ranging from automated mail sorting via zip-code recognition to digitizing numbers written on bank checks. This project addresses the multi-class classification problem using a subset of the MNIST dataset consisting of 5,000 training examples. Each digit is represented by a $20 \times 20$ pixel grayscale image, which is unrolled into a 400-dimensional vector. 

MLP Neural Core provides an educational and practical framework to study the mathematics and implementation details of feedforward neural networks. It contrasts a custom NumPy-based Softmax function with TensorFlow's high-performance native implementations, and demonstrates the advantage of using logits for numerical precision during loss computation.

---

## ✨ Features

* **Custom Softmax Implementation**: Includes a mathematically verified, numerically stable softmax function (`my_softmax`) that prevents overflow/underflow issues.
* **Three-Layer MLP Architecture**: Implements a sequential feedforward network with 25 hidden units (Layer 1 with ReLU), 15 hidden units (Layer 2 with ReLU), and 10 linear output units (Layer 3).
* **Numerically Stable Loss Integration**: Uses the `SparseCategoricalCrossentropy` loss function configured with `from_logits=True` to compute losses directly from the linear outputs.
* **Loss & Epoch Tracking**: Compiles and trains the model with the Adam optimizer over 40 epochs while tracking metrics.
* **Prediction Visualizations**: Automatically predicts and visualizes class probabilities using Matplotlib, including error display utilities to analyze misclassified digits.
* **Comprehensive Testing Suite**: Uses validation functions to verify model layout, layer properties, and softmax outputs.

---

## 🛠️ Technology Stack

* **Language**: Python 3.8+
* **Deep Learning Framework**: TensorFlow 2.x, Keras
* **Scientific Computing**: NumPy, SciPy (data ingestion of MATLAB formats)
* **Visualization**: Matplotlib (custom formatting stylesheet `deeplearning.mplstyle`)
* **Interactive Environment**: Jupyter Notebook, ipywidgets

---

## 📸 Demo and Preview

### 1. Common Activation Functions
Visualizes the linear, sigmoid, and Rectified Linear Unit (ReLU) activation functions used across network layers.

### 2. Model Architecture Summary
```text
Model: "my_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 L1 (Dense)                  (None, 25)                10025     
 L2 (Dense)                  (None, 15)                390       
 L3 (Dense)                  (None, 10)                160       
=================================================================
Total params: 10,575 (41.31 KB)
Trainable params: 10,575 (41.31 KB)
Non-trainable params: 0 (0.00 B)
_________________________________________________________________
```

### 3. Training Performance Curve
Plots the cross-entropy loss over 40 training epochs to illustrate learning convergence.

---

## ⚙️ Installation

### Prerequisites
Make sure you have Python 3.8+ installed on your system.

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jayshah-92/mlp-neural-core.git
   cd mlp-neural-core
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install numpy tensorflow scipy matplotlib ipywidgets jupyter
   ```

4. **Launch the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Open [neural-networks.ipynb](file:///Users/jayshah/Downloads/jayshah-92/mlp-neural-core/neural-networks.ipynb) in your browser.

---

## 🚀 Usage

Follow the steps inside [neural-networks.ipynb](file:///Users/jayshah/Downloads/jayshah-92/mlp-neural-core/neural-networks.ipynb) to load data, build, train, and evaluate the neural network.

### Code Snippet: Building the MLP Model
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    tf.keras.Input(shape=(400,)),
    Dense(25, activation='relu', name="L1"),
    Dense(15, activation='relu', name="L2"),
    Dense(10, activation='linear', name="L3")
], name="my_model")
```

### Code Snippet: Compiling and Training
```python
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)
history = model.fit(X, y, epochs=40)
```

### Code Snippet: Inference
```python
import numpy as np

# Predict raw scores (logits)
prediction = model.predict(image.reshape(1, 400))

# Convert logits to probabilities using Softmax
probabilities = tf.nn.softmax(prediction)
predicted_class = np.argmax(probabilities)
```

---

## 🔧 Configuration

No complex configuration files or `.env` settings are required to run this project. However, the styling of plots and interactive widgets relies on local settings:
* **Matplotlib Stylesheet**: Formats graphical outputs via the `deeplearning.mplstyle` file.
* **Random Seed Setting**: Model weights are initialized using `tf.random.set_seed(1234)` to guarantee consistent and reproducible results across runs.

---

## 🧪 Testing

The project includes an automated test suite located in [public_tests.py](file:///Users/jayshah/Downloads/jayshah-92/mlp-neural-core/public_tests.py) to check the implementation of your network layers and softmax functions.

To run the unit tests, execute the corresponding testing cells inside the notebook:
```python
# Test custom softmax activation
test_my_softmax(my_softmax)

# Test model shape and layout configuration
test_model(model, classes=10, input_size=400)
```

**Expected Output:**
```text
All tests passed.
All tests passed!
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](file:///Users/jayshah/Downloads/jayshah-92/mlp-neural-core/LICENSE) file for details.

---

## ❓ FAQs

#### Q: Why does the last layer output linear units (logits) instead of softmax probabilities?
A: For numerical stability during training. Compiling the loss function with `from_logits=True` allows TensorFlow to optimize the categorical crossentropy calculation, avoiding extreme value exponentiation. We apply `tf.nn.softmax` manually during prediction to obtain probabilities.

#### Q: How is the dataset formatted?
A: The dataset contains 5,000 handwritten digit images. The inputs are matrices of shape $5000 \times 400$ (where each row represents a flattened $20 \times 20$ grayscale image), and the labels $y$ are vectors of shape $5000 \times 1$ containing values from 0 to 9.

#### Q: How can I resolve interactive widget plotting issues?
A: Ensure that `ipywidgets` is installed in your virtual environment and check that the `%matplotlib widget` cell magic is activated at the beginning of the notebook.

---

## ✉️ Contact Information

If you have any questions, suggestions, or would like to collaborate, feel free to reach out:
* **Developer**: Jay Shah
* **GitHub Profile**: [@jayshah-92](https://github.com/jayshah-92)
