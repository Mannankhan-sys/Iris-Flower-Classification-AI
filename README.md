# 🌸 Iris Flower Classification with a Neural Network (Keras + TensorFlow)

A simple yet effective implementation of a **multi-layer perceptron (MLP)** neural network using **Keras** (with a TensorFlow backend) to classify the famous **Iris dataset** into three species based on flower measurements.

---

## 📋 Features

* Loads and processes the classic **Iris dataset**
* One-hot encodes class labels for multi-class classification
* Builds a fully connected feedforward neural network with:

  * Input layer
  * Two hidden layers
  * Output layer with softmax activation
* Compiles, trains, and evaluates the model
* Displays final test accuracy

---

## 🛠️ Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn
* TensorFlow (with Keras)

Install the required libraries using:

```bash
pip install numpy pandas scikit-learn tensorflow
```

---

## 📌 How It Works

1. **Load the Iris dataset:**
   Uses `scikit-learn`'s built-in `load_iris()` function.

2. **Preprocess Data:**

   * One-hot encodes the target labels for multi-class classification.
   * Splits data into training and test sets (80-20 split).

3. **Build Neural Network:**

   * Input layer size: 4 (sepal length, sepal width, petal length, petal width)
   * 1st Hidden layer: 10 neurons, ReLU activation
   * 2nd Hidden layer: 8 neurons, ReLU activation
   * Output layer: 3 neurons (for 3 Iris species), softmax activation

4. **Compile and Train:**

   * Optimizer: Adam
   * Loss: Categorical Crossentropy
   * Metric: Accuracy
   * Runs for 50 epochs with batch size 5.

5. **Evaluate Model:**

   * Tests accuracy on the held-out test data.
   * Prints final accuracy score.

---

## 🚀 How to Run
  Run the script:

```bash
git clone https://github.com/Mannankhan-sys/Iris-Flower-Classification-AI.git
cd Iris-Flower-Classification-AI
pip install numpy pandas scikit-learn tensorflow
python main.py
```

You should see training progress followed by a final test accuracy printout.

---

## 📊 Example Output

```text
Epoch 50/50
24/24 [==============================] - 0s 2ms/step - loss: 0.0992 - accuracy: 0.9750
Test Accuracy: 96.67%
```

---

## 📚 Concepts Covered

* Iris dataset exploration
* Multi-class classification
* One-hot encoding
* Multi-layer perceptron (MLP)
* Categorical crossentropy loss
* Model training and evaluation in Keras

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
