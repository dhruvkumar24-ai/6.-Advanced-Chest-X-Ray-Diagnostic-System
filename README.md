# XRay-Vision

# CXR-Net: Multi-Class Thoracic Disease Classification

This project is a complete deep learning pipeline for classifying 14 different thoracic pathologies from chest X-ray images, built using Python and TensorFlow/Keras. The model is a custom-built Convolutional Neural Network (CNN) designed for multi-label medical image analysis.

The entire workflow, from local environment setup to data preprocessing and model training, is documented to showcase a realistic end-to-end development process.

---
## 📋 Key Features

* **Multi-Label Classification:** The core model is designed to detect the presence of one or more of 14 thoracic diseases (e.g., Pneumonia, Effusion, Cardiomegaly) from a single CXR image.
* **Memory-Efficient Data Pipeline:** Utilizes Keras `ImageDataGenerator` and `flow_from_dataframe` to load and augment thousands of images in batches, allowing the model to be trained on a local machine with limited RAM.
* **Patient-Aware Data Splitting:** Implements a robust data splitting method that separates patients between the training, validation, and test sets to prevent data leakage and ensure the model generalizes to unseen patient data.
* **Class Imbalance Handling:** Employs a custom sample weighting strategy to address the significant class imbalance inherent in the medical dataset, helping the model learn from underrepresented disease classes.
* **End-to-End Local Development:** The project includes a detailed history of setting up a local development environment on Windows, from managing a `venv` and installing dependencies to troubleshooting OS-level permission errors with third-party security software (Kaspersky).

---
## 🛠️ Tech Stack

* **Core Libraries:** TensorFlow, Keras, Pandas, Scikit-learn
* **Data Visualization:** Matplotlib, Seaborn
* **Image Processing:** OpenCV, Scikit-image
* **Development Environment:** Python, Jupyter Notebook, VS Code

---
## 💾 Dataset

The model is trained on a 5,000-image subset of the **NIH ChestX-ray14 dataset**. The data was acquired via the Kaggle API, and scripts were developed to perform random sampling to create a manageable yet representative dataset for local training.
