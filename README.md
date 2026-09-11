# Chest X-Ray Pneumonia Detection

A deep learning application that uses a Convolutional Neural Network (CNN) to classify chest X-ray images as **NORMAL** or **PNEUMONIA**.

The model is trained using the publicly available Chest X-Ray Images (Pneumonia) dataset and deployed as an interactive Streamlit web application.

**Developed by:** Burhan Arshad (Computer Science Student)

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pneumonia-detection-burhan.streamlit.app/)

**Live Application:**
https://pneumonia-detection-burhan.streamlit.app/

## Project Overview

Pneumonia is a respiratory infection that can appear as abnormalities in chest X-ray images. This project demonstrates how a CNN can learn visual patterns from chest X-ray images and perform binary image classification.

The application allows users to:

* Upload a chest X-ray image
* Run the image through the trained CNN
* Receive a NORMAL or PNEUMONIA prediction
* View prediction probabilities
* Test sample X-ray images
* Explore information about the trained model

## Model

The model is a custom Convolutional Neural Network built using TensorFlow/Keras.

### Architecture

The network contains:

* Input layer
* Data augmentation
* Image normalization
* 3 convolutional layers
* Max pooling layers
* Flatten layer
* Dense layer
* Dropout
* Sigmoid output layer

### Input

Images are resized to:

```text
150 × 150 × 3
```

The model accepts RGB images.

### Output

The model performs binary classification:

```text
0 → NORMAL
1 → PNEUMONIA
```

The final layer uses a sigmoid activation function to produce the pneumonia probability.

## Model Performance

Evaluation on the project's test dataset produced approximately:

| Metric              | Score |
| ------------------- | ----: |
| Accuracy            |   88% |
| NORMAL Precision    |   89% |
| NORMAL Recall       |   76% |
| NORMAL F1-Score     |  0.82 |
| PNEUMONIA Precision |   87% |
| PNEUMONIA Recall    |   94% |
| PNEUMONIA F1-Score  |  0.90 |

The model achieved particularly strong recall for the PNEUMONIA class on the project's test dataset.

## Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset available on Kaggle.

The dataset contains pediatric chest X-ray images categorized into:

* NORMAL
* PNEUMONIA

The original dataset contains separate training and testing directories.

The dataset itself is **not included in this repository** because of its size.

## Data Preprocessing

The application performs the following preprocessing before inference:

1. EXIF orientation correction
2. RGB conversion
3. Image resizing to 150 × 150 pixels
4. Conversion to a NumPy array
5. Batch dimension expansion

Pixel normalization is performed inside the saved Keras model using a `Rescaling(1./255)` layer.

Therefore, the Streamlit application does not normalize the image separately.

## Project Structure

```text
pneumonia-detection/
│
├── app.py
├── best_pneumonia_model.keras
├── requirements.txt
├── README.md
├── .gitignore
└── samples/
```

The original dataset directory is intentionally excluded from Git:

```text
chest_xray/
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd pneumonia-detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Deployment

This application can be deployed using Streamlit Community Cloud.

The repository should contain:

```text
app.py
best_pneumonia_model.keras
requirements.txt
README.md
.gitignore
```

The training dataset does not need to be uploaded to the deployment server because the deployed application only requires the trained model for inference.

### Live Deployment

The deployed application is available at:

https://pneumonia-detection-burhan.streamlit.app/

## Important Note About the Model

The reported evaluation metrics are based on the project's test dataset.

Performance on external chest X-ray images may differ because images from different hospitals, datasets, scanners, populations, and preprocessing pipelines can have different characteristics.

Therefore, the model should not be considered a clinically validated diagnostic system.

## Medical Disclaimer

This project is intended for **educational and demonstration purposes only**.

It is not a medical device and should not be used to diagnose, treat, or make clinical decisions about pneumonia or any other medical condition.

Always consult a qualified healthcare professional for medical diagnosis and treatment.

## Technologies

* Python
* TensorFlow
* Keras
* NumPy
* Pillow
* Streamlit
* Convolutional Neural Networks
* Deep Learning
* Computer Vision

## Future Improvements

Potential improvements include:

* Transfer learning using architectures such as EfficientNet or ResNet
* Better handling of class imbalance
* Cross-validation
* External dataset evaluation
* Explainable AI using Grad-CAM
* Confusion matrix visualization
* ROC-AUC analysis
* Improved probability calibration
* Evaluation on images from different datasets
* Model optimization for deployment

## Author

**Burhan**

Computer Science Student | Machine Learning & Deep Learning

This project was developed as part of a practical deep learning and computer vision learning journey.

## License

This project is intended as an educational machine learning project.

Dataset licensing and usage terms should be checked on the original dataset source before redistribution.
