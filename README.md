# Plant Disease Prediction using ResNet-18 Model

## Project Description

This project aims to build a machine learning model that classifies plant diseases based on images of plant leaves using a ResNet-18 model. The model is trained on a labeled dataset containing images of plant leaves, categorized by disease type. The model learns distinctive patterns from these images to accurately classify diseases and assist in plant health diagnosis.

### Key Features:
- **Dataset**: The dataset consists of images from 10 categories, including both diseased and healthy leaves of the Tomato crop.
- **Input**: The model takes in labeled plant images to train a ResNet-18 model.
- **Output**: After training, the model can predict the disease in new, unseen images. The performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and a confusion matrix.

The project aims to develop a classifier that helps identify plant diseases, offering potential assistance to farmers and agricultural experts in diagnosing plant health problems.

## Dataset

The dataset used for this project is sourced from Kaggle, which contains images of tomato leaves labeled with various disease types, as well as healthy leaves.

- [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/pawanmugalihalli/plantvillagedataset2)

## Steps to Execute the Program

### Method 1: Using Kaggle Notebook

1. Visit the [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/pawanmugalihalli/plantvillagedataset2).
2. Click on **New Notebook** to create a new Kaggle notebook.
3. Navigate to **Settings** -> **Go to Accelerator** and select one of the available GPUs (if available).
4. Copy the Python script from the provided link (in your VPL or repository) and paste it into a new cell in the Kaggle notebook.
5. Run the notebook to start training the model.

### Method 2: Using Kaggle Notebook Link

Alternatively, you can use the Kaggle notebook made public for this project:

1. Open the [Kaggle Notebook for this project](https://www.kaggle.com/code/pawanmugalihalli/resnet50-2).
2. Click **Edit** to open the notebook in edit mode.
3. Run each cell sequentially to train the model and evaluate its performance.

## Dependencies

Ensure you have the following Python packages installed:

- `torch` (PyTorch)
- `torchvision`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `opencv-python`

You can install them using pip:
```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn opencv-python
