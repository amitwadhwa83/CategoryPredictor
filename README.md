# Excel Classification and Processing with Sentence Embeddings

This repository provides tools for classifying descriptions from Excel files using machine learning models based on sentence embeddings. It includes scripts for training a model, processing Excel files, and a user interface for uploading, processing, and downloading files via Streamlit.

---

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Running the Streamlit Application](#running-the-streamlit-application)
- [Dockerization](#dockerization)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

---

### Overview

This project includes:
- **Model Training**: Train a machine learning model to classify descriptions.
- **Excel Processing**: A script to upload, process, and classify data from Excel files.
- **User Interface**: A Streamlit-based web interface for uploading files and downloading processed results.

### Requirements

- Python 3.7+
- Libraries: Install with `requirements.txt`
  
### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/excel-classification-processing.git
   cd excel-classification-processing

2. **Set Up a Virtual Environment**:
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Required Packages**:
    ```bash
    pip install -r requirements.txt

### Project Structure
    excel-classification-processing/
    │
    ├── app.py                  # Streamlit application file
    ├── excel_processor.py      # Excel processing script
    ├── model_training.py       # Model training script
    ├── requirements.txt        # Required dependencies
    ├── README.md               # Project documentation
    ├── data/
    │   └── your_data.csv       # Example labeled data file
    └── models/
        ├── model.pkl           # Saved classification model
        └── embedding_model.pkl # Saved sentence embedding model

### Data Preparation
The dataset should contain at least two columns:

- Description: The text data that will be classified.
- Category: The label or category for each description.

Save the file as data/your_data.csv and ensure it has the necessary columns.

### Training the Model
The model_training.py script trains a classification model and saves it for future use in the processing script.

    python model_training.py

This will:

- Load and preprocess the data.
- Generate sentence embeddings for each description using sentence-transformers.
- Train a KNeighborsClassifier model.
- Save the model as model.pkl and the sentence embedding model as embedding_model.pkl in the models directory.

### Running the Streamlit Application
The Streamlit application provides a user-friendly interface to upload files, process them, and download results.

1. Run the Application:


    streamlit run app.py
2. Access the Application:

- Open your web browser and navigate to http://localhost:8501.
3. Interface Overview:

- File Upload: Upload an Excel file.
- Process Button: Process the file using the pre-trained model.
- Download: Download the processed file with predictions and confidence scores.

