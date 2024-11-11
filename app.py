import base64
import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
import sys
import os


# Load the pre-trained model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/embedding_model.pkl")

logging.basicConfig(stream=sys.stdout, level=logging. INFO, format='%(asctime)s %(levelname)s:%(message)s')
logging.info("Model loaded...")

# Function to process the uploaded file
def process_file(df):
    if "Description" not in df.columns:
        st.error("Uploaded file does not contain 'Description' column.")
        return None
    logging.info("Description ")
    # Preprocess data (this is a placeholder, adjust as per your actual preprocessing function)
    df['cleaned_description'] = df['Description'].apply(lambda x: x.lower().strip())

    # Transform descriptions to embeddings
    descriptions = df['cleaned_description']
    embeddings = vectorizer.transform(descriptions)

    # Get predictions and confidence scores
    predictions = model.predict(embeddings)
    confidence_scores = model.predict_proba(embeddings).max(axis=1)

    # Save predictions and confidence scores in the dataframe
    df['Predicted_Profile'] = predictions
    df['Confidence_Score'] = confidence_scores
    confidence_threshold = 0.9
    df['Potentially_Inaccurate'] = np.where(df['Confidence_Score'] < confidence_threshold, "Yes", "No")

    return df[['Description', 'Predicted_Profile', 'Confidence_Score', 'Potentially_Inaccurate']]


# UI for file upload
st.title("Description Category Prediction Tool")
st.write("Upload a file to get started")

# Upload file component
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Read the uploaded file
    try:
        df = pd.read_excel(uploaded_file)
        st.write("File uploaded successfully!")

        # Process the file and display the output
        processed_df = process_file(df)

        if processed_df is not None:
            st.write("Processed Data:", processed_df)

            # Create new filename for the processed file
            new_filename = f"{os.path.splitext(uploaded_file.name)[0]}_processed.xlsx"

            # Save the processed DataFrame to an Excel file
            processed_df.to_excel(new_filename, index=False, engine='xlsxwriter')

            # Provide a download Link for the processed file
            with open(new_filename, "rb") as f:
                st.download_button(
                    label="Download Processed Data",
                    data=f,
                    file_name=new_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
