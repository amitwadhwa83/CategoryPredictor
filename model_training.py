# model_training.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the data
data = pd.read_csv("data.csv")

# Check if required columns are present
if 'Description' not in data.columns or 'Category' not in data.columns:
    raise ValueError("Data should contain 'Description' and 'Category' columns.")

# Preprocess and prepare data
X = data['Description'].fillna("")  # Fill NaNs with empty strings
y = data['Category']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text data (train and test)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the KNeighborsClassifier model (KNN with cosine distance)
knn = KNeighborsClassifier(metric='cosine', n_neighbors=3)
knn.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = knn.predict(X_test_tfidf)
print("Model evaluation metrics:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model and embedding model
joblib.dump(knn, "model.pkl")
joblib.dump(vectorizer, "embedding_model.pkl")

print("Model and embedding model saved successfully as 'model.pkl' and 'embedding_model.pkl'.")
