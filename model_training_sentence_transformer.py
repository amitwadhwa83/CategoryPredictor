# model_training.py
import pandas as pd
import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

print(sentence_transformers.__version__)
print(transformers.__version__)
# Load the data
data = pd.read_csv("data/data.csv")

# Check if required columns are present
if 'Description' not in data.columns or 'Category' not in data.columns:
    raise ValueError("Data should contain 'Description' and 'Category' columns.")

model = SentenceTransformer('all-MiniLM-L6-v2')
# Preprocess and prepare data
texts  = data['Description'].fillna("")  # Fill NaNs with empty strings
labels = data['Category'].tolist()

training_embeddings= model.encode(texts)
# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(training_embeddings, labels, test_size=0.2, random_state=42)

# Train the KNeighborsClassifier model (KNN with cosine distance)
knn = KNeighborsClassifier(metric='cosine', n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

print("Model evaluation metrics:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model and embedding model
joblib.dump(knn, "models/model_st.pkl")
joblib.dump(model, "models/embedding_model_st.pkl")

print("Model and embedding model saved successfully as 'model_st.pkl' and 'embedding_model_st.pkl'.")
