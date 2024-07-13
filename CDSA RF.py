import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Load the Excel file
file_path = 'dataclean01lab.xlsx'
data = pd.read_excel(file_path, skiprows=0, usecols=[0, 1], nrows=2441)

# Rename columns if necessary to match the expected names
data.columns = ['Text', 'Label']

# Ensure that Label column contains only strings and handle NaNs
data['Label'] = data['Label'].astype(str).fillna('')

# Split the data into training and analysis parts
training_data = data[:1716].copy()  # First 1716 rows for training
analysis_data = data[1716:2440].copy()  # Next 724 rows for analysis

# Initialize TF-IDF Vectorizer and Random Forest Classifier
vectorizer = TfidfVectorizer(max_features=1000)
rf = RandomForestClassifier(n_estimators=100)

# Create a pipeline
pipeline = Pipeline([('vectorizer', vectorizer), ('rf', rf)])

# Encode labels
le = LabelEncoder()
training_labels = le.fit_transform(training_data['Label'])

# Fit the pipeline on training data
pipeline.fit(training_data['Text'], training_labels)

# Predict on training data
training_data.loc[:, 'AI_Predicted'] = le.inverse_transform(pipeline.predict(training_data['Text']))

# Calculate the accuracy of the AI predictions in the training data
accuracy = accuracy_score(training_data['Label'], training_data['AI_Predicted'])
print(f"Accuracy of AI predictions on training data: {accuracy:.5f}")

# Calculate precision for each class
precision = precision_score(training_data['Label'], training_data['AI_Predicted'], average=None, labels=le.classes_, zero_division=0)

# Extract precision for Positif and Negatif classes
positif_precision = precision[le.transform(['Positif'])[0]]
negatif_precision = precision[le.transform(['Negatif'])[0]]

print(f"Precision for Positif class: {positif_precision:.5f}")
print(f"Precision for Negatif class: {negatif_precision:.5f}")

# Count the Positif and Negatif sentiments in the training data
positif_count = (training_data['AI_Predicted'] == 'Positif').sum()
negatif_count = (training_data['AI_Predicted'] == 'Negatif').sum()

print(f"Number of Positif sentiments in training data: {positif_count}")
print(f"Number of Negatif sentiments in training data: {negatif_count}")

# Predict on analysis data
analysis_data.loc[:, 'AI_Predicted'] = le.inverse_transform(pipeline.predict(analysis_data['Text']))

# Predict sentiment for both training and analysis data
data.loc[:, 'AI_Predicted'] = le.inverse_transform(pipeline.predict(data['Text']))

# Save to a new Excel file
output_file_path = 'sentiment_cdsa_rf.xlsx'
data.to_excel(output_file_path, index=False)

print(f"Data has been saved to {output_file_path}")
