import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

# Load the training Excel file
training_file_path = 'combined_sentiments.xlsx'
training_data = pd.read_excel(training_file_path, skiprows=0, usecols=[0, 1], nrows=938)
print("Training Data Columns:", training_data.columns)

# Rename columns for clarity
training_data.columns = ['Text', 'Label']

# Ensure that Label column contains only strings and handle NaNs
training_data['Label'] = training_data['Label'].astype(str).fillna('')

# Ensure that Text column contains only strings
training_data['Text'] = training_data['Text'].astype(str)

# Load the test Excel file
test_file_path = 'combined_sentiments_test.xlsx'
test_data = pd.read_excel(test_file_path, skiprows=0, usecols=[0, 1], nrows=402)

# Rename columns for clarity
test_data.columns = ['Text', 'Label']

# Ensure that Text and Label columns contain only strings
test_data['Text'] = test_data['Text'].astype(str)
test_data['Label'] = test_data['Label'].astype(str)

# Initialize TF-IDF Vectorizer and SVM Classifier
vectorizer = TfidfVectorizer()
svm = LinearSVC()

# Create a pipeline
pipeline = Pipeline([('vectorizer', vectorizer), ('svm', svm)])

# Encode labels
le = LabelEncoder()
training_labels = le.fit_transform(training_data['Label'])

# Fit the pipeline on training data
pipeline.fit(training_data['Text'], training_labels)

# Predict on test data
test_labels = le.transform(test_data['Label'])
test_predictions = pipeline.predict(test_data['Text'])
test_data['AI_Predicted'] = le.inverse_transform(test_predictions)



# Calculate metrics for the AI predictions in the test data
accuracy = accuracy_score(test_labels, test_predictions)

print(f"Accuracy of AI predictions on training data: {accuracy:.5f}")
precision = precision_score(test_labels, test_predictions, average=None, zero_division=0)
recall = recall_score(test_labels, test_predictions, average=None, zero_division=0)
f1 = f1_score(test_labels, test_predictions, average=None, zero_division=0)

# Generate classification report
report = classification_report(test_labels, test_predictions, zero_division=0, output_dict=True)

# Create a DataFrame from the classification report
report_df = pd.DataFrame(report).transpose()
print(report_df)

# Generate confusion matrix
conf_matrix = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Count the correct and incorrect classifications in the test data
correct_predictions = test_data[test_data['Label'] == test_data['AI_Predicted']]
incorrect_predictions = test_data[test_data['Label'] != test_data['AI_Predicted']]

correct_counts = correct_predictions['Label'].value_counts()
incorrect_counts = incorrect_predictions['Label'].value_counts()

# Save only the test data to a new Excel file
output_file_path = 'sentiment_svm.xlsx'
test_data.to_excel(output_file_path, index=False)

print(f"Test data has been saved to {output_file_path}")

# Visualization
# Convert 'Positif' and 'Negatif' to numerical values for clearer scatter plot
test_data['Label_Num'] = test_data['Label'].map({'Positif': 1, 'Negatif': 0})
test_data['AI_Predicted_Num'] = test_data['AI_Predicted'].map({'Positif': 1, 'Negatif': 0})

# Scatter plot of actual vs predicted sentiments
plt.figure(figsize=(12, 6))
plt.scatter(range(len(test_data)), test_data['Label_Num'], alpha=0.5, label='Actual Sentiment', color='blue')
plt.scatter(range(len(test_data)), test_data['AI_Predicted_Num'], alpha=0.5, label='Predicted Sentiment', color='orange', marker='x')
plt.xlabel('Data Points')
plt.ylabel('Sentiment')
plt.yticks([0, 1], ['Negatif', 'Positif'])
plt.title('Actual vs Predicted Sentiments in Test Data')
plt.legend()
plt.show()

# Bar chart for correct and incorrect sentiment counts
labels = le.classes_
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, correct_counts.reindex(labels, fill_value=0), width, label='Correct')
bars2 = ax.bar(x + width/2, incorrect_counts.reindex(labels, fill_value=0), width, label='Incorrect')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Class Labels')
ax.set_ylabel('Count')
ax.set_title('Correct vs Incorrect Sentiment Counts in Test Data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add counts on top of the bars
def add_counts(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_counts(bars1)
add_counts(bars2)

plt.show()
