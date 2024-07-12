import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score

# Load the model and tokenizer
model_name = "indolem/indobert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function to truncate text to fit within the model's maximum length
def truncate_text(text, max_length=512):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.convert_tokens_to_string(tokens)

# Function to analyze sentiment with truncation
def analyze_sentiment(text):
    truncated_text = truncate_text(text)
    result = sentiment_analysis(truncated_text)
    print(f"Text: {text}\nTruncated Text: {truncated_text}\nResult: {result}\n")  # Debug print
    return result[0]['label']

# Function to map model labels to Indonesian labels
def map_labels(label):
    label_mapping = {
        'LABEL_0': 'Negatif',
        'LABEL_1': 'Positif',   
        'LABEL_2': 'Positif'    
    }
    return label_mapping.get(label, 'Negatif')  # Default to 'Negatif' if not found

# Load the Excel file
file_path = 'dataclean01lab.xlsx'
data = pd.read_excel(file_path, skiprows=0, usecols=[0, 1], nrows=2441)

# Print the column names to check for any discrepancies
print(f"Column names: {data.columns.tolist()}")

# Rename columns if necessary to match the expected names
data.columns = ['Text', 'Label']

# Ensure that Label column contains only strings and handle NaNs
data['Label'] = data['Label'].astype(str).fillna('')

# Standardize the Label column to lower case and strip whitespace
data['Label'] = data['Label'].str.lower().str.strip()

# Print unique labels to check for consistency
print(f"Unique labels in the dataset: {data['Label'].unique()}")

# Split the data into training and analysis parts
training_data = data[:1716]  # First 1716 rows for training
analysis_data = data[1716:2440]  # Next 724 rows for analysis

# Perform sentiment analysis on the training data
training_data['AI_Predicted'] = training_data['Text'].apply(analyze_sentiment)

# Map model labels to Indonesian labels
training_data['AI_Predicted'] = training_data['AI_Predicted'].apply(map_labels)

# Ensure AI_Predicted column contains only strings
training_data['AI_Predicted'] = training_data['AI_Predicted'].astype(str)

# Standardize the AI_Predicted column to lower case and strip whitespace
training_data['AI_Predicted'] = training_data['AI_Predicted'].str.lower().str.strip()

# Print unique AI predictions to check for consistency
print(f"Unique AI predictions: {training_data['AI_Predicted'].unique()}")

# Calculate the distribution of sentiments in the training data
positive_count = sum(training_data['AI_Predicted'] == 'positif')
negative_count = sum(training_data['AI_Predicted'] == 'negatif')

print(f"Training data positive count: {positive_count}")
print(f"Training data negative count: {negative_count}")

# Calculate the accuracy of the AI predictions in the training data
accuracy = accuracy_score(training_data['Label'], training_data['AI_Predicted'])
print(f"Accuracy of AI predictions on training data: {accuracy:.5f}")

# Calculate precision for each class
precision = precision_score(training_data['Label'], training_data['AI_Predicted'], average=None, labels=['positif', 'negatif'])

# Extract precision for Positif and Negatif classes
positif_precision = precision[0]
negatif_precision = precision[1]

print(f"Precision for Positif class: {positif_precision:.5f}")
print(f"Precision for Negatif class: {negatif_precision:.5f}")

# Determine the dominant sentiment
dominant_sentiment = 'positif' if positive_count > negative_count else 'negatif'

# Function to analyze sentiment with dominant sentiment for neutral handling
def analyze_sentiment_with_dominant(text):
    truncated_text = truncate_text(text)
    result = sentiment_analysis(truncated_text)
    label = result[0]['label']
    if label == 'LABEL_1':  # Assuming 'LABEL_1' is neutral
        return dominant_sentiment
    return map_labels(label)

# Perform sentiment analysis on the analysis data
analysis_data['AI_Predicted'] = analysis_data['Text'].apply(analyze_sentiment_with_dominant)

# Predict sentiment for both training and analysis data
data['AI_Predicted'] = data['Text'].apply(analyze_sentiment_with_dominant).str.lower().str.strip()

# Save to a new Excel file
output_file_path = 'sentiment_indobert.xlsx'
data.to_excel(output_file_path, index=False)

print(f"Data has been saved to {output_file_path}")
