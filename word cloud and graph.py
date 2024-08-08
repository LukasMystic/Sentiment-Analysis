import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

# Define a list of Indonesian stopwords
stopwords_id = set([
    'dan', 'di', 'ke', 'dari', 'yang', 'untuk', 'adalah', 'pada', 'ini', 'itu', 'sebuah', 'dengan', 'atau', 'juga', 'sudah', 'oleh', 'ada', 'saya', 'kita', 'kami', 'itu', 'mereka', 'apakah', 'akan', 'tersebut', 'selain', 'boleh', 'harus', 'sebagai', 'saat', 'tetapi', 'jadi', 'dapat', 'lebih', 'lagi', 'melalui', 'seperti', 'jika', 'seperti', 'namun', 'nya', 'pak', 'bisa', 'tidak', 'bapak'
])

# Load the Excel file
file_path = 'combined_sentiments_test.xlsx'  # Replace with the path to your Excel file
data = pd.read_excel(file_path)

# Preprocess the text in column A
text = ' '.join(data['text'].astype(str))
text = text.lower()
text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words (1-2 characters)
text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

# Remove stopwords
words = text.split()
filtered_words = [word for word in words if word not in stopwords_id]

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Column A')
plt.show()

# Generate the word frequency counter
word_counts = Counter(filtered_words)

# Get the most common words
most_common_words = word_counts.most_common(10)
words, counts = zip(*most_common_words)

# Display the word graph (bar chart)
plt.figure(figsize=(10, 5))
plt.bar(words, counts, color='skyblue')
plt.xlabel('Words')
plt.ylabel('Counts')
plt.title('Top 10 Most Common Words in Column A')
plt.xticks(rotation=45)
plt.show()
