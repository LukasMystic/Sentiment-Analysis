import pandas as pd
import matplotlib.pyplot as plt


file_path = 'combined_sentiments.xlsx'  
data = pd.read_excel(file_path)

# Count the occurrences of each label in column B
label_counts = data['sentimen'].value_counts()

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff6666'])
plt.title('Distribution of Positif and Negatif')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the pie chart
plt.show()
