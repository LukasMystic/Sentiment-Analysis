import pandas as pd

# Load the data
file_path = 'CDSADataset2_.xlsx'
df = pd.read_excel(file_path)

# Filter the first positive and negative sentiments
positive_df = df[df['sentimen'] == 'Positif'].head(201)
negative_df = df[df['sentimen'] == 'Negatif'].head(201)

# Combine them into a single DataFrame
combined_df = pd.concat([positive_df, negative_df])

# Save to a new Excel file
output_file_path = 'combined_sentiments_test.xlsx'
combined_df.to_excel(output_file_path, index=False)

print(f"Combined file saved to {output_file_path}")
