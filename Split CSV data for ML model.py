import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset into a pandas dataframe
data = pd.read_csv('your_dataset.csv')

# Split your dataset into training and combined validation/test sets
train_data, val_test_data = train_test_split(data, test_size=0.2, random_state=42)

# Split the validation/test set into separate validation and test sets
val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

# Save the datasets to CSV files and download them
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

from google.colab import files
files.download('train_data.csv')
files.download('val_data.csv')
files.download('test_data.csv')
