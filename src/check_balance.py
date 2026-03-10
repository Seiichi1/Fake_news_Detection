import pandas as pd

try:
    df = pd.read_csv('data/WELFake_Dataset.csv')
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    print("\nClass Distribution (%):")
    print(df['label'].value_counts(normalize=True))
except Exception as e:
    print(f"Error: {e}")
