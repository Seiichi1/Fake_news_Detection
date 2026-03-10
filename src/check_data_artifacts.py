import pandas as pd

try:
    df = pd.read_csv('data/WELFake_Dataset.csv')
    
    # Check Real News (Label 1)
    print("--- Real News Samples (Label 1) ---")
    real_news = df[df['label'] == 1]['text'].dropna().head(10).tolist()
    for i, text in enumerate(real_news):
        print(f"{i+1}. {text[:100]}...")

    # Check Fake News (Label 0)
    print("\n--- Fake News Samples (Label 0) ---")
    fake_news = df[df['label'] == 0]['text'].dropna().head(10).tolist()
    for i, text in enumerate(fake_news):
        print(f"{i+1}. {text[:100]}...")

except Exception as e:
    print(f"Error: {e}")
