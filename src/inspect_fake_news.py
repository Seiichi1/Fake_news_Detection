import pandas as pd

try:
    df = pd.read_csv('data/WELFake_Dataset.csv')
    fake_news = df[df['label'] == 0]['text'].dropna().head(5).tolist()
    
    print("--- Fake News Samples ---")
    for i, text in enumerate(fake_news):
        print(f"\nSample {i+1}:")
        print(f"{text[:300]}...")
        print("-" * 20)

except Exception as e:
    print(f"Error: {e}")
