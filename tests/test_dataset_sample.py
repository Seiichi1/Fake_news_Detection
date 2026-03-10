import pandas as pd
import subprocess

try:
    # Load dataset
    df = pd.read_csv('data/WELFake_Dataset.csv')
    
    # Get first Fake news (Label 0)
    fake_text = df[df['label'] == 0]['text'].dropna().iloc[0]
    
    print(f"--- Testing with Dataset Sample (Label 0) ---")
    print(f"Text Length: {len(fake_text)}")
    print(f"Snippet: {fake_text[:100]}...")
    
    # Run prediction
    # We use subprocess to call the existing debug_predict.py to avoid reloading model
    # Escape quotes for command line
    safe_text = fake_text.replace('"', '\\"').replace("'", "\\'")
    
    # Just use the first 500 chars to avoid command line length issues
    short_text = fake_text[:500]
    
    print("\nRunning inference...")
    subprocess.run([".venv_py311\\Scripts\\python.exe", "debug_predict.py", "--text", short_text])

except Exception as e:
    print(f"Error: {e}")
