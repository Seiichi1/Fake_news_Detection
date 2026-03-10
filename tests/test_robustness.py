import pandas as pd
import subprocess

try:
    df = pd.read_csv('data/WELFake_Dataset.csv')
    
    # Get a Real news sample (Label 1) that starts with "WASHINGTON"
    real_text = df[df['label'] == 1]['text'].dropna()
    target_real = None
    for text in real_text:
        if "WASHINGTON" in text[:20]:
            target_real = text
            break
            
    if target_real:
        print(f"--- Original Real News ---")
        print(f"Snippet: {target_real[:100]}...")
        # Run prediction on original
        subprocess.run([".venv_py311\\Scripts\\python.exe", "debug_predict.py", "--text", target_real[:500]])
        
        print(f"\n--- Modified Real News (Removed 'WASHINGTON') ---")
        # Remove the location prefix roughly
        modified_text = target_real.split("—", 1)[-1].strip() if "—" in target_real else target_real[20:]
        print(f"Snippet: {modified_text[:100]}...")
        # Run prediction on modified
        subprocess.run([".venv_py311\\Scripts\\python.exe", "debug_predict.py", "--text", modified_text[:500]])
    else:
        print("Could not find a suitable Real news sample.")

except Exception as e:
    print(f"Error: {e}")
