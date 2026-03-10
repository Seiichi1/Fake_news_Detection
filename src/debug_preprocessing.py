import re
import string

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Test Case
title = "Aliens Land"
text = "They want pizza."
combined = title + " [SEP] " + text

print(f"Original: {combined}")
cleaned = clean_text(combined)
print(f"Cleaned:  {cleaned}")

if "sep" not in cleaned:
    print("\n[!] WARNING: [SEP] token is removed or corrupted!")
else:
    print("\n[OK] [SEP] token preserved (check format).")
