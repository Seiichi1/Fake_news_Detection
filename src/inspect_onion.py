import csv

filename = 'data/OnionOrNot.csv'

try:
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        print(f"Columns: {headers}")
        
        print("\nFirst 5 rows:")
        rows = []
        for i in range(5):
            rows.append(next(reader))
        for row in rows:
            print(row)
            
        # Count labels roughly
        print("\nScanning for label distribution (first 1000)...")
        label_counts = {}
        f.seek(0)
        next(reader) # skip header
        count = 0
        for row in reader:
            if count > 1000: break
            label = row[1] if len(row) > 1 else 'unknown'
            label_counts[label] = label_counts.get(label, 0) + 1
            count += 1
        print("Label Counts (sample):", label_counts)
            
except Exception as e:
    print(f"Error: {e}")
