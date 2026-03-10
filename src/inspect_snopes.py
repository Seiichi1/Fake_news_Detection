import csv

filename = 'data/snopeswithsum.csv'

try:
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rates = set()
        
        # Collect unique rates and show first few examples
        print("Scanning file...")
        row_count = 0
        for row in reader:
            if 'rate' in row:
                rates.add(row['rate'])
            row_count += 1
            if row_count < 5:
                print(f"Row {row_count}: Rate='{row.get('rate')}', Claim='{row.get('claim')[:50]}...'")

        print("\nUnique values in 'rate' column:")
        for r in sorted(list(rates)):
            print(f"- {r}")
            
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"Error: {e}")
