import os
import csv

def check_gold_standard(gold_file, raw_dir):
    missing = []
    found = []
    
    with open(gold_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            doc_id = row['document']
            term = row['term'].lower()
            
            raw_file = os.path.join(raw_dir, f"{doc_id}.txt")
            
            if not os.path.exists(raw_file):
                missing.append(f"FILE NOT FOUND: {raw_file}")
                continue
            
            with open(raw_file, 'r', encoding='utf-8', errors='ignore') as rf:
                content = rf.read().lower()
            
            if term in content:
                found.append((doc_id, term))
            else:
                missing.append((doc_id, term))
    
    return found, missing

# Check dev (doc_01 to doc_20)
print("=== Checking gold_standard_dev.tsv ===")
found, missing = check_gold_standard(
    "data/gold_standard/gold_standard_dev.tsv",
    "data/raw/dev"
)
print(f"Found: {len(found)}")
print(f"Missing: {len(missing)}")
if missing:
    print("\nTerms NOT found in their documents:")
    for item in missing:
        print(f"  {item}")

# Check test (doc_21 to doc_26)
print("\n=== Checking gold_standard_test.tsv ===")
found, missing = check_gold_standard(
    "data/gold_standard/gold_standard_test.tsv",
    "data/raw/test"
)
print(f"Found: {len(found)}")
print(f"Missing: {len(missing)}")
if missing:
    print("\nTerms NOT found in their documents:")
    for item in missing:
        print(f"  {item}")
