import os, csv

missing = []
found = []

with open('data/gold_standard/gold_standard_test.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        doc_id = row.get('document')
        term = row.get('term')
        if not doc_id or not term:
            print('SKIPPING empty row:', row)
            continue
        term_lower = term.lower()
        raw_file = os.path.join('data/raw/test', doc_id + '.txt')
        if not os.path.exists(raw_file):
            missing.append('FILE NOT FOUND: ' + raw_file)
            continue
        with open(raw_file, encoding='utf-8', errors='ignore') as rf:
            content = rf.read().lower()
        if term_lower in content:
            found.append((doc_id, term))
        else:
            missing.append((doc_id, term))

print('Found:', len(found))
print('Missing:', len(missing))
for m in missing:
    print(' ', m)
