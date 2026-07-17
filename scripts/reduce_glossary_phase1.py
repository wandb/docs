import csv
import re
import sys

input_file = '/Users/matt.linville/Downloads/glossary_orig.csv'
output_file = '/Users/matt.linville/Downloads/glossary_phase1.csv'

seen_terms = set()
filtered_rows = []
removed_rows = []

# Heuristics for removal
def is_formatting_artifact(term):
    # Markdown/HTML indicators and UI specific artifacts
    artifacts = ['<kbd>', '####', '**', '(Status', '`', '[', ']', '()']
    return any(a in term for a in artifacts)

# A starter set of generic words we noticed in the sample
generic_terms = {
    "minimum", "numbers", "optionally", "as a table", "active", 
    "parallel processing", "monitoring", "attributes", "notebook",
    "add two", "training set", "related assets", "create queue",
    "details", "example", "examples", "configure", "create"
}

def is_generic(term):
    return term.lower().strip() in generic_terms

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            term = row['term']
            term_lower = term.lower().strip()
            
            reason = None
            
            # 1. Deduplication
            if term_lower in seen_terms:
                reason = "Duplicate"
            # 2. Formatting artifacts
            elif is_formatting_artifact(term):
                reason = "Formatting Artifact"
            # 3. Generic terms
            elif is_generic(term):
                reason = "Generic Word"
                
            if reason:
                removed_rows.append((term, reason))
                continue
                
            seen_terms.add(term_lower)
            filtered_rows.append(row)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Original rows: {len(filtered_rows) + len(removed_rows)}")
    print(f"Remaining rows: {len(filtered_rows)}")
    print(f"Removed rows: {len(removed_rows)}")
    print("\nSample of removed rows:")
    for term, reason in removed_rows[:15]:
        print(f" - [{reason}] {term}")
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
