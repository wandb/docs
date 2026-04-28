import csv
import sys
from collections import defaultdict

input_file = '/Users/matt.linville/Downloads/glossary_phase2.csv'
output_file = '/Users/matt.linville/Downloads/glossary_phase2_5.csv'
output_md = '/Users/matt.linville/Downloads/glossary_audit_report.md'

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    filtered_rows = []
    removed_rows = []

    for row in rows:
        term = row['term']
        words = term.split()
        
        # Heuristic: more than 4 words, or contains a period that looks like sentence punctuation
        is_long = len(words) > 4
        has_sentence_punct = any(w.endswith('.') for w in words if len(w) > 2)

        if is_long or (len(words) > 2 and has_sentence_punct):
            removed_rows.append((term, "Long phrase or sentence"))
        else:
            filtered_rows.append(row)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    # Regenerate MD report
    groups = defaultdict(list)
    for row in filtered_rows:
        tag = row['definition']
        if tag not in ["UI Label", "W&B Product Feature", "Technical Term"]:
            tag = "Other / Specific Context"
        groups[tag].append(row)

    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# W&B Glossary Audit Report\n\n")
        f.write("Please review the following terms and their translations. Ensure that W&B specific terms, UI labels, and technical jargon are translated correctly and consistently.\n\n")
        
        for tag in sorted(groups.keys()):
            f.write(f"## {tag} ({len(groups[tag])} terms)\n\n")
            f.write("| Term | French | Japanese | Korean |\n")
            f.write("|------|--------|----------|--------|\n")
            for row in sorted(groups[tag], key=lambda x: x['term'].lower()):
                term = row['term'].replace('|', '\\|')
                fr = row.get('translation_fr', '').replace('|', '\\|')
                ja = row.get('translation_ja', '').replace('|', '\\|')
                ko = row.get('translation_ko', '').replace('|', '\\|')
                f.write(f"| `{term}` | {fr} | {ja} | {ko} |\n")
            f.write("\n")

    print(f"Original rows (Phase 2): {len(rows)}")
    print(f"Remaining rows: {len(filtered_rows)}")
    print(f"Removed rows: {len(removed_rows)}")
    print(f"Successfully regenerated markdown report at {output_md}")
    
    print("\nSample of removed phrases:")
    for term, reason in removed_rows[:15]:
        print(f" - {term}")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
