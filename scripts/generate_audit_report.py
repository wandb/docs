import csv
from collections import defaultdict

input_file = '/Users/matt.linville/Downloads/glossary_phase2.csv'
output_md = '/Users/matt.linville/Downloads/glossary_audit_report.md'

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by definition (context tag)
    groups = defaultdict(list)
    for row in rows:
        # Group all the short original definitions under "Other / Specific Context"
        # if they aren't one of our standardized tags
        tag = row['definition']
        if tag not in ["UI Label", "W&B Product Feature", "Technical Term"]:
            tag = "Other / Specific Context"
        
        groups[tag].append(row)

    # Write Markdown report
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# W&B Glossary Audit Report\n\n")
        f.write("Please review the following terms and their translations. Ensure that W&B specific terms, UI labels, and technical jargon are translated correctly and consistently.\n\n")
        
        for tag in sorted(groups.keys()):
            f.write(f"## {tag} ({len(groups[tag])} terms)\n\n")
            f.write("| Term | French | Japanese | Korean |\n")
            f.write("|------|--------|----------|--------|\n")
            
            # Sort terms alphabetically within each group
            for row in sorted(groups[tag], key=lambda x: x['term'].lower()):
                term = row['term'].replace('|', '\\|')
                fr = row.get('translation_fr', '').replace('|', '\\|')
                ja = row.get('translation_ja', '').replace('|', '\\|')
                ko = row.get('translation_ko', '').replace('|', '\\|')
                f.write(f"| `{term}` | {fr} | {ja} | {ko} |\n")
            
            f.write("\n")

    print(f"Successfully generated markdown report at {output_md}")
    print("Here is a preview of the report structure and first few entries:\n")
    
    # Print a preview
    for tag in sorted(groups.keys()):
        print(f"## {tag} ({len(groups[tag])} terms)")
        print("| Term | French | Japanese | Korean |")
        print("|------|--------|----------|--------|")
        preview_rows = sorted(groups[tag], key=lambda x: x['term'].lower())[:3]
        for row in preview_rows:
            print(f"| `{row['term']}` | {row.get('translation_fr', '')} | {row.get('translation_ja', '')} | {row.get('translation_ko', '')} |")
        print("...\n")

except Exception as e:
    print(f"Error: {e}")
