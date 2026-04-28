import csv
import sys

input_file = '/Users/matt.linville/Downloads/glossary_phase1.csv'
output_file = '/Users/matt.linville/Downloads/glossary_phase2.csv'

# Known W&B terms
wandb_terms = {
    "artifact", "artifacts", "sweep", "sweeps", "weave", "project", "projects", 
    "run", "runs", "report", "reports", "launch", "model", "models", 
    "registry", "workspace", "entity", "entities"
}

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    filtered_rows = []
    removed_rows = []
    modified_defs = 0

    for row in rows:
        term = row['term']
        term_lower = term.lower().strip()
        definition = row['definition']
        
        # If definition is long or verbose
        if len(definition) > 50:
            def_lower = definition.lower()
            
            # Determine new definition or remove
            if "ui" in def_lower or "button" in def_lower or "tab" in def_lower or "menu" in def_lower or "label" in def_lower:
                row['definition'] = "UI Label"
                modified_defs += 1
                filtered_rows.append(row)
            elif term_lower in wandb_terms or "w&b" in def_lower or "weights & biases" in def_lower or "feature" in def_lower or "product" in def_lower:
                row['definition'] = "W&B Product Feature"
                modified_defs += 1
                filtered_rows.append(row)
            else:
                # If it's a long definition without UI or W&B context, it's likely generic steer prompt for standard jargon.
                # Let's remove it if it's not capitalized (heuristic for non-proper noun) and not a known term.
                if term.islower():
                    removed_rows.append((term, "Standard Jargon / Redundant Steer Prompt"))
                    continue
                else:
                    row['definition'] = "Technical Term"
                    modified_defs += 1
                    filtered_rows.append(row)
        else:
            filtered_rows.append(row)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Original rows (Phase 1): {len(rows)}")
    print(f"Remaining rows: {len(filtered_rows)}")
    print(f"Removed rows: {len(removed_rows)}")
    print(f"Modified definitions: {modified_defs}")
    print("\nSample of removed rows:")
    for term, reason in removed_rows[:15]:
        print(f" - [{reason}] {term}")
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
