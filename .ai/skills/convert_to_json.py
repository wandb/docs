#!/usr/bin/env python3
"""
Script to convert llm_evaluation_tasks.csv to tasks.json
Mapping: id -> Reference Script, task -> Task Description, category -> Category
"""

import csv
import json


def convert_csv_to_json(csv_file: str, json_file: str) -> None:
    """Convert CSV file to JSON with specified column mapping."""
    tasks = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Skip empty rows
            if not row.get('Reference Script') or not row.get('Reference Script').strip():
                continue

            task_obj = {
                "id": row['Reference Script'].strip(),
                "task": row['Task Description'].strip(),
                "category": row['Category'].strip()
            }
            tasks.append(task_obj)

    # Create output structure
    output = {"tasks": tasks}

    # Write to JSON file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Successfully converted {len(tasks)} tasks to {json_file}")


if __name__ == "__main__":
    convert_csv_to_json("llm_evaluation_tasks.csv", "tasks.json")
