""""Generate a markdown table of SDK extras from pyproject.toml.

This script reads the `pyproject.toml` file to extract optional dependencies (extras)
and generates a markdown table listing each extra along with the packages it includes.

Note that this script excludes certain extras defined in the EXCLUDE list.
"""
import os
import re
import tomli

# W&B Python SDK Extras to exclude from the table.
EXCLUDE = ["models", "kubeflow", "launch", "importers", "perf"]

DESCRIPTION = {
    "aws" : "Use `s3://` artifact references.",
    "gcp" : "Use `gs://` artifact references.",
    "azure" : "Use Azure Blob Storage artifact references.",
    "media" : "Log images, video, audio, or plots from raw data (numpy arrays, tensors).",
    "sweeps" : "Run local sweep controller (`wandb.controller()`).",
    "workspaces" : "Programmatically manage workspaces.",
}


def header():
    """Return the markdown table headers."""
    return "| Extra | Packages included | Install if you |\n"

def header_row(head_topics: int = 3) -> str:
    """Return the markdown table header row.
    
    Args:
        head_topics (int): The number of header topics.

    Returns:
        str: A markdown table header row.
    """
    return "|---------"*head_topics + "|\n"

def make_table_row(extra: str, packages: str) -> str:
    """Generate a markdown table row for the given extra and packages.
    
    Args:
        extra (str): The name of the extra.
        packages (str): The markdown links for the packages.

    Returns:
        str: A markdown table row.
    """
    return f"| `{extra}` | {packages} | {DESCRIPTION.get(extra, '')} |\n"

def clean_deps(deps: list[str]) -> list[str]:
    """Remove version specifiers from a list of dependency strings.
    
    Args:
        deps (list[str]): A list of dependency strings.

    Returns:
        list[str]: A list of dependency strings without version specifiers.
    """
    pattern = r"(>=|<=|==|~=|>|<|!=).*"
    return [re.sub(pattern, "", url) for url in deps]

def make_dep_link(deps: list[str]) -> str:
    """Create a markdown links for dependencies.
    
    Args:
        deps (list[str]): A list of dependency names.

    Returns:
        str: A string of markdown links for the dependencies.
    """
    print(f"Creating dependency links for: {deps}")
    dep_names = [dep.split(" ")[0] for dep in deps]
    return ", ".join(
        f"[{name}](https://pypi.org/project/{name}/)" for name in dep_names
    )


def generate_table(pyproject_path: str) -> str:
    """Generate the markdown table for the SDK extras.

    Args:
        pyproject_path (str): The path to the pyproject.toml file.

    Returns:
        str: A string representing the markdown table.
    """
    rows = []
    rows.append(header())
    rows.append(header_row())

    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)
        extras = pyproject.get("project", {}).get("optional-dependencies", {})
        for extra, deps in extras.items():
            if extra in EXCLUDE:
                continue
            dep_links = make_dep_link(clean_deps(deps))
            table_row = make_table_row(extra, dep_links)
            rows.append(table_row)

    return "".join(rows)


def replace_content_between_markers(
    file_path: str, start_marker: str, end_marker: str, replacement: str
) -> bool:
    """Replace content between start and end markers in a file.

    The markers themselves are preserved in the file. Only the content
    between them is replaced with the new content.

    Args:
        file_path (str): The path to the file to modify.
        start_marker (str): The starting marker string.
        end_marker (str): The ending marker string.
        replacement (str): The text to insert between the markers.

    Returns:
        bool: True if both markers were found and content was replaced, False otherwise.
    """
    with open(file_path, "r") as f:
        content = f.read()

    # Check if both markers exist
    if start_marker not in content:
        print(f"Error: Start marker '{start_marker}' not found in '{file_path}'.")
        return False

    if end_marker not in content:
        print(f"Error: End marker '{end_marker}' not found in '{file_path}'.")
        return False

    # Find the positions of the markers
    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker)

    # Check if markers are in the correct order
    if start_pos >= end_pos:
        print(f"Error: Start marker must come before end marker in '{file_path}'.")
        return False

    # Split content into three parts: before, between (to replace), and after
    before_start = content[:start_pos + len(start_marker)]
    after_end = content[end_pos:]

    # Construct the new content with replacement between markers
    updated_content = before_start + "\n" + replacement + after_end

    # Write the updated content back to the file
    with open(file_path, "w") as f:
        f.write(updated_content)

    return True


def main():
    # Path to pyproject.toml in the wandb SDK repository.
    pyproject_path = "../../wandb/pyproject.toml"

    # MDX file path that contains the markers.
    mdx_file_path = "../models/ref/python.mdx"

    # Start and end markers that enclose the table in the MDX file.
    start_marker = "{/* python-extras-start */}"
    end_marker = "{/* python-extras-end */}"

    # Generate the markdown table.
    table = generate_table(pyproject_path)

    # Replace the content between markers in the MDX file with the generated table.
    if replace_content_between_markers(mdx_file_path, start_marker, end_marker, table):
        print(f"Successfully replaced content between markers in '{mdx_file_path}'.")
    else:
        print(f"Failed to replace content between markers in '{mdx_file_path}'.")

if __name__ == "__main__":
    # To do: Add argument parsing for custom paths
    main()