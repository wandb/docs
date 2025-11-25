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

def header():
    """Return the markdown table headers."""
    return "| Extra | Packages included |\n"

def header_row(head_topics: int = 2):
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
    return f"| `{extra}` | {packages} |\n"

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


def generate_table(pyproject_path: str) -> list[str]:
    """Generate the markdown table for the SDK extras.
    
    Args:
        pyproject_path (str): The path to the pyproject.toml file.

    Returns:
        list[str]: A list of strings representing the markdown table.
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

    return rows


def main():
    # Pyproject.toml is expected to be in the same directory as this script.
    pyproject_path = "./pyproject.toml"

    # Output path for the generated markdown table.
    output_path = "../snippets/en/_includes/python-sdk-extras.mdx"

    # Create the output file if it doesn't exist
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            f.write("Creating new file...\n")
        print(f"File '{output_path}' created.")
    else:
        print(f"File '{output_path}' already exists.")

    # Generate the markdown table and write to the output file
    table_lines = generate_table(pyproject_path)
    with open(output_path, "w") as f:
        f.writelines(table_lines) 
    print(f"Markdown table written to '{output_path}'.")

if __name__ == "__main__":
    # To do: Add argument parsing for custom paths
    main()