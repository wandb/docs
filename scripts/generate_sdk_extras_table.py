import os
import re
import tomli

pyproject_path = "./pyproject.toml"
output_path = "../snippets/en/_includes/python-sdk-extras.mdx"
version_specifiers = [">=", "<=", "==", "~=", ">", "<", "!="]

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

def make_table_row(extra, packages):
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

    print("These are deps" , deps)
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
    main()