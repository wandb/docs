"""Checks markdown files to ensure they have required metadata for Docusaurus. If header is missing metadata, 
    this script will print out the path along with information on how to correct the error.

    Note: Only use this script for developer guide content. 
    For e.g. docodile/docs/guide and/or docodile/i18n/ja/docusaurus-plugin-content-docs/current/guides/

Usage: sidebar_header_check.py -d <path-to-markdown-directory
"""

import os
import re
import argparse

parser = argparse.ArgumentParser(
    prog='validate_markdown_header',
    description='Checks markdown files have description and sidebar values.'
)
parser.add_argument('-d','--directory', type=str, help='The directory that contains the markdown files.')


def check_markdown_files(directory:str, available_sidebars: list):
    """Reads in markdown files in directory, and its sub-directories. 
    
    Args:
        directory (str): Markdown directory file path.
    
    """
    for root, subdirs, files in os.walk(directory):
    
        for filename in os.listdir(root):
            if filename.endswith('.md'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as file:
                    content = file.read()
                    metadata = extract_metadata(content)

                if metadata is None:
                    raise ValueError(f"Missing header metadata in file: {file_path}")
                else:
                    #check_description(metadata, file_path)
                    check_displayed_sidebar(metadata, file_path, available_sidebars)

    print("\nMarkdown header check complete.")


def check_displayed_sidebar(metadata, file_path, available_sidebars):
    """Checks that the metadata 1) has a displayed option specified and 2) the specified sidebar is valid.

    Args:
        metadata (None or dict): Contains the metadata of markdown files.
        filepath (str): Filepath to a specific markdown file.
        available_sidebars (list): Contains available language sidebar options.

    Returns:
        Raises a ValueError exception if no displayed_sidebar is missing or value provided does not match an existing sidebar option. 

    """
    displayed_sidebar = metadata.get('displayed_sidebar')
    
    if displayed_sidebar is None:
        raise ValueError(f"Missing 'displayed_sidebar' key in file header: {file_path}")
    elif displayed_sidebar not in available_sidebars:
        raise ValueError("Invalid value for 'displayed_sidebar' in file header: {}. Available sidebars are: {}".format(file_path, available_sidebars))
    return


def check_description(metadata, file_path):
    """Checks that the metadata has 

    Args:
        metadata (None or dict): Contains the metadata of markdown files.
        filepath (str): Filepath to a specific markdown file.
        available_sidebars (list): Contains available language sidebar options.

    """    
    description = metadata.get('description')
    if description is None or description == "":
        print(f"Missing description key in file: {file_path}")
    return



def extract_metadata(content:str):
    """Checks for start and end of markdown header. If found, it returns the metadata stored.

    Args:
        content (str): Markdown header information.
    
    Returns:
        metadata (dict): Returns a dictionary with metadata key-value pairs.
    """
    pattern = r'^---\n(.*?)^---\n'
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        metadata_str = match.group(1).strip()
        metadata = {}
        lines = metadata_str.split('\n')
        for line in lines:
            key_value = line.split(':', 1)
            if len(key_value) == 2:
                key, value = key_value
                metadata[key.strip()] = value.strip()
        return metadata
    return None


def main(args):

    directory_path = args.directory

    # Update this list with new sidebar options.
    available_sidebars = ['default', 'ja']

    # Starting point of markdown check
    check_markdown_files(directory_path, available_sidebars)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)