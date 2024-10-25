import frontmatter
import sys
import os
import json
import glob

# Specify the directory containing the markdown files
directory = 'docs/support'
tagList = dict()
outputTemplate = """---
title: {{tag}} 
---
The following support questions are tagged with "{{tag}}". If you don't see 
your question answered, try [asking the community](https://community.wandb.ai/), 
or email [support@wandb.com](mailto:support@wandb.com).

"""
def append_topic_to_tag_page(tag,markdown):
    tag = tag.lower()
    file_path = directory + "/index_" + tag + ".md"
    with open(file_path, "a") as f:
        f.write("- " + markdown + "\n")
def write_tag_page(tag):
    tag = tag.lower()
    file_path = directory + "/index_" + tag + ".md"
    with open(file_path, "w") as file:
        file.write(outputTemplate.replace('{{tag}}',tag.title())) 
def delete_files_matching_pattern(pattern, directory="."):
    """Deletes files matching the given pattern in the specified directory."""

    for file in glob.glob(os.path.join(directory, pattern)):
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")

# Example usage:
delete_files_matching_pattern("docs/support/index_*.md")  # Deletes all existing support nav files

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file has a .md extension
    if filename.endswith('.md'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Open and read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Print the file name and content
            try:
                data = frontmatter.loads(content)
                if 'tags' in data: # 'tags' front-matter exists
                    if (data['tags']): # front-matter is not empty array i.e. []
                        for tag in data['tags']:
                            tag = tag.lower().strip()
                            if tag not in tagList:
                                tagList[tag] = []
                                write_tag_page(tag)
                            tagList[tag].append('[' + data['title'] + '](' + file_path.replace('docs/support/','') + ')')
                        

            except Exception as error:
                print("ERROR:",error,file_path)

# Rewrite support section sidebar
sidebar_prefix = "  support: [{type: 'doc',id: 'support/index',label: 'Support',},"
topic_additions = ""
sidebar_suffix = "]"
for tag in tagList:
    tagList[tag].sort()
    for markdown in tagList[tag]:
        append_topic_to_tag_page(tag,markdown)
with open('sidebars.js', 'r') as infile, open('output.txt', 'w') as outfile:
    for line in infile:
        if sidebar_prefix in line:
            for tag in sorted(tagList):
                topic_additions += "'support/index_" + tag.lower() + "',"
            line = sidebar_prefix + topic_additions + sidebar_suffix + '\n'
        outfile.write(line)
os.remove('sidebars.js')
os.rename('output.txt', 'sidebars.js')
print(tagList)
