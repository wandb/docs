import frontmatter
import sys
import os

# Specify the directory containing the markdown files
directory = 'docs/support'

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
                        print(file_path,data['tags'])

            except Exception as error:
                print("ERROR:",error,file_path)