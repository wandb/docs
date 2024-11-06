import os
import re
import string

# List of common prepositions to remove
prepositions = {"in", "on", "at", "by", "for", "with", "about", "against", "between",
                "into", "through", "during", "before", "after", "above", "below", "to",
                "from", "up", "down", "off", "over", "under", "again", "further", 
                "than", "then", "once", "of", "as", "a", "an", 'is', 'are', 'was',
                'were', 'be', 'been', 'being', 'the', 'you', 'I', 'what', 'happens',
                'if', 'want', 'and', 'or', 'not', 'do', 'does', 'did', 'done', 'doing',
                'using', 'use', 'which','when','my', 'your', 'our', 'i'
                }


def clean_title(title):
    # Remove HTML tags
    title = re.sub(r'<.*?>', '', title)  # Remove any HTML tags

    # Remove hyperlinks in the format [text](http://...) or plain URLs (http://...)
    title = re.sub(r'\[.*?\]\(.*?\)', '', title)  # Remove markdown-style hyperlinks
    title = re.sub(r'http\S+', '', title)  # Remove plain URLs

    # Convert title to lowercase and remove punctuation
    title = title.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Split the title into words, and filter out prepositions
    words = title.split()
    filtered_words = [word for word in words if word not in prepositions]
    
    # Limit to a maximum of 10 words for the file name
    limited_words = filtered_words[:10]
    
    # Join the remaining words with underscores for the file name
    return '_'.join(limited_words)

def add_front_matter(title, tags):
    # Create the front matter section with bullet-point tags
    tags_formatted = '\n   - '.join(tags)  # Add `-` before each tag with proper indentation
    front_matter = f"---\ntitle: \"{title}\"\ntags:\n   - {tags_formatted}\n---\n\n"
    return front_matter

def split_markdown(file_path, output_dir, tags):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the content of the markdown file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content based on H3 headers
    sections = re.split(r'(### .+)', content)

    # Loop through the split content and write each section to a new file
    for i in range(1, len(sections), 2):
        raw_title = sections[i].strip().replace('### ', '')  # Extract the raw title
        clean_file_name = clean_title(raw_title)  # Clean the title for the file name
        body = sections[i + 1].strip()

        # Prepare the file name and path
        file_name = f"{clean_file_name}.md"
        output_file = os.path.join(output_dir, file_name)

        # Add front matter to the file with the specific tags
        front_matter = add_front_matter(raw_title, tags)

        # Write each section to a new markdown file with front matter only (no H3 title)
        with open(output_file, 'w', encoding='utf-8') as out_file:
            out_file.write(front_matter + body)

        print(f"Generated {output_file}")

# Example usage with a dictionary


markdown_files_with_tags = {
'../docs/guides/technical-faq/general.md': ['None'],
'../docs/guides/technical-faq/admin.md' :['None'],
'../docs/guides/technical-faq/metrics-and-performance.md' :['None'],
'../docs/guides/technical-faq/setup.md' :  ['None'],
'../docs/guides/technical-faq/troubleshooting.md': ['None'],
'../docs/guides/track/tracking-faq.md': ['experiments'],
'../docs/guides/sweeps/faq.md': ['sweeps'],
'../docs/guides/launch/launch-faqs.md': ['launch'],
'../docs/guides/artifacts/artifacts-faqs.md': ['artifacts'],
'../docs/guides/reports/reports-faq.md': ['reports'],
'../docs/guides/track/log/logging-faqs.md': ['experiments'],    
}


output_dir = '../docs/support/'  # Directory where separate markdown files will be saved

# Process each markdown file with its associated tags
for file_path, tags in markdown_files_with_tags.items():
    split_markdown(file_path, output_dir, tags)
