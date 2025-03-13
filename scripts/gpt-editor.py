import sys
import os
import json
import subprocess
from openai import OpenAI
import weave

weave.init("gpt-markdown-editor-v0.0.5")  # Initialize Weave project

@weave.op  # Log file reading
def read_markdown_file(file_path):
    """Reads the markdown file and returns its content split into sections."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content.split('---')  # [blank, frontmatter, content]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

@weave.op  # Track Vale outputs
def run_vale_linter(file_path):
    """Runs Vale linter on the file and returns the parsed JSON output."""
    result = subprocess.run(["vale", "--output=JSON", file_path], text=True, capture_output=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Warning: Failed to parse Vale JSON output.")
        return {}

def generate_prompt(content, vale_output):
    """Generates the prompt for the OpenAI model."""
    return f"""Given a page comprised of the following markdown, rewrite the text for clarity, brevity, and adherence to the Google Technical Documentation style guide.

### Markdown File Linting Issues:
{json.dumps(vale_output, indent=2)}

Make sure to:
- Leave Hugo markup tags such as `{{< relref >}}` and `{{< note >}}` intact.
- Avoid future tense (e.g., do not use "will").
- Avoid Latin abbreviations like "i.e." and "e.g."
- Avoid wrapping the output in triple backticks or labeling it as markdown.
- Do not use passive voice (e.g., "be added" → "adds").
- Use direct and inclusive language (e.g., "allowed" instead of "whitelisted").
- Do not use first-person pronouns (e.g., "I," "we").
- Use the Oxford comma when appropriate.
- Commas and periods must go inside quotation marks.
- Headings must use sentence-style capitalization.

Here is the markdown content:

```md
{content}
```
Please rewrite it accordingly."""

@weave.op  # Log OpenAI API calls
def get_gpt_rewrite(client, model, prompt, content):
    """Gets a rewrite of the content using OpenAI's GPT model."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
    )
    return response.choices[0].message.content

@weave.op  # Log before/after comparisons
def main():
    """Main execution function."""
    if len(sys.argv) <= 1:
        print("Usage: python scripts/gpt-editor.py path/to/markdown-file.md")
        sys.exit(1)
    
    file_path = sys.argv[1]
    data = read_markdown_file(file_path)
    if len(data) < 3:
        print("Error: Invalid markdown file structure. Expected frontmatter and content.")
        sys.exit(1)
    
    frontmatter, content = data[1], data[2]
    vale_output = run_vale_linter(file_path)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = generate_prompt(content, vale_output)
    
    new_content = get_gpt_rewrite(client, "gpt-4o-mini", prompt, content)
    
    print("OLD CONTENT:\n", content)
    print("NEW CONTENT:\n", new_content)
    
    output = f"---{frontmatter}---\n{new_content}"
    
    with open(file_path, "w") as file:
        file.write(output)
    print(f"File '{file_path}' successfully updated.")

if __name__ == "__main__":
    main()