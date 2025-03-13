import sys
import os
import json
import subprocess
from openai import OpenAI
import weave

weave.init("gpt-editor-v0.0.5")  # Initialize Weave project

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
    return f"""You are a brilliant. technical documentation editor. Given a page comprised of the following markdown, please rewrite the text for clarity, brevity, and adherence to the Google Technical Documentation style guide.

### Markdown File Linting Issues:
{json.dumps(vale_output, indent=2)}

When handling the Vale feedback and using it to rewrite the following markdown content, here are your general instructions:
- If Vale feedback matches line 1, column 1, that is Vale's way of saying that it's a general comment on the entire markdown file, so please keep that feedback in mind throughout the text.
- Leave Hugo markup tags such as `{{< relref >}}` and `{{< note >}}` intact.
- Avoid future tense (e.g., do not use "will").
- Avoid Latin abbreviations like "i.e." and "e.g."
- Avoid wrapping the output in triple backticks or labeling it as markdown.
- Use active voice and correct instances of passive voice (for example, change "be added" to "adds").
- Use direct and inclusive language (for example, use "allowed" instead of "whitelisted").
- Do not use first-person pronouns (for example, do not use "I," or "we").
- Use the Oxford comma when appropriate.
- Commas and periods must go inside quotation marks.
- Headings must use sentence-style capitalization.
- You can touch any of the example code inside the markdown, except for the code comments
- Avoid using indirect, soft terms like "may," "should," and "perhaps." Technical documentation is prescriptive and documents exactly what happens and when.
- We want to hit a Flesch-Kincaid readability level of 7th grade and Flesch-Kincaid ease-of-reading score above 70.
- If Vale reports violations of a Microsoft rule and a Google rule and the error messages seem to conflict, favor the Google style guide.

Here is the markdown content:

```md
{content}
```
Please rewrite it accordingly. Thank you!"""

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