import sys
from openai import OpenAI
import os
import subprocess
import json 

# This file uses GPT to do an edit pass on a markdown file. 
#
# Usage (run from root of repo):
# OPENAI_API_KEY={key} python scripts/gpt-editor.py path/to/markdown-file.md


if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    print("No command line arguments provided.")
with open(file, 'r') as file:
    input = file.read()
    file.close()

data = input.split('---') # data[0]=blank, data[1]=frontmatter, data[2]=content

result = subprocess.run(
    ["vale", "--output=JSON", file.name], 
    text=True, 
    capture_output=True
)

try:
    vale_output = json.loads(result.stdout)
except json.JSONDecodeError:
    print("Failed to parse Vale JSON output")
    vale_output = {}

# Set your API key
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY", 1),
)

prompt = f"""Given a page comprised of the following markdown, rewrite the text for clarity, brevity, and adherence to the Google Technical Documentation style guide. 

### Markdown File Linting Issues:
{vale_output}

Make sure to:
- Do not remove import statements at the top.
- Leave markup tags such as `<TabItem>` and `<Tab>` intact.
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
{file}
```
Please rewrite it accordingly. """

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": data[2]},
    ]
)
new_content = response.choices[0].message.content
# Print the response
print("OLD CONTENT:")
print(data[2])
# Print the response
print("NEW CONTENT:")
print(new_content)

output = "---" + data[1] + "---\n" + new_content

with open(sys.argv[1], "w") as file:
    file.write(output) 


