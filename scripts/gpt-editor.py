import sys
from openai import OpenAI
import os

if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    print("No command line arguments provided.")
with open(file, 'r') as file:
    input = file.read()
    file.close()

data = input.split('---') # data[0]=blank, data[1]=frontmatter, data[2]=content

# Set your API key
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY", 1),
)

prompt = """
Given a page comprised of the following markdown, rewrite the text for clarity, brevity, and adherence to the Google Technical Documentation style guide. Do not remove things like the import statements at the top of the markdown file as that is used to tell our markdown processor that it needs to import certain libraries to render the content correctly. Be sure to leave markup intact, as well, such as <TabItem> and <Tab> tags. Avoid the use of plural pronouns like "we" and the use of future tense, such as "W&B will do x,y,z." Avoid the use of Latin abbreviations such as "i.e." and "e.g." Do not wrap the output in triple tics or label it as markdown, it will be parsed as markdown already.
"""
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

