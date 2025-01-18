import frontmatter
import sys
from openai import OpenAI
import os

if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    print("No command line arguments provided.")
with open(file, 'r') as file:
    input = file.read()

data = frontmatter.loads(input)

print("title:", data["title"])
print("description:",data["description"])

# Set your API key
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY", 1),
)

prompt = """
Given a page comprised of the following markdown, generate a one-sentence summary, suitable for use in a <META tag='description'> tag. Just return the description text, not the HTML. Make sure the output is suitable for use in an HTML attribute -- i.e. there are no quote marks or non-ascii characters."
"""
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": data.content},
    ]
)
# Print the response
data["description"] = response.choices[0].message.content
print("NEW DESCRIPTION:",data["description"])

prompt = """
Given a page comprised of the following markdown, generate a list of one to five keywords, suitable for use in a <META tag='keyword'> tag. Ideally, just do one, but if you have to go up to five to cover everything, that's okay. Just print the keywords, not the HTML. Make sure the output is suitable for use in an HTML attribute -- i.e. there are no quote marks or non-ascii characters."
"""
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": data.content},
    ]
)
# Print the response
data["keywords"] = response.choices[0].message.content
print("NEW KEYWORDS:",data["keywords"])

prompt = """
Given a page comprised of the following markdown, generate a bulleted list in markdown that summarizes the text in no more than three one-sentence bullet points.
"""
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": data.content},
    ]
)
# Print the response
bullets = response.choices[0].message.content
print("Summary bullet-points:\n",bullets)