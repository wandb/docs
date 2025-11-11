# Marimo Examples

This folder contains Marimo notebook code that can be embedded in MDX files.

## Usage

In your MDX file, import and use the Marimo component:

```mdx
import {Marimo} from "/snippets/Marimo.jsx";

<Marimo file="/marimo-examples/slider-example.txt" />
```

## File Format

Files should contain markdown-formatted Python code blocks:

```
\`\`\`python
import marimo as mo
\`\`\`

\`\`\`python
# Your code here
\`\`\`
```

Each code block becomes a cell in the Marimo notebook.

## How It Works

1. The Marimo component fetches the `.txt` file
2. Parses the markdown to extract Python code blocks
3. Creates proper HTML elements (`<pre><code class="language-python">`)
4. Marimo's script processes these and creates an interactive notebook

## Available Examples

- `slider-example.txt` - Interactive slider demo with emoji output
- `hello-world.txt` - Basic text input example
