# Marimo Examples

This folder contains raw Marimo notebook code that can be embedded in MDX files without Mintlify processing them.

## Usage

In your MDX file, use the Marimo component with the `file` prop:

```mdx
<Marimo file="/marimo-examples/slider-example.txt" />
```

## Why External Files?

Storing Marimo code in external `.txt` files avoids:
- Mintlify's markdown processing adding "Copy" buttons
- Complex escaping of backticks in MDX
- Rendering issues during build

## Creating New Examples

1. Create a new `.txt` file in this folder
2. Write your Marimo code using standard markdown code blocks:

```
\`\`\`python
import marimo as mo
\`\`\`

\`\`\`python
# Your code here
\`\`\`
```

3. Reference it in your MDX:

```mdx
<Marimo file="/marimo-examples/your-example.txt" />
```

## Available Examples

- `slider-example.txt` - Interactive slider demo
- `hello-world.txt` - Basic text input example
