# Alt Text Checker

This directory contains scripts for the Alt Text Checker GitHub Action.

## Overview

The Alt Text Checker automatically reviews pull requests for missing or empty alt text in image references and suggests improvements based on context and existing patterns in the repository.

## Features

- **Multi-format support**: Checks HTML `<img>` tags, Markdown images `![alt](src)`, and Hugo `{{< img >}}` shortcodes
- **Context-aware suggestions**: Analyzes surrounding text to generate relevant alt text suggestions
- **Pattern learning**: Studies existing alt text in the repository to model suggestions after successful patterns
- **Accessibility focus**: Provides educational content about alt text best practices
- **PR integration**: Automatically comments on pull requests with detailed suggestions

## Files

### `alt-text-checker.py`

The main Python script that:

1. **Analyzes changed files** in pull requests for image references
2. **Extracts context** around each image for better suggestions
3. **Learns from existing patterns** by studying alt text already in the repository
4. **Generates intelligent suggestions** based on:
   - File paths and names
   - Surrounding content
   - W&B-specific terminology
   - Existing successful patterns
5. **Posts detailed PR comments** with suggested fixes

#### Key Components

- **Pattern Recognition**: Regex patterns for different image formats
- **Context Analysis**: Extracts surrounding text for context-aware suggestions
- **Smart Suggestions**: Uses W&B terminology and common patterns for relevant alt text
- **GitHub Integration**: Posts formatted comments with actionable suggestions

#### Supported Image Formats

```markdown
# Markdown images
![alt text](path/to/image.png)

# Hugo shortcodes
{{< img src="/images/example.png" alt="Description" >}}

# HTML img tags
<img src="/images/example.png" alt="Description">
```

## How It Works

### 1. Trigger
The action runs on pull request events (opened, synchronize, reopened) when files in `content/**/*.md` or `content/**/*.html` are changed.

### 2. Analysis
- Scans all changed files for image references
- Identifies images missing alt text or with empty alt attributes
- Collects existing alt text patterns from the repository for learning

### 3. Context Extraction
- Analyzes text surrounding each image
- Removes markdown formatting for cleaner context
- Identifies key terms and concepts

### 4. Suggestion Generation
- Matches context against W&B-specific terminology
- Considers file type (GIF vs static image)
- Applies existing patterns from the repository
- Generates concise, descriptive alt text

### 5. PR Feedback
- Creates a comprehensive comment listing all issues
- Provides specific suggestions for each image
- Includes educational content about alt text best practices
- Offers actionable code snippets for easy fixes

## Example Output

When the action finds missing alt text, it posts a comment like:

```markdown
## 🖼️ Alt Text Issues Found

### Issues Found:

#### 1. Hugo img shortcode in `content/en/guides/example.md` (line 15)

**Issue:** Missing alt text

**Current:**
{{< img src="/images/dashboard.png" >}}

**Suggested:**
{{< img src="/images/dashboard.png" alt="W&B dashboard view" >}}

**Context:** This screenshot shows the main dashboard where you can view...
```

## Alt Text Best Practices

The checker enforces these accessibility guidelines:

- **Be descriptive**: Explain what the image shows in context
- **Be concise**: Keep it focused and relevant
- **Avoid redundancy**: Don't use phrases like "image of" or "picture showing"
- **Consider context**: Alt text should add value to the surrounding content
- **Empty for decoration**: Use `alt=""` only for purely decorative images

## Development

To test the script locally:

```bash
# Set environment variables
export GITHUB_TOKEN="your_token"
export PR_NUMBER="123"
export REPO_OWNER="wandb"
export REPO_NAME="docs"
export CHANGED_FILES="content/en/example.md content/en/another.md"

# Run the script
python .github/scripts/alt-text-checker.py
```

## Dependencies

- `requests`: GitHub API interactions
- `beautifulsoup4`: HTML parsing
- `lxml`: XML/HTML processing
- Standard library: `re`, `os`, `subprocess`, `pathlib` 