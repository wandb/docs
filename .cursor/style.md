# W&B documentation style guide

This file contains detailed style guidelines for creating and editing W&B documentation.

## Style hierarchy

<style_hierarchy>
1. **Match existing content first**: When editing near existing content, match its style to maintain consistency.
2. **Google Developer Style Guide**: Primary reference for new content.
3. **Microsoft Style Guide**: Secondary reference when Google doesn't cover something.
4. **Chicago Manual of Style**: Tertiary reference for edge cases.

### Critical: Checking existing style

**Before making any edit**, agents should:
1. Read the entire file being edited to identify established patterns.
2. Check similar files in the same directory for conventions.
3. Look for patterns in:
   - Heading formatting (sentence case vs title case)
   - List punctuation (periods or not)
   - Code example structure
   - Term usage (for example, "W&B Run" vs "run")
   - Voice and tense choices

**Example**: If editing `/content/en/guides/track/log.md`, first check:
- How are headings capitalized in this file?
- How are similar concepts described in other files in `/content/en/guides/track/`?
- What code patterns are used in existing examples?

**Remember**: Consistency within the existing corpus trumps perfect adherence to this style guide. Only apply new style rules when:
- Creating entirely new content
- Doing an explicit style refactor
- The existing style directly contradicts technical accuracy
</style_hierarchy>

## Detailed style rules

<heading_style>
### Headings
- Use sentence case for all headings (capitalize only the first word and proper nouns).
- Keep headings concise and descriptive.
- Use imperative mood for task-based headings.
- Examples:
  - ✓ "Configure your environment"
  - ✓ "Get started with W&B"
  - ✓ "Integrate with GitHub Actions"
  - ✗ "Configuring Your Environment"
  - ✗ "Getting Started with W&B"
  - ✗ "Integrating With GitHub Actions"
</heading_style>

<product_naming>
### Product and feature names

#### Company name
- Use "W&B" (not "Weights & Biases") in running text.
- Exception: Use full name in legal documents or first-time introductions to new audiences.
- **Avoid using "W&B" in headings**: Shorter headings are clearer, easier to scan, and better for SEO. When editing existing content, match the heading style already in use. For new content, omit "W&B" from headings unless necessary for disambiguation (for example, when comparing W&B features to other tools).

#### Product name patterns
First mention on a page:
- "W&B Models" → subsequent mentions: "Models"
- "W&B Weave" → subsequent mentions: "Weave"
- "W&B Launch" → subsequent mentions: "Launch"
- "W&B Automations" → subsequent mentions: "Automations"

#### Feature capitalization
Some features become lowercase after first mention:
- "W&B Run" → "run" (because it's an API object)
- "W&B Sweep" → "sweep" (because it's an API object)
- "W&B Report" → "report" (common noun usage)

Special cases:
- "W&B artifact" - always lowercase, even on first mention (matches API naming)
- "W&B project" - always lowercase (common noun)

#### Examples in context
- "Configure W&B Automations to monitor your runs. Automations can send alerts when specific conditions are met."
- "Create a W&B Run to track your experiment. The run captures metrics, system information, and outputs."
- "W&B artifacts help version your datasets. Each artifact is immutable once created."
</product_naming>

<writing_style>
### Voice and tone

#### General guidelines
- **Direct and concise**: Get to the point quickly.
- **Second person and imperative mood**: Use imperative commands for instructions ("Configure the API key"). The "you" is implied and makes instructions more direct. Use explicit "you" for explanations and descriptions.
- **Active voice**: "Configure the API key" not "The API key should be configured".
- **Present tense**: Describe what happens, not what will happen.
- **Simple verbs**: "Use" not "Utilize", "Start" not "Initiate".

#### Examples
Instructions (use imperative):
✓ "Configure authentication by setting the API key."
✓ "Set the API key in your environment variables."
✗ "You should configure authentication by setting the API key."

Explanations (use "you" explicitly):
✓ "You can track experiments with W&B."
✓ "When you run this code, W&B logs the metrics."
✗ "Experiments will be tracked by W&B."
</writing_style>

<accessibility_guidelines>
### Accessibility guidelines

#### Emoji usage
**Never use emojis in documentation content**. They create multiple problems:
- Screen readers may announce them disruptively.
- Translation tools handle them inconsistently.
- Cultural interpretations vary.
- They don't display consistently across platforms.

**Limited exceptions** (use sparingly):
- Style guides showing correct/incorrect examples (✓/✗).
- Feature comparison matrices.
- Only when they add clarity without relying on color.

**Prefer**:
- Plain text descriptions.
- Unicode symbols (✓) over emoji equivalents (✅).
- Clear, descriptive text over any symbol.

#### Plain language
Replace complex terms and Latin phrases:
- ✓ "for example" → ✗ "e.g."
- ✓ "that is" → ✗ "i.e."
- ✓ "and so on" → ✗ "etc."
- ✓ "versus" → ✗ "vs."
- ✓ "through" → ✗ "via"
</accessibility_guidelines>

<punctuation_rules>
### Punctuation guidelines

#### Quotes and apostrophes
Always use straight quotes and apostrophes:
- ✓ `"straight quotes"` and `'apostrophes'`
- ✗ "curly quotes" and 'smart apostrophes'

**Why**: Smart quotes can break code examples and cause parsing issues. They often creep in from copy-pasting from rich text editors.

#### Simplify punctuation
Avoid in regular documentation text:
- Exclamation points (!)
- Ellipses (...)
- Semicolons (;)
- Em dashes (—) and en dashes (–)
- Complex inline lists with nested punctuation.

#### Lists
- Use sentence case for all list items.
- **Include periods for most list items**. This is a tech writing convention.
- **Use colons for definition-style lists**: When a list item introduces sub-items or provides an explanation, use a colon instead of a hyphen.
- Include periods for:
  - Complete sentences
  - Long phrases (more than 5-7 words)  
  - Items with explanatory text or descriptions
  - Any item that contains multiple words beyond simple labels
- Omit periods only for very short fragments (2-3 words that are simple labels).

Examples of proper formatting:
- ✓ **Bold term**: Explanation of what this means.
- ✓ **Another term**: Its definition or description.
- ✗ **Wrong format** - Should use colon, not hyphen.

Examples of when to use periods:
- ✓ "Configure your API key." (complete sentence)
- ✓ "Repository access for creating branches and pushing changes." (long phrase)
- ✓ "Python 3.8 or higher with pip installed." (detailed requirement)
- ✓ "Code examples with proper syntax highlighting." (descriptive item)
- ✓ "API reference" (short fragment - no period)
- ✓ "Quick links" (short label - no period)
</punctuation_rules>

<code_style>
### Code and technical terms

#### Inline code
Use backticks for:
- File names: `config.yaml`
- Commands: `wandb login`
- Code elements: `wandb.init()`
- Directory paths: `runbooks/`
- Function names: `log_metrics()`
- Variable names: `api_key`

#### Code blocks
Always specify the language:
```python
# Python code
with wandb.init(project="my-project") as run:
    run.log({"accuracy": 0.95})
```

```bash
# Shell commands
wandb login
pip install wandb
```

```yaml
# YAML configuration
project: my-project
entity: my-team
```

#### Code example best practices
1. **Follow public API patterns** from https://docs.wandb.ai/ref/python/
2. **Use context managers** for `wandb.init()`
3. **Include imports** and minimal setup
4. **Make examples runnable** without additional context
5. **Test when possible** to ensure correctness
6. **Avoid placeholder names** like "foo" or "bar" - use meaningful examples
</code_style>

<formatting_standards>
### Text formatting standards

#### Emphasis
- **Bold** for UI elements: "Click the **New Project** button"
- *Italics* sparingly for introducing new terms
- `Code` for technical terms, commands, and code elements
- Avoid combining multiple formatting styles.

#### Links
- Use descriptive link text.
- ✓ "See the [API reference](link) for details"
- ✗ "Click [here](link) for more information"
- ✗ "The API reference is available [at this link](link)"

#### File paths
- Use forward slashes even for Windows paths.
- Start with `/` for absolute paths
- Omit leading `/` for relative paths.
- Use backticks: `/home/user/project` or `data/dataset.csv`
</formatting_standards>

<common_patterns>
### Common documentation patterns

#### Prerequisites sections
```markdown
## Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher installed.
- A W&B account. [Sign up here](https://wandb.ai/signup) if needed.
- Basic familiarity with machine learning concepts.
```

#### Step-by-step instructions
```markdown
## Getting started

Follow these steps to set up W&B:

1. **Install the W&B library**:
   ```bash
   pip install wandb
   ```

2. **Authenticate your account**:
   ```bash
   wandb login
   ```

3. **Initialize your first run**:
   ```python
   import wandb
   
   with wandb.init(project="quickstart") as run:
       run.log({"metric": 1.0})
   ```
```

#### Troubleshooting sections
```markdown
## Troubleshooting

### Issue: Authentication fails

**Symptoms**: Error message "Failed to authenticate"

**Solution**: 
1. Verify your API key is correct.
2. Check your internet connection.
3. Ensure your firewall allows HTTPS traffic to api.wandb.ai.
```
</common_patterns>

<writing_tips>
### Writing tips for AI agents

1. **Check surrounding content first** - Match the style of nearby content
2. **Be consistent within a page** - Don't switch styles mid-document
3. **Avoid over-formatting** - Clean, simple formatting is best
4. **Think about scanning** - Users scan documentation, make it easy with clear headings and lists
5. **Include examples** - Show, don't just tell
6. **Test your instructions** - Can someone follow them without additional context?
7. **Review for accessibility** - Would this work for non-native English speakers?
8. **Keep sentences short** - Aim for 20 words or less when possible
9. **One idea per paragraph** - Don't pack too much into a single paragraph
10. **Use transition words** - Help readers follow your logic with words like "First", "Next", "However"
</writing_tips>

## Important reminders

<reminders>
- This style guide supplements but doesn't replace the main rules in `rules.md`
- When in doubt, consistency trumps perfection.
- Style changes should be separate from content changes.
- Always prioritize clarity and user success over strict style adherence.
- Check existing content for established patterns before creating new ones.
</reminders>
