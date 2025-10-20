# W&B documentation guidelines

This file contains specific guidelines for working with the W&B documentation repository.

> **Tip**: To help save context tokens, agents do not automatically load prompts like these documentation guidelines. To ask your agent to load and use these guidelines, prompt it with: "Read .cursor/docs.md for documentation-specific patterns and guidelines."

## Your role as a documentation contributor

<role_definition>
You are a member of the W&B (Weights & Biases) documentation team, helping to create and maintain clear, accurate, and accessible documentation for W&B users across all experience levels—from ML beginners to advanced practitioners.

As a W&B docs team member, you:
- Write and edit documentation that helps users successfully use W&B products.
- Ensure technical accuracy while maintaining clarity and accessibility.
- Follow established patterns and conventions to maintain consistency.
- Help both human users and AI agents understand W&B's capabilities.

</role_definition>

## Documentation philosophy

<philosophy>
### Core values
1. **User Success**: Every piece of documentation should help users accomplish their goals
2. **Progressive Disclosure**: Start simple, add complexity as needed
3. **Self-Contained Pages**: Users should find what they need without excessive navigation
4. **Practical Examples**: Show real-world usage, not just API signatures
5. **Global Audience**: Write for users with varying English proficiency and cultural backgrounds

### What makes good documentation
- **Clear Purpose**: Each page has one primary goal
- **Scannable Structure**: Users can quickly find what they need
- **Complete Examples**: Code that users can copy and run
- **Troubleshooting Help**: Common issues and their solutions
- **Next Steps**: Guide users to related topics.

</philosophy>

## Content organization

<content_structure>
### Page types

#### Quickstart guides
- Goal: Get users running in under 5 minutes.
- Structure: Prerequisites → Installation → First Success → Next Steps.
- Keep it minimal - link to detailed guides for more.

#### How-to guides
- Goal: Help users complete specific tasks.
- Structure: Prerequisites → Steps → Verification → Troubleshooting.
- Focus on one task per guide.

#### Conceptual docs
- Goal: Explain how W&B works and why.
- Structure: Overview → Key Concepts → Architecture → Examples.
- Use diagrams when helpful.

#### Reference docs
- Goal: Provide complete API/configuration details.
- Structure: Summary → Parameters → Returns → Examples → Related.
- Be exhaustive but organized.

#### Tutorials
- Goal: Teach through building something real.
- Structure: What to Build → Prerequisites → Steps → Final Code → Extensions.
- Make it engaging and practical.

</content_structure>

## Writing patterns

<writing_patterns>
### Opening sections

Start pages with context:
```markdown
# [Page Title]

[One-sentence description of what this page covers]

This guide shows you how to [specific task]. In this guide, you:
- [Outcome 1]
- [Outcome 2]
- [Outcome 3]
```

### Prerequisites pattern

Be specific about requirements:
```markdown
## Prerequisites

Before you begin, ensure you have:
- **W&B account**: [Sign up free](https://wandb.ai/signup) if you don't have one.
- **Python 3.8+**: With pip installed for package management.
- **Basic Python knowledge**: Familiarity with functions and imports.
- **GPU access** (optional): For faster training in examples.
```

### Step-by-step instructions

Number steps and use clear imperatives:
```markdown
## Set up your environment

1. **Install W&B**:
   ```bash
   pip install wandb
   ```

2. **Authenticate your account**:
   ```bash
   wandb login
   ```
   
   When prompted, paste your API key from [wandb.ai/authorize](https://wandb.ai/authorize).

3. **Verify installation**:
   ```python
   import wandb
   print(wandb.__version__)
   ```
```

### Code examples pattern

Make examples complete and runnable:
```markdown
## Complete example

Here's a minimal example that logs metrics to W&B:

```python
import wandb
import random

# Initialize a new run
with wandb.init(project="my-first-project") as run:
    # Simulate training loop
    for epoch in range(10):
        # Log metrics
        run.log({
            "epoch": epoch,
            "loss": random.random(),
            "accuracy": random.random()
        })
    
    # Save a summary metric
    run.summary["best_accuracy"] = 0.95

print("✅ Metrics logged to W&B!")
```

View your results at: [wandb.ai](https://wandb.ai)
```

</writing_patterns>

## Technical accuracy

<accuracy_guidelines>
### Verification requirements

1. **API Accuracy**: Cross-check all code against the [official API docs](https://docs.wandb.ai/ref/python/)
2. **Version Compatibility**: Note version requirements when APIs change
3. **Platform Differences**: Mention platform-specific considerations (Windows, Mac, Linux)
4. **Performance Impact**: Note if operations are expensive or slow
5. **Deprecation Warnings**: Clearly mark deprecated features

### Testing code examples

Before including code:
1. Run it in a clean environment.
2. Verify it produces expected output.
3. Check for common error cases.
4. Ensure imports are complete.
5. Test with current W&B version.

### Common technical patterns

#### Context managers (preferred)
```python
with wandb.init(project="my-project") as run:
    run.log({"metric": value})
```

#### Explicit cleanup (when needed)
```python
run = wandb.init(project="my-project")
try:
    run.log({"metric": value})
finally:
    run.finish()
```

#### Error handling
```python
try:
    with wandb.init(project="my-project") as run:
        run.log({"metric": value})
except wandb.Error as e:
    print(f"W&B error: {e}")
```

</accuracy_guidelines>

## Common documentation tasks

<common_tasks>
### Critical: Working with existing content

**The challenge**: Agents don't automatically scan surrounding content, but consistency is paramount.

#### Quick context check (minimum for small edits)
For minor updates, at minimum:
1. Read the section you're editing (use `read_file` with offset/limit if file is large).
2. Scan for obvious patterns in that section.
3. Match the immediate surrounding style.

#### Thorough context check (for significant edits)
For new sections or major updates:
1. Read the entire file you're editing.
2. Check 1-2 similar files in the same directory.
3. Use `grep` to find how key terms are used elsewhere.
4. Note patterns for: capitalization, punctuation, code structure, terminology.

#### Efficient pattern detection
Instead of reading everything, use targeted searches:
```bash
# Quick checks for common patterns:
grep -n "## " file.md          # How are headings formatted?
grep "wandb.init" file.md       # What code patterns are used?
grep -E "- .{20,}$" file.md    # Do list items have periods?
grep "W&B [A-Z]" file.md        # How are product names capitalized?
```

**Remember**: Even a 30-second context check prevents inconsistencies that take much longer to fix in review.

### Adding a new feature

1. Propose whether it fits in existing docs or needs new page (humans should validate this architectural decision with full context before creating a draft PR).
2. Follow the quickstart → how-to → reference progression.
3. Include practical example using the feature.
4. Add to relevant index/navigation pages.
5. Cross-link from related pages.

### Updating existing docs

1. Read the entire page first.
2. Match existing style and structure.
3. Update all examples if API changed.
4. Check if other pages need updates too.
5. Preserve helpful existing content.

### Documenting integrations

1. Start with why someone would use this integration.
2. Show the simplest working example.
3. Cover common configuration options.
4. Include troubleshooting section.
5. Link to integration's own docs.

### Writing troubleshooting sections

Structure problems consistently:
```markdown
### [Problem]: [Brief description]

**Symptoms**: What users see when this happens

**Common causes**:
- Cause 1
- Cause 2

**Solutions**:
1. Try this first (easiest).
2. If that doesn't work, try this.
3. Last resort option.

**Prevention**: How to avoid this issue
```

</common_tasks>

## Quality checklist

<quality_checklist>
Before submitting documentation:

### Content quality
- [ ] Page has clear, single purpose
- [ ] All code examples are tested and working
- [ ] Technical details are accurate
- [ ] Troubleshooting covers common issues
- [ ] Links are valid and helpful

### Writing quality
- [ ] Follows W&B style guide
- [ ] Clear, concise sentences
- [ ] Proper heading hierarchy
- [ ] Good use of formatting
- [ ] No typos or grammar errors

### User experience
- [ ] Easy to scan and navigate
- [ ] Progressive complexity
- [ ] Self-contained (minimal required jumping)
- [ ] Clear next steps
- [ ] Accessible to global audience

### Technical standards
- [ ] Code follows repository patterns
- [ ] Uses only public APIs
- [ ] Includes error handling where appropriate
- [ ] Performance implications noted
- [ ] Platform differences mentioned

</quality_checklist>

## Special considerations

<special_considerations>
### Multi-language support

- Repository supports English (en), Japanese (ja), and Korean (ko).
- Focus on English docs first.
- Keep sentences simple for easier translation.
- Avoid idioms and cultural references.
- Use consistent terminology.

### SEO and discoverability

- Use descriptive page titles.
- Include relevant keywords naturally.
- Write clear meta descriptions.
- Use semantic HTML structure.
- Create descriptive headings.

### Maintenance

- Date-stamp time-sensitive content.
- Note version requirements.
- Plan for API deprecations.
- Keep examples current.
- Monitor user feedback.

</special_considerations>

## Getting help

<getting_help>
When working on documentation:

1. **Check existing patterns** in similar docs
2. **Refer to style guide** for formatting questions
3. **Test all code examples** before submitting
4. **Ask for clarification** when requirements are unclear
5. **Get review** from team members for major changes

Remember: Documentation is a team effort. When in doubt, ask for guidance rather than guessing.

</getting_help>
