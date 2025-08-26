# W&B Documentation Style Guide

This guide outlines writing style preferences for W&B documentation. Follow these guidelines to maintain consistency across all documentation.

## General Writing Principles

### Readability
- Use American English spellings and idioms and avoid commonwealth English spelling or idioms.
- Aim for 8th grade reading comprehension level or simpler when possible.
- Use clear, concise, complete sentences.
- Where possible, break up long sentences.
- Where possible, use simple words over complex ones, and define new terms on their first use.
- Break complex topics into digestible sections.
- Explain technical terms on first use.
- Provide examples for abstract concepts.
- Prefer active voice over passive voice.

### Tone
- Use second person ("you") when addressing the reader.
- Use imperative mood for instructions ("Run the command" not "You should run the command").
- Use professional but approachable language that is direct and informative.
- Focus on helping users accomplish their goals.
- Provide use cases when describing features.
- Avoid overly casual language.
- Avoid jokes or puns, which may not make sense to everyone.

**Examples of appropriate tone:**
- Good: "Follow these steps to configure your first integration."
- Good: "Configure your integration by completing these steps."
- Good: "This guide shows you how to set up monitoring for your models."

**Too casual:**
- Avoid: "Hey there! Time to set up some cool integrations!"
- Avoid: "Let's dive into this awesome feature!"

**Too formal/academic:**
- Avoid: "The user must configure the integration module."
- Avoid: "One should ensure proper configuration of the system."

## Content Formatting

### Accessibility
- Use descriptive link text so the reader knows what to expect. Avoid generic link text like "here", "click here", or "this".
- Avoid able-ist language. For example, use "turn on" and "turn off" rather than "enable" and "disable".
- When adding or updating images or content with image references, add alt text if it is missing.

### Things to avoid
- **No emojis**: Do not add emojis to content. Remove them if you see them in existing content.
  - Exception: Checkmarks (✅) and X marks (❌) may be used in feature matrices or support tables to clearly indicate "supported" and "not supported" respectively. These Unicode symbols are accessible to screen readers and clearer than empty cells or the confusing convention where X means "yes".
- Avoid redundant phrases like "documentation" when linking (use "guide" not "guide documentation").
- Avoid unnecessary formatting such as boldface on link text.
- Avoid Latin phrasing when possible. Writers sometimes struggle to use it correctly. For example, use "for example" instead of "e.g.".

### Punctuation and plain text
- Keep punctuation simple. Avoid complex punctuation that is often misused:
  - Em dashes (—) - use commas or parentheses instead
  - En dashes (–) - use "to" for ranges (for example, "2020 to 2023")
  - Semicolons (;) - use periods to create separate sentences
- Use straight quotes (") not smart quotes (""). Since we write in Markdown, rely on the static site generator to handle typographic conversions.
- Avoid other rich text formatting that might paste in from word processors. Keep everything in plain text.

### Links
- Use descriptive link text that explains what users will find.
- Link text should generally align with the target section or page title.
- For internal links: use descriptive text only.
- Prefer action-oriented language: "To configure X" rather than "For configuring X".
- When linking to API or SDK references, use the page or section title as the link text.
- When linking to external documentation, align link text with the target page title and clarify the source after the link. For example: "See the [W&B Integration] section in the OpenAI documentation".

### Third-party names and trademarks
- Use third-party company names, products, and trademarks exactly as the company uses them.
- Follow the company's official branding, not how it might appear in logos or stylized text.
- Examples:
  - "GitHub" not "Github" or "github"
  - "PyTorch" not "Pytorch" or "pytorch"  
  - "Red Hat" not "RedHat" or "redhat"
  - "macOS" not "MacOS" or "Mac OS"

### W&B-Specific terminology

- For W&B-specific products:
  - On first mention in a page, add "W&B". For example, "W&B Weave".
  - On following mentions, omit the "W&B". For example, "Weave".
- For other W&B-specific features and terms:
    - On first mention in a page, add "W&B" and capitalize the feature or term. For example, "W&B Run" or "W&B Sweeps".
    - On following mentions, omit the W&B and lowercase the feature or term. For example, "run" or "sweep".
    - Exception: When referring to API classes or methods, use the exact capitalization (for example, `wandb.Artifact()` or "the Artifact class").
- For compound terms containing "W&B":
    - Include "W&B" when needed for clarity, especially when distinguishing from other services.
    - For example, "W&B API key" when discussing multiple services, but "API key" when the context is clear.
- Refer to the W&B user interface as "W&B App". Avoid "W&B UI" or "W&B App UI" and fix them where you notice them.
  - In following mentions, omit the "W&B". 
- Refer to W&B deployment types consistently and fix inconsistencies when you notice them:
    - W&B Multi-tenant Cloud
    - W&B Dedicated Cloud
    - W&B Self-Managed

### Headings
- Don't add "W&B" to headings, even if it is the first mention. It lengthens headings and increases the risk of a broken anchor.
- Keep headings concise and descriptive.
- Use sentence case for headings, not title case.

## Style references
We aim to keep to a small set of style guidelines. When in doubt, consult these resources in order:
1. Google Developer Style Guide (freely available)
2. Microsoft Style Guide (freely available)
3. Chicago Manual of Style

### Consistency
- Use consistent terminology throughout a document.
- Maintain consistent formatting and structure.
- Follow established patterns for similar content types.
    - When adding content to an existing page, use the existing content as context.
    - When adding a page to an existing section, use the existing content in the section as context.

## Writing principles and judgment calls

Technical writing often requires balancing competing priorities. When guidelines seem to conflict, consider:

- **Context matters**: A heading under a descriptive parent can be more concise.
- **Purpose over rules**: Focus on helping the reader accomplish their goal.
- **Progressive disclosure**: Start simple, add detail as needed.
- **Consistency within sections**: Follow patterns established in nearby content.

Common judgment calls:
- **Concise vs. descriptive headings**: Favor action-oriented clarity (for example, "Track experiments" not just "Experiments").
- **Complete vs. focused information**: Include what readers need for the task at hand, link to details.
- **Technical accuracy vs. simplicity**: Be accurate but approachable, define terms when needed.

When in doubt, look at similar content in the docs for patterns to follow.
