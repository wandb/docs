# W&B technical documentation guidelines

This guide covers technical aspects of W&B documentation, including code examples, SDK usage, and API documentation standards.

## Code examples

### General requirements
- Strive for code examples that are immediately runnable when practical
- Always include pip install commands to show required dependencies
- Balance between completeness and focus:
  - Use focused snippets with ellipses (...) when showing specific concepts
  - Provide complete code when the full context is essential
  - Consider linking to complete notebooks in wandb/examples or Weave cookbooks for complex examples
- Include all necessary imports for the code shown
- Test complete code examples before including them in documentation

### Language-specific guidelines
- Use language-appropriate conventions and idioms
- Follow established style guides for each language (PEP 8 for Python, etc.)
- Include language-specific dependency files when creating new examples

## W&B SDK guidelines

### Current SDK compliance
- Always use the current SDK methods and patterns
- Check for deprecated methods and update accordingly
- Reference official SDK documentation for the latest patterns

### Internal code restrictions
- **Never use or document internal code**: Classes, methods, or attributes starting with underscore (_) are internal
- Only document public API methods
- Do not expose implementation details

### Import statements
- Always show complete import statements
- Use standard import patterns:
  ```python
  import wandb
  from wandb import Api
  ```

## API documentation

### Method documentation
- Include all required parameters
- Document optional parameters with defaults
- Provide clear return value descriptions
- Include usage examples

### Error handling
- Document common errors and their solutions
- Show proper error handling in examples
- Explain error messages users might encounter

## Code quality standards

### Completeness
- Use ellipses (...) thoughtfully:
  - Acceptable when focusing on specific concepts to avoid distraction
  - Use clear comments to indicate what is omitted
  - Ensure the shown code illustrates the key point effectively
- Provide complete examples when:
  - The full context is necessary for understanding
  - The example is meant to be copy-pasted and run as-is
- Include error handling where appropriate

### Dependencies
- List all required packages with versions
- Create appropriate dependency files:
  - `requirements.txt` for Python
  - `package.json` for JavaScript
  - Other language-appropriate dependency files

### Security
- Never include real API keys or sensitive data
- Use environment variables for configuration
- Show secure practices in examples

## Technical accuracy

### Version compatibility
- Specify minimum required versions
- Note any version-specific features
- Test examples with supported versions

### Platform considerations
- Note any platform-specific requirements
- Provide alternatives for different environments
- Test on multiple platforms when possible

## Documentation updates

### When updating code examples
- Verify the code runs successfully
- Update any related explanatory text
- Check for consistency with other examples
- Update dependency versions if needed

### Deprecation handling
- Clearly mark deprecated features
- Provide migration paths
- Update examples to use current methods

## Best practices

### Code formatting
- Use consistent indentation (4 spaces for Python)
- Follow language-specific formatting conventions
- Keep line lengths reasonable for documentation display

### Comments
- Add helpful comments for complex logic
- Explain W&B-specific concepts
- Don't over-comment obvious code

### Example structure
1. Brief explanation of what the example demonstrates
2. Complete, runnable code
3. Expected output or behavior
4. Common variations or extensions
