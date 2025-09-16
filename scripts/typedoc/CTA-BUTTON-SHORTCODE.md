# ✅ GitHub Links Now Use Hugo CTA Button Shortcode [[memory:6975056]]

## What Changed

Updated the documentation generator to use your custom Hugo `cta-button` shortcode for GitHub source links instead of regular markdown links.

### Before (Regular Markdown)
```markdown
## Source

View the source code on GitHub: [ConfigDict](https://github.com/wandb/core/blob/master/frontends/weave/src/core/ops/types.ts#ConfigDict)
```

### After (Hugo Shortcode)
```markdown
## Source

{{< cta-button githubLink=https://github.com/wandb/core/blob/master/frontends/weave/src/core/ops/types.ts#ConfigDict >}}
```

## Benefits

Using the `cta-button` shortcode provides:

1. **Consistent Styling** - Matches other CTAs in your documentation
2. **Visual Prominence** - Button-style links are more noticeable than text links
3. **Theme Integration** - Respects your Hugo theme's button styling
4. **Maintainability** - Centralized button component in Hugo

## Implementation

The `postprocess-hugo.js` script now:
```javascript
const sourceSection = `
## Source

{{< cta-button githubLink=${sourceUrl} >}}
`;
```

## Examples

### Data Type (ConfigDict)
```markdown
## Source

{{< cta-button githubLink=https://github.com/wandb/core/blob/master/frontends/weave/src/core/ops/types.ts#ConfigDict >}}
```

### Operations (Run Operations)
```markdown
## Source

{{< cta-button githubLink=https://github.com/wandb/core/blob/master/frontends/weave/src/core/ops/runOperations.ts >}}
```

## Result

All documentation pages now have:
- A prominent **Source** section
- A styled CTA button linking to GitHub
- Consistent formatting with your documentation standards
- Better visual hierarchy and user experience

The source links are now properly integrated with your Hugo theme's component system!
