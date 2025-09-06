# PR Review Comments for #1594

Thank you for creating this comprehensive migration guide! I've reviewed it and have some feedback based on my own research and the additional insights provided. Overall, the guide is excellent and captures the key challenges well.

## ✅ Strengths of the Guide

1. **Accurate identification of the core challenge**: The lack of manual URL assignment in Mintlify is correctly identified as the biggest migration hurdle
2. **Comprehensive redirect analysis**: The 472 redirects from Hugo's `_redirects` file is an important finding
3. **Clear phasing**: The audit → restructure → migrate → redirects → test sequence is logical
4. **Good examples**: The redirect conversion examples are helpful

## 🔧 Corrections Needed

### 1. **CLI Commands** (Line 158)
```diff
- mint broken-links
+ npx mintlify broken-links
```
All Mintlify CLI commands should use `npx`. Also consider adding:
```bash
npx mintlify dev  # For local development testing
```

### 2. **Configuration File Consistency**
While you correctly identify `mint.json` as the config file (line 75), consider removing the parenthetical "(not `docs.json`)" to avoid confusion. Just state it definitively.

### 3. **Wildcard Parameter Naming** (Line 96)
The guide shows `:slug*` but should clarify that the parameter name is arbitrary:
```diff
- Wildcard syntax uses `:slug*` instead of `:splat` or `*`
+ Wildcard syntax uses path-to-regexp style captures (`:slug*`, `:path*`, etc. - the param name is arbitrary)
```

## 📝 Important Additions to Consider

### 1. **Redirect Rule Ordering**
Add a note that order matters when redirect rules overlap - specific rules should come before broad wildcards.

### 2. **Navigation Planning**
Add to Phase 1:
- Define the navigation structure in `mint.json` BEFORE moving files
- This helps catch path conflicts early

### 3. **Content Migration Details**
The guide should include:
- **Internal link normalization**: Convert to relative links, drop `.md` extensions
- **Shortcode conversion**: Hugo shortcodes → MDX components or Markdown callouts
- **Static assets**: Move `static/` directories and update references
- **Admonitions**: Convert Docusaurus admonitions to Mintlify callouts

### 4. **i18n Considerations**
Expand the language support section with concrete options:
- Separate Mintlify sites per language
- Or single site with `/en/`, `/ja/`, `/ko/` directories

### 5. **Subpath Hosting**
Add a note about preferring subdomain hosting (docs.wandb.ai) over subpaths on Mintlify Cloud.

## 💡 Suggested Enhancements

### 1. **Manual URL Assignment Caveat**
Soften the "No manual URL assignment" statement to:
> "No documented frontmatter URL override. URLs are path-based by default. For legacy URLs that must be preserved, use redirects. If manual per-page slugs are a hard requirement, confirm with Mintlify support."

### 2. **Redirect Consolidation**
In Phase 3, emphasize consolidating the 472+ redirects using patterns where possible to improve maintainability.

### 3. **Testing Section**
Add `npx mintlify dev` for local testing before the broken links check.

## 📎 Additional Resources

I've created a revised version of the guide incorporating these suggestions: `mintlify-migration-guide-revised.md`

The revised version includes:
- All CLI command corrections
- Expanded migration tasks (links, assets, shortcodes)
- Clearer redirect ordering guidance
- More detailed i18n options
- Subpath hosting considerations

Both versions complement each other well - yours provides the solid foundation and structure, while the revisions add operational details that will help during actual migration execution.