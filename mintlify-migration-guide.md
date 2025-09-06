# Mintlify Migration Guide: From Hugo & Docusaurus

## Executive Summary

This guide provides a comprehensive migration strategy for moving your documentation from Hugo (wandb/docs) and Docusaurus (wandb/weave) to Mintlify. Based on my research, here are the key findings and recommendations.

## Key Differences: URL Handling

### 1. **Manual URL Assignment**

**Current Solutions:**
- **Hugo**: Uses `url:` frontmatter key to manually assign URLs
  - Example: `url: /support/:filename` 
- **Docusaurus**: Uses `slug:` frontmatter key to customize URLs
  - Example: `slug: /` (makes the page the root)

**Mintlify Approach:**
- **No manual URL assignment via frontmatter**
- URLs are determined entirely by file path structure
- File at `docs/guides/setup.md` → URL: `/guides/setup`
- This is a significant departure from your current flexibility

**Migration Impact:**
- You'll need to restructure your file system to match desired URLs
- Custom URL patterns (like Hugo's `:filename` placeholder) won't work
- Files that currently use custom URLs will need to be moved to match their intended paths

### 2. **File Path to URL Mapping**

**Current Solutions:**
- **Hugo**: Uses permalinks configuration and frontmatter overrides
- **Docusaurus**: Combines `routeBasePath` config with file structure

**Mintlify Approach:**
- Direct 1:1 mapping between file paths and URLs
- No configuration options to modify this behavior
- Directory structure = URL structure

**Recommendations:**
1. Audit all pages with custom URLs in both repositories
2. Create a mapping table of current URLs to required file paths
3. Plan directory restructuring before migration

## Redirect Handling

### Current Solutions

**Hugo (wandb/docs):**
- Uses Netlify's `_redirects` file format
- Located at `static/_redirects`
- Contains 472 lines of redirects
- Supports both individual redirects and wildcard patterns
- Example:
  ```
  /app/features/alerts/ /guides/runs/alert/ 301
  /app/features/* /guides/runs/:splat
  ```

**Docusaurus (wandb/weave):**
- Uses `@docusaurus/plugin-client-redirects` plugin
- Configured in `docusaurus.config.ts`
- Example:
  ```javascript
  redirects: [
    {
      from: ['/guides/evaluation/imperative_evaluations'],
      to: '/guides/evaluation/evaluation_logger',
    }
  ]
  ```

### Mintlify Solution

**Configuration File:** 
- Despite some conflicting information, Mintlify uses `mint.json` (not `docs.json`) for configuration
- Redirects are defined in the `redirects` array

**Format:**
```json
{
  "redirects": [
    {
      "source": "/old-path",
      "destination": "/new-path"
    },
    {
      "source": "/beta/:slug*",
      "destination": "/v2/:slug*"
    }
  ]
}
```

**Key Differences:**
1. Uses JSON format instead of plain text (_redirects) or JavaScript config
2. Wildcard syntax uses `:slug*` instead of `:splat` or `*`
3. All redirects are 301 (permanent) by default

## Migration Strategy

### Phase 1: Content Audit and Planning

1. **Export Current URL Structure**
   - Extract all URLs from Hugo content with custom `url:` frontmatter
   - List all Docusaurus pages with custom `slug:` values
   - Document the current redirect rules from both platforms

2. **Design New File Structure**
   - Map current URLs to required Mintlify file paths
   - Identify conflicts where multiple pages might need the same path
   - Plan for URL changes that might be necessary

### Phase 2: Content Migration

1. **Use Mintlify's Migration Tools**
   ```bash
   # Install the scraping tool
   npm install @mintlify/scraping@latest -g
   
   # Scrape Docusaurus content
   mintlify-scrape section https://weave-docs.wandb.ai
   ```
   
   Note: No automatic tool exists for Hugo, so manual migration is required.

2. **Manual Content Transfer for Hugo**
   - Copy markdown files from `content/en/` to Mintlify structure
   - Remove Hugo-specific frontmatter (url, menu, taxonomies)
   - Adjust internal links to match new paths

3. **File Organization**
   - Create directory structure matching desired URLs
   - Move files to their correct locations
   - Ensure `index.md` files for directory landing pages

### Phase 3: Redirect Implementation

1. **Convert Hugo _redirects**
   ```bash
   # Example conversion script needed
   # From: /app/features/* /guides/runs/:splat
   # To: { "source": "/app/features/:slug*", "destination": "/guides/runs/:slug*" }
   ```

2. **Convert Docusaurus redirects**
   - Transform JavaScript config to JSON format
   - Update path references

3. **Consolidate Redirects**
   - Merge redirects from both platforms
   - Resolve any conflicts
   - Add to `mint.json`

### Phase 4: Testing and Validation

1. **Use Mintlify's broken links checker**
   ```bash
   mint broken-links
   ```

2. **Test critical user journeys**
   - Verify all major documentation paths work
   - Test redirect chains
   - Validate external links still function

## Special Considerations

### 1. Language Support
- Hugo currently supports multiple languages (en, ja, ko)
- Research if Mintlify supports internationalization
- May need separate deployments per language

### 2. Dynamic URL Patterns
- Hugo's `:filename` pattern in `url: /support/:filename` won't work
- These pages will need explicit file names matching their URLs

### 3. Large Redirect Volume
- With 472+ redirects from Hugo alone, consider:
  - Whether all redirects are still needed
  - If some can be consolidated using wildcard patterns
  - Performance implications of large redirect lists

### 4. Search and Navigation
- Both current solutions have robust search
- Ensure Mintlify's search indexes all content properly
- May need to adjust navigation structure in `mint.json`

## Recommended Timeline

1. **Week 1-2**: Complete audit and planning
2. **Week 3-4**: Set up test Mintlify instance and migrate subset
3. **Week 5-6**: Full content migration and redirect setup
4. **Week 7-8**: Testing, fixes, and documentation updates
5. **Week 9**: Deployment and monitoring

## Risk Mitigation

1. **URL Changes**: Some URLs may need to change due to Mintlify limitations
   - Document all changes
   - Communicate to users
   - Implement comprehensive redirects

2. **Feature Gaps**: Some Hugo/Docusaurus features may not have Mintlify equivalents
   - Custom shortcodes
   - Advanced templating
   - Plugin functionality

3. **SEO Impact**: URL changes and redirect chains can affect search rankings
   - Minimize URL changes where possible
   - Implement redirects immediately
   - Monitor search console for issues

## Next Steps

1. Review this guide with your team
2. Create detailed migration checklist
3. Set up Mintlify test environment
4. Begin with small subset migration
5. Iterate based on findings

This migration will require careful planning due to the fundamental differences in how Mintlify handles URLs compared to your current solutions. The lack of manual URL control in Mintlify means significant restructuring of your content organization will be necessary.