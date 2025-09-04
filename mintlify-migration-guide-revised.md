# Mintlify Migration Guide: From Hugo & Docusaurus (Revised)

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
- **No documented frontmatter URL override** - URLs are path-based by default
- URLs are determined entirely by file path structure
- File at `docs/guides/setup.md` → URL: `/guides/setup`
- For legacy URLs that must be preserved, use redirects
- If manual per-page slugs are a hard requirement, confirm with Mintlify support

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
1. Define the new navigation structure in `mint.json` before moving files
2. Audit all pages with custom URLs in both repositories
3. Create a mapping table of current URLs to required file paths
4. Plan directory restructuring before migration

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
- Mintlify uses `mint.json` for all configuration including redirects
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
      "source": "/beta/:path*",
      "destination": "/v2/:path*"
    }
  ]
}
```

**Key Differences:**
1. Uses JSON format instead of plain text (_redirects) or JavaScript config
2. Wildcard syntax uses path-to-regexp style captures (`:path*`, `:slug*`, etc. - the param name is arbitrary)
3. All redirects are 301 (permanent) by default - no per-rule status code field
4. **Order matters** - put specific rules before broad wildcards when rules overlap

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

3. **Define Navigation Early**
   - Create the navigation structure in `mint.json` first
   - This helps catch path conflicts before moving files

### Phase 2: Content Migration

1. **Use Mintlify's Migration Tools**
   ```bash
   # Install the scraping tool
   npm install @mintlify/scraping@latest -g
   
   # Scrape Docusaurus content
   npx mintlify-scrape section https://weave-docs.wandb.ai
   ```
   
   Note: No automatic tool exists for Hugo, so manual migration is required.

2. **Manual Content Transfer for Hugo**
   - Copy markdown files from `content/en/` to Mintlify structure
   - Remove Hugo-specific frontmatter (url, menu, taxonomies)
   - Convert Hugo shortcodes to MDX components or Markdown callouts
   - Replace admonitions with Mintlify-supported callouts

3. **File Organization**
   - Create directory structure matching desired URLs
   - Move files to their correct locations
   - Ensure `index.md` files for directory landing pages

4. **Internal Link Normalization**
   - Convert absolute links to relative where appropriate
   - Drop file extensions (link to `/guides/setup`, not `/guides/setup.md`)
   - Update anchor formats and ensure section IDs align post-MDX conversion

5. **Static Assets Migration**
   - Move Hugo `static/` and Docusaurus `static/` assets to Mintlify's expected location
   - Update all image and asset references accordingly

### Phase 3: Redirect Implementation

1. **Convert Hugo _redirects**
   ```bash
   # Example conversion
   # From: /app/features/* /guides/runs/:splat
   # To: { "source": "/app/features/:path*", "destination": "/guides/runs/:path*" }
   ```

2. **Convert Docusaurus redirects**
   ```bash
   # From: { from: ['/guides/evaluation/imperative_evaluations'], to: '/guides/evaluation/evaluation_logger' }
   # To: { "source": "/guides/evaluation/imperative_evaluations", "destination": "/guides/evaluation/evaluation_logger" }
   ```

3. **Consolidate and Optimize Redirects**
   - Merge redirects from both platforms
   - Deduplicate and consolidate rules
   - Prefer broad patterns (e.g., `/app/features/:path*` → `/guides/runs/:path*`) over hundreds of single-page rules
   - Place specific rules before wildcards
   - Add to `mint.json`

### Phase 4: Testing and Validation

1. **Use Mintlify's broken links checker**
   ```bash
   npx mintlify broken-links
   ```

2. **Local Development Testing**
   ```bash
   npx mintlify dev
   ```

3. **Test critical user journeys**
   - Verify all major documentation paths work
   - Test redirect chains
   - Validate external links still function

## Special Considerations

### 1. Language Support
- Hugo currently supports multiple languages (en, ja, ko)
- Mintlify doesn't provide first-class multi-locale switching in a single site
- Options:
  - Separate sites (one per language) on Mintlify Cloud
  - Language directories (e.g., `/ja/`, `/ko/`) with separate navigation groups
- Decide early and plan redirects accordingly

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
- Navigation structure must be defined in `mint.json`

### 5. Subpath Hosting
- If currently mounting docs under a subpath (e.g., `wandb.ai/docs`)
- Prefer a docs subdomain on Mintlify Cloud (e.g., `docs.wandb.ai`)
- Deep subpaths may require reverse proxying outside Mintlify

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
   - Custom shortcodes → MDX components
   - Advanced templating → Simplified structure
   - Plugin functionality → Native Mintlify features or workarounds

3. **SEO Impact**: URL changes and redirect chains can affect search rankings
   - Minimize URL changes where possible
   - Implement redirects immediately
   - Monitor search console for issues

## Next Steps

1. Review this guide with your team
2. Create detailed migration checklist including:
   - Internal link normalization tasks
   - Shortcode conversion mapping
   - Static asset inventory
3. Set up Mintlify test environment
4. Begin with small subset migration
5. Iterate based on findings

This migration will require careful planning due to the fundamental differences in how Mintlify handles URLs compared to your current solutions. The lack of manual URL control in Mintlify means significant restructuring of your content organization will be necessary.