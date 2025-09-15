# Alternative to tj-actions/changed-files

## Option 1: Using GitHub's Built-in Context

The `${{ github.event.pull_request.changed_files }}` only gives you a count, not the actual file list. To get file details, you'd need to use the GitHub API.

### Challenges:
1. **No built-in categorization** - Files aren't sorted by added/modified/deleted/renamed
2. **No filtering** - All files mixed together (including non-content files)
3. **API calls required** - Need to call GitHub API to get actual file lists
4. **Complex processing** - Need custom logic to categorize and filter

## Option 2: Custom GitHub Script Implementation

Here's what we'd need to implement:

```javascript
// Rough implementation outline
const { data: files } = await github.rest.pulls.listFiles({
  owner: context.repo.owner,
  repo: context.repo.repo,
  pull_number: context.issue.number,
  per_page: 100
});

// Categorize files
const added = [];
const modified = [];
const deleted = [];
const renamed = [];

for (const file of files) {
  // Filter for relevant paths
  if (!file.filename.match(/^(content|static|assets|layouts|i18n|configs)\//)) {
    continue;
  }
  
  switch (file.status) {
    case 'added':
      added.push(file.filename);
      break;
    case 'modified':
      modified.push(file.filename);
      break;
    case 'removed':
      deleted.push(file.filename);
      break;
    case 'renamed':
      renamed.push({
        old: file.previous_filename,
        new: file.filename
      });
      break;
  }
}
```

### Complexity Assessment:
- **Medium complexity** - Need to rewrite significant portions of both workflows
- **API pagination** - Need to handle if PR has >100 changed files
- **Error handling** - Need robust error handling for API calls
- **Testing burden** - Need extensive testing to ensure feature parity

## Option 3: Alternative GitHub Actions

Some alternatives mentioned in search results:
- `step-security/changed-files` (mentioned as a secure replacement)
- `jitterbit/get-changed-files`
- `yumemi-inc/changed-files`

These would need evaluation for:
- Security track record
- Feature compatibility
- Active maintenance
- Community trust

## Recommendation

Given the complexity of replacing tj-actions/changed-files with custom code, I'd recommend:

1. **Short term**: Investigate if v47 is actually safe (check GitHub releases, security advisories)
2. **Medium term**: Evaluate alternative actions like `step-security/changed-files`
3. **Long term**: Consider implementing custom solution if no suitable alternatives exist

The custom implementation would take approximately 2-3 days to develop and test properly, including:
- Rewriting the file detection logic
- Handling all edge cases (renamed files, pagination, etc.)
- Testing with various PR scenarios
- Updating both workflow files