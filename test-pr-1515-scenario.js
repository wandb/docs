const fs = require('fs');
const path = require('path');

// Load pageurls.json
const pageMap = JSON.parse(fs.readFileSync('public/pageurls.json', 'utf8'));

// Function to check if a file is an include
function isIncludeFile(filePath) {
  return /^content\/[^\/]+\/_includes\//.test(filePath);
}

// Function to find pages that use a specific include
function findPagesUsingInclude(includePath) {
  const includeFileName = includePath.split('/').pop();
  const escapedFileName = includeFileName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const includePattern = `[^"']*\\/_includes\\/${escapedFileName}`;
  const pagesUsingInclude = [];
  
  // Search through all pages in pageMap
  for (const page of pageMap) {
    if (page.path && page.path.endsWith('.md')) {
      try {
        // Construct the correct file path: content/<lang>/<page.path>
        const contentPath = path.join('content', page.lang || 'en', page.path);
        if (fs.existsSync(contentPath)) {
          const content = fs.readFileSync(contentPath, 'utf8');
          // Check for both readfile syntaxes with flexible path matching
          const pattern1 = new RegExp(`\\{\\{%\\s*readfile\\s+(?:file=)?["'](${includePattern})["']\\s*%\\}\\}`, 'i');
          const pattern2 = new RegExp(`\\{\\{<\\s*readfile\\s+file=["'](${includePattern})["']\\s*>\\}\\}`, 'i');
          
          if (pattern1.test(content) || pattern2.test(content)) {
            pagesUsingInclude.push(page);
          }
        }
      } catch (e) {
        // Skip if file can't be read
      }
    }
  }
  
  return pagesUsingInclude;
}

// Test with PR #1515 scenario
const includeFile = 'content/en/_includes/project-visibility-settings.md';

console.log('Testing PR #1515 scenario...');
console.log(`Include file: ${includeFile}`);
console.log(`Is include file: ${isIncludeFile(includeFile)}`);

const pages = findPagesUsingInclude(includeFile);
console.log(`\nFound ${pages.length} pages using this include:`);
pages.forEach(page => {
  console.log(`  - ${page.title} (${page.path})`);
});

// Check specific files
console.log('\nChecking specific files from PR #1515:');
const filesToCheck = [
  'content/en/guides/models/track/project-page.md',
  'content/en/support/kb-articles/project_make_public.md'
];

filesToCheck.forEach(file => {
  if (fs.existsSync(file)) {
    const content = fs.readFileSync(file, 'utf8');
    const includeMatch = content.match(/\{\{%\s*readfile\s+file=["']([^"']+)["']\s*%\}\}/);
    if (includeMatch) {
      console.log(`\n${file}:`);
      console.log(`  Include reference: ${includeMatch[1]}`);
    }
  }
});