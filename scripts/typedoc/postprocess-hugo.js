#!/usr/bin/env node

/**
 * Post-processes TypeDoc markdown output for Hugo compatibility
 * Enhances documentation with Hugo-specific features and formatting
 * Organizes output into proper directory structure
 */

const fs = require('fs');
const path = require('path');

// Output directory relative to this script
const OUTPUT_DIR = path.join(__dirname, '../../content/en/ref/query-panel');

/**
 * Get readable type name and description
 */
function getReadableTypeName(fileName) {
  const typeDescriptions = {
    'Run': { name: 'Run', description: 'Experiment run with metadata and metrics' },
    'Artifact': { name: 'Artifact', description: 'Versioned files or directories' },
    'ArtifactVersion': { name: 'ArtifactVersion', description: 'Specific artifact version' },
    'ArtifactType': { name: 'ArtifactType', description: 'Artifact type definition' },
    'ConfigDict': { name: 'ConfigDict', description: 'Configuration parameters' },
    'SummaryDict': { name: 'SummaryDict', description: 'Summary metrics' },
    'User': { name: 'User', description: 'User information' },
    'Project': { name: 'Project', description: 'Project details' },
    'Entity': { name: 'Entity', description: 'Team or user entity' },
    'Table': { name: 'Table', description: 'Tabular data' }
  };
  
  return typeDescriptions[fileName] || { 
    name: fileName, 
    description: 'W&B data type' 
  };
}

/**
 * Process a single markdown file
 */
function processFile(filePath) {
  const fileName = path.basename(filePath);
  console.log(`Processing: ${fileName}`);
  
  let content = fs.readFileSync(filePath, 'utf8');
  
  // Add/enhance front matter
  content = enhanceFrontMatter(content, filePath);
  
  // Ensure data types have examples
  content = ensureDataTypeExample(content, fileName);
  
  // Fix markdown formatting
  content = fixMarkdownFormatting(content, filePath);
  
  // Enhance cross-references
  content = enhanceCrossReferences(content, filePath);
  
  // Format code examples
  content = formatCodeExamples(content);
  
  // Add links to documented types in property tables
  content = addTypeLinksInTables(content, filePath);
  
  fs.writeFileSync(filePath, content);
}

/**
 * Enhance Hugo front matter
 */
function enhanceFrontMatter(content, filePath) {
  const fileName = path.basename(filePath, '.md');
  const isIndex = fileName === '_index';
  
  // Extract existing front matter
  const frontMatterMatch = content.match(/^---\n([\s\S]*?)\n---/);
  let frontMatter = {};
  
  if (frontMatterMatch) {
    // Parse existing front matter
    const lines = frontMatterMatch[1].split('\n');
    lines.forEach(line => {
      const [key, ...valueParts] = line.split(':');
      if (key && valueParts.length) {
        frontMatter[key.trim()] = valueParts.join(':').trim();
      }
    });
  }
  
  // Enhance front matter
  if (!frontMatter.title) {
    // Clean up the title - remove prefixes and format nicely
    frontMatter.title = fileName
      .replace(/^W_B_Query_Expression_Language\./, '') // Remove prefix if still present
      .replace(/_/g, ' ')  // Replace underscores with spaces
      .replace(/-/g, ' ')  // Replace hyphens with spaces
      .replace(/\b\w/g, c => c.toUpperCase()); // Capitalize words
  }
  
  if (!frontMatter.description) {
    // Extract first paragraph as description
    const descMatch = content.match(/\n\n(.+?)\n\n/);
    if (descMatch) {
      frontMatter.description = descMatch[1].replace(/[#*`]/g, '');
    }
  }
  
  // Don't add menu entries to individual files - they'll inherit from _index.md cascade
  
  // Build new front matter
  let newFrontMatter = '---\n';
  newFrontMatter += `title: ${frontMatter.title}\n`;
  if (frontMatter.description) {
    newFrontMatter += `description: ${frontMatter.description}\n`;
  }
  newFrontMatter += '---\n';
  
  // Replace or add front matter
  if (frontMatterMatch) {
    content = content.replace(/^---\n[\s\S]*?\n---/, newFrontMatter);
  } else {
    content = newFrontMatter + '\n' + content;
  }
  
  // Remove any duplicate title that might appear right after front matter
  // This can happen if TypeDoc includes title in both front matter and content
  const titlePattern = new RegExp(`^(---\\n[\\s\\S]*?\\n---\\n)(title: ${frontMatter.title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\n)?`, 'm');
  content = content.replace(titlePattern, '$1');
  
  // Also clean up any duplicate title on line 4 (common issue)
  const lines = content.split('\n');
  if (lines[3] && lines[3].startsWith('title:')) {
    lines[3] = '';
  }
  content = lines.join('\n');
  
  // Remove excessive blank lines after front matter
  content = content.replace(/^(---\n[\s\S]*?\n---\n)\n+/, '$1\n');
  
  return content;
}

/**
 * Ensure data types have structure examples
 */
function ensureDataTypeExample(content, fileName) {
  // Only process data type files
  if (!fileName || fileName.includes('Operations') || fileName === '_index.md') {
    return content;
  }
  
  // Check if example already exists
  if (content.includes('**`Example`**') || content.includes('## Example')) {
    return content;
  }
  
  // Extract type name from filename
  const typeName = fileName.replace('.md', '');
  
  // Generate a default example if specific one not found
  const defaultExamples = {
    'Artifact': `**\`Example\`**

\`\`\`typescript
const artifact: Artifact = {
  id: "artifact_abc123",
  name: "model-weights",
  type: artifactType,
  description: "Trained model weights",
  aliases: ["latest", "production"],
  createdAt: new Date("2024-01-15")
};
\`\`\``,
    'Entity': `**\`Example\`**

\`\`\`typescript
const entity: Entity = {
  id: "entity_abc123",
  name: "my-team",
  isTeam: true
};
\`\`\``,
    'Project': `**\`Example\`**

\`\`\`typescript
const project: Project = {
  name: "my-awesome-project",
  entity: entity,
  createdAt: new Date("2023-01-01"),
  updatedAt: new Date("2024-01-20")
};
\`\`\``,
    'Table': `**\`Example\`**

\`\`\`typescript
const table: Table = {
  columns: ["epoch", "loss", "accuracy"],
  data: [
    [1, 0.5, 0.75],
    [2, 0.3, 0.85],
    [3, 0.2, 0.90]
  ]
};
\`\`\``,
    'ArtifactType': `**\`Example\`**

\`\`\`typescript
const artifactType: ArtifactType = {
  name: "model"
};
\`\`\``,
    'ArtifactVersion': `**\`Example\`**

\`\`\`typescript
const artifactVersion: ArtifactVersion = {
  id: "version_xyz789",
  version: "v3",
  versionIndex: 3,
  aliases: ["latest", "production"],
  createdAt: new Date("2024-01-15"),
  metadata: {
    accuracy: 0.95,
    model_type: "transformer"
  }
};
\`\`\``,
    'User': `**\`Example\`**

\`\`\`typescript
const user: User = {
  id: "user_123",
  username: "john_doe",
  name: "John Doe",
  email: "john@example.com"
};
\`\`\``,
    'Run': `**\`Example\`**

\`\`\`typescript
const run: Run = {
  id: "run_abc123",
  name: "sunny-dawn-42",
  state: "finished",
  config: {
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 10
  },
  summaryMetrics: {
    loss: 0.023,
    accuracy: 0.95,
    val_accuracy: 0.93
  },
  createdAt: new Date("2024-01-15T10:30:00Z"),
  updatedAt: new Date("2024-01-15T14:45:00Z")
};
\`\`\``,
    'ConfigDict': `**\`Example\`**

\`\`\`typescript
const config: ConfigDict = {
  learning_rate: 0.001,
  batch_size: 32,
  epochs: 100,
  optimizer: "adam",
  model_name: "resnet50",
  dropout_rate: 0.2,
  dataset: "imagenet",
  device: "cuda",
  seed: 42
};
\`\`\`

ConfigDict stores hyperparameters and settings. Accepts strings, numbers, booleans, arrays, nested objects.
W&B Tables/Artifacts appear as structured reference objects with metadata when logged.`,
    'SummaryDict': `**\`Example\`**

\`\`\`typescript
const summary: SummaryDict = {
  final_loss: 0.023,
  final_accuracy: 0.95,
  best_val_accuracy: 0.961,
  best_epoch: 87,
  total_steps: 50000,
  training_time: 3600.5,
  early_stopped: false
};
\`\`\`

SummaryDict stores final metrics and results. Accepts strings, numbers, booleans, arrays, nested objects.
W&B Tables/Artifacts/Histograms appear as structured objects with metadata when logged.`
  };
  
  const example = defaultExamples[typeName];
  if (!example) {
    return content;
  }
  
  // Insert example after version/Since section, before Properties
  const insertPattern = /(\d+\.\d+\.\d+)\n\n/;
  if (content.match(insertPattern)) {
    content = content.replace(insertPattern, `$1\n\n${example}\n\n`);
  } else {
    // Insert before Properties section if no version found
    content = content.replace(/## (Properties|Indexable)/, `${example}\n\n## $1`);
  }
  
  return content;
}

/**
 * Fix markdown formatting issues
 */
function fixMarkdownFormatting(content, filePath) {
  // Remove H1 headers - title is already in front matter
  // Matches lines like "# Interface: ConfigDict" or "# Module: Artifact Operations"
  content = content.replace(/^#\s+.+\n\n?/gm, '');
  
  // Remove bold "Description" header - unnecessary
  content = content.replace(/\*\*`Description`\*\*\n\n/gm, '');
  
  // Remove "Chainable Operations Functions" header
  content = content.replace(/## Chainable Operations Functions\n\n/gm, '');
  
  // For operations files, bump H3 operations to H2
  if (filePath.includes('/operations/')) {
    // Convert H3 operations to H2
    content = content.replace(/^### ([a-zA-Z]+)$/gm, '## $1');
    
    // Keep subsections at H4 (don't bump them) so they don't clutter the TOC
    // Parameters, Examples, and See Also stay at H4
  }
  
  // Remove "Defined in" sections - source repo is private
  content = content.replace(/#### Defined in\n\n.+\n\n?/gm, '');
  
  // Remove "Since" version sections - private repo version info
  content = content.replace(/\*\*`Since`\*\*\n\n[\d.]+\n\n?/gm, '');
  
  // Remove broken module references (e.g., [W&B Query Expression Language](../modules/W_B_Query_Expression_Language.md).ConfigDict)
  // These are redundant since we already know the context from the navigation
  content = content.replace(/\[W&B Query Expression Language\]\([^)]*\)\.(\w+)/g, '');
  
  // Also handle variations like [modules/W_B_Query_Expression_Language](...)
  content = content.replace(/\[modules\/W_B_Query_Expression_Language\]\([^)]*\)\.(\w+)/g, '');
  
  // Fix cross-references to data types (TypeDoc generates ../interfaces/ links)
  content = content.replace(/\[([^\]]+)\]\(\.\.\/interfaces\/([^)]+)\.md\)/g, (match, text, file) => {
    // Remove W_B_Query_Expression_Language prefix if present and convert to lowercase
    let cleanFile = file.replace(/^W_B_Query_Expression_Language\./, '');
    cleanFile = cleanFile.toLowerCase();
    return `[${text}](../data-types/${cleanFile}.md)`;
  });
  
  // Fix same-page anchor links - remove filename for portability
  // Matches patterns like [text](SameFileName.md#anchor) and converts to [text](#anchor)
  const fileName = path.basename(filePath, '.md');
  const anchorPattern = new RegExp(`\\[([^\\]]+)\\]\\(${fileName}\\.md(#[^)]+)\\)`, 'g');
  content = content.replace(anchorPattern, '[$1]($2)');
  
  // Remove Table of contents section - Hugo auto-generates this
  content = content.replace(/## Table of contents\n\n(### .+\n\n)?(-\s+\[.+\]\(.+\)\n)+\n?/gm, '');
  
  // Fix table formatting
  content = content.replace(/\|\s*Parameter\s*\|\s*Type\s*\|\s*Description\s*\|/g,
    '| Parameter | Type | Required | Description |');
  
  // Remove Returns sections - redundant since return type is in signature
  // Multiple patterns to catch different formats
  content = content.replace(/#### Returns\n\n.*?\n\n.*?(?=\n####|\n##|\n___|\n$)/gs, '');
  content = content.replace(/#### Returns\n\n`[^`]+`\n\n[^\n#]+/gm, '');
  content = content.replace(/#### Returns\n\n.+?\n(?=\n|$)/gm, '');
  content = content.replace(/#### Returns\n\n\| Type \| Description \|\n\| :------ \| :------ \|\n\|[^\n]+\|\n\n?/gm, '');
  
  // Remove duplicate headers
  content = content.replace(/^(#{1,6})\s+(.+)\n\n\1\s+\2$/gm, '$1 $2');
  
  // Fix example formatting - remove bold "Example" text
  content = content.replace(/\*\*`Example`\*\*\n\n/gm, '');
  
  // For non-operations files, convert H3 example headings to H4 with "Example: " prefix
  if (!filePath.includes('/operations/')) {
    // But preserve H3 for function/method names
    const functionNames = [
      'artifactLink', 'artifactName', 'artifactVersionAlias', 'artifactVersionCreatedAt',
      'artifactVersionDigest', 'artifactVersionNumber', 'artifactVersionSize', 'artifactVersions',
      'runConfig', 'runCreatedAt', 'runDuration', 'runHeartbeatAt', 'runId', 'runJobType',
      'runLoggedArtifactVersion', 'runLoggedArtifactVersions', 'runName', 'runRuntime',
      'runSummary', 'runTags', 'runUrl', 'runUsedArtifactVersions', 'runUser'
    ];
    
    // Split into lines for more precise processing
    const lines = content.split('\n');
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].startsWith('### ')) {
        const headingText = lines[i].substring(4);
        // If it's not a function name, it's likely an example heading
        if (!functionNames.includes(headingText)) {
          lines[i] = `#### Example: ${headingText}`;
        }
      }
    }
    content = lines.join('\n');
  }
  
  // Fix "See" section formatting - convert bold to H4
  // Keep at H4 level so it doesn't clutter the TOC
  content = content.replace(/\*\*`See`\*\*\n\n/gm, '#### See Also\n\n');
  
  // Convert function signatures to code blocks
  // Matches: ▸ **functionName**(params): returnType
  content = content.replace(/▸ \*\*([^*]+)\*\*\(([^)]*)\): (.+)$/gm, (match, funcName, params, returnType) => {
    // Remove backticks from parameters since they'll be in a code block
    let cleanParams = params.replace(/`/g, '');
    let cleanReturnType = returnType.replace(/`/g, '');
    
    // Remove markdown links - extract just the text from [text](url) patterns
    cleanParams = cleanParams.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
    cleanReturnType = cleanReturnType.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
    
    // Don't escape pipes - they're valid TypeScript union type syntax
    cleanReturnType = cleanReturnType.replace(/\\\|/g, '|');
    
    return `\`\`\`typescript\n${funcName}(${cleanParams}): ${cleanReturnType}\n\`\`\``;
  });
  
  // Fix code block language hints
  content = content.replace(/```ts\b/g, '```typescript');
  content = content.replace(/```js\b/g, '```javascript');
  
  // Clean up escaped pipes in TypeScript code blocks and property descriptions
  // Union types shouldn't be escaped (e.g., string | undefined)
  const lines = content.split('\n');
  let inCodeBlock = false;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('```typescript')) {
      inCodeBlock = true;
    } else if (lines[i] === '```') {
      inCodeBlock = false;
    } else if (inCodeBlock && lines[i].includes('\\|')) {
      lines[i] = lines[i].replace(/\\\|/g, '|');
    } else if (lines[i].includes('\\|') && lines[i].includes('``')) {
      // Also fix escaped pipes in property descriptions with inline code
      lines[i] = lines[i].replace(/\\\|/g, '|');
    }
  }
  content = lines.join('\n');
  
  // Add proper spacing around headers
  content = content.replace(/\n(#{1,6}\s+.+)\n/g, '\n\n$1\n\n');
  
  // Fix list formatting
  content = content.replace(/^\*\s+/gm, '- ');
  
  // Clean up any resulting multiple blank lines
  content = content.replace(/\n\n\n+/g, '\n\n');
  
  return content;
}

/**
 * Enhance cross-references
 */
function enhanceCrossReferences(content, filePath) {
  // Convert TypeDoc links to Hugo refs
  content = content.replace(/\[([^\]]+)\]\(\.\/([^)]+)\.md\)/g,
    '[$1]({{< relref "$2" >}})');
  
  // Don't add generic "See Also" sections
  // Keep specific cross-references that TypeDoc generates
  
  return content;
}

/**
 * Add links to documented types in property tables and function signatures
 */
function addTypeLinksInTables(content, filePath) {
  // Map of type names to their documentation files (with lowercase filenames)
  const typeLinks = {
    'ConfigDict': '[`ConfigDict`](../data-types/configdict.md)',
    'SummaryDict': '[`SummaryDict`](../data-types/summarydict.md)',
    'Entity': '[`Entity`](../data-types/entity.md)',
    'ArtifactType': '[`ArtifactType`](../data-types/artifacttype.md)',
    'Artifact': '[`Artifact`](../data-types/artifact.md)',
    'User': '[`User`](../data-types/user.md)',
    'Project': '[`Project`](../data-types/project.md)',
    'Run': '[`Run`](../data-types/run.md)',
    'Table': '[`Table`](../data-types/table.md)',
    'ArtifactVersion': '[`ArtifactVersion`](../data-types/artifactversion.md)'
  };
  
  // Get current file name to avoid self-linking
  const currentFile = path.basename(filePath, '.md');
  
  // Find property tables and add links
  const tableMatch = content.match(/\| Property \| Type \| Description \|\n\| :[-]+.*\n([\s\S]*?)(?:\n\n|$)/);
  
  if (tableMatch) {
    let tableContent = tableMatch[0];
    
    Object.entries(typeLinks).forEach(([typeName, linkedType]) => {
      // Don't link to self
      if (currentFile === typeName) return;
      
      // Replace the type with its linked version
      const pattern = new RegExp(`\\| \`${typeName}\` \\|`, 'g');
      tableContent = tableContent.replace(pattern, `| ${linkedType} |`);
      
      // Also handle arrays
      const arrayPattern = new RegExp(`\\| \`Array<${typeName}>\` \\|`, 'g');
      tableContent = tableContent.replace(arrayPattern, `| \`Array<\`${linkedType}\`>\` |`);
    });
    
    content = content.replace(tableMatch[0], tableContent);
  }
  
  // Also add links in function signatures (for operations)
  if (filePath.includes('/operations/')) {
    Object.entries(typeLinks).forEach(([typeName, linkedType]) => {
      // Match type in return position of function signatures: ): TypeName
      const returnPattern = new RegExp(`\\): ${typeName}(\\n|\\s|$)`, 'g');
      content = content.replace(returnPattern, `): ${linkedType}$1`);
      
      // Match type with union: TypeName | undefined
      const unionPattern = new RegExp(`\\): ${typeName} \\|`, 'g');
      content = content.replace(unionPattern, `): ${linkedType} |`);
    });
  }
  
  return content;
}

/**
 * Format code examples
 */
function formatCodeExamples(content) {
  // Ensure proper syntax highlighting
  content = content.replace(/```\n([\s\S]*?)\n```/g, (match, code) => {
    // Detect language if not specified
    if (code.includes('const ') || code.includes('function ')) {
      return '```typescript\n' + code + '\n```';
    }
    return match;
  });
  
  // Add copy button hint
  content = content.replace(/```typescript/g, '```typescript {linenos=false}');
  
  return content;
}

/**
 * Organize files into proper directory structure
 */
function organizeFiles() {
  console.log('Organizing files into proper structure...\n');
  
  const dataTypesDir = path.join(OUTPUT_DIR, 'data-types');
  const operationsDir = path.join(OUTPUT_DIR, 'operations');
  
  // Create directories if they don't exist
  if (!fs.existsSync(dataTypesDir)) {
    fs.mkdirSync(dataTypesDir, { recursive: true });
  }
  if (!fs.existsSync(operationsDir)) {
    fs.mkdirSync(operationsDir, { recursive: true });
  }
  
  // Move interface files to data-types with lowercase names
  const interfacesDir = path.join(OUTPUT_DIR, 'interfaces');
  if (fs.existsSync(interfacesDir)) {
    const interfaceFiles = fs.readdirSync(interfacesDir)
      .filter(file => file.endsWith('.md'));
    
    interfaceFiles.forEach(file => {
      const oldPath = path.join(interfacesDir, file);
      // Remove W_B_Query_Expression_Language prefix and convert to lowercase
      let newFileName = file.replace(/^W_B_Query_Expression_Language\./, '');
      newFileName = newFileName.toLowerCase();
      const newPath = path.join(dataTypesDir, newFileName);
      fs.renameSync(oldPath, newPath);
      console.log(`Moved ${file} to data-types/${newFileName}`);
    });
    
    // Remove empty interfaces directory
    fs.rmdirSync(interfacesDir);
  }
  
  // Move operation modules to operations with lowercase kebab-case names
  const modulesDir = path.join(OUTPUT_DIR, 'modules');
  if (fs.existsSync(modulesDir)) {
    const moduleFiles = fs.readdirSync(modulesDir)
      .filter(file => file.includes('Operations.md'));
    
    moduleFiles.forEach(file => {
      const oldPath = path.join(modulesDir, file);
      // Convert to kebab-case: Run_Operations.md → run-operations.md
      const newFileName = file.replace(/_/g, '-').toLowerCase();
      const newPath = path.join(operationsDir, newFileName);
      fs.renameSync(oldPath, newPath);
      console.log(`Moved ${file} to operations/${newFileName}`);
    });
    
    // Delete any remaining files in modules (like W_B_Query_Expression_Language.md)
    const remainingFiles = fs.readdirSync(modulesDir);
    remainingFiles.forEach(file => {
      const filePath = path.join(modulesDir, file);
      fs.unlinkSync(filePath);
      console.log(`Removed redundant file: ${file}`);
    });
    
    // Remove empty modules directory
    fs.rmdirSync(modulesDir);
  }
  
  // Remove redundant modules.md file if it exists
  const modulesFile = path.join(OUTPUT_DIR, 'modules.md');
  if (fs.existsSync(modulesFile)) {
    fs.unlinkSync(modulesFile);
    console.log('Removed redundant modules.md');
  }
  
  // Create _index.md files for subdirectories
  createIndexFiles();
}

/**
 * Create _index.md files for proper Hugo menu structure
 */
function createIndexFiles() {
  // Get list of data type files for dynamic generation
  const dataTypesDir = path.join(OUTPUT_DIR, 'data-types');
  let dataTypesList = '';
  
  if (fs.existsSync(dataTypesDir)) {
    const dataTypeFiles = fs.readdirSync(dataTypesDir)
      .filter(file => file.endsWith('.md') && file !== '_index.md')
      .sort();
    
    dataTypesList = dataTypeFiles.map(file => {
      const name = file.replace('.md', '');
      const readableType = getReadableTypeName(name);
      return `- **[${readableType.name}](${file})** - ${readableType.description}`;
    }).join('\n');
  }
  
  // Create data-types/_index.md
  const dataTypesIndex = `---
title: Data Types
description: Core data types used in the W&B Query Expression Language
menu:
  reference:
    parent: query-panel-generated
    identifier: query-panel-generated-data-types
    weight: 20
cascade:
  menu:
    reference:
      parent: query-panel-generated-data-types
---

# Data Types

Core data type definitions for the W&B Query Expression Language.

## Available Types

${dataTypesList || 'See the individual type documentation for details on each data structure.'}
`;
  
  fs.writeFileSync(path.join(OUTPUT_DIR, 'data-types', '_index.md'), dataTypesIndex);
  
  // Create operations/_index.md
  const operationsIndex = `---
title: Operations
description: Operations for querying and manipulating W&B data
menu:
  reference:
    parent: query-panel-generated
    identifier: query-panel-generated-operations
    weight: 10
cascade:
  menu:
    reference:
      parent: query-panel-generated-operations
---

# Operations

Functions and operations for querying and manipulating W&B data in the Query Expression Language.

## Available Operations

See the individual operation modules for detailed documentation.
`;
  
  fs.writeFileSync(path.join(OUTPUT_DIR, 'operations', '_index.md'), operationsIndex);
}

/**
 * Process all markdown files in the output directory
 */
function processAllFiles() {
  console.log('Post-processing TypeDoc output for Hugo...\n');
  
  if (!fs.existsSync(OUTPUT_DIR)) {
    console.error(`Output directory not found: ${OUTPUT_DIR}`);
    console.log('Run TypeDoc first to generate the documentation.');
    process.exit(1);
  }
  
  // First organize files into proper structure
  organizeFiles();
  
  // Process main directory files
  const mainFiles = fs.readdirSync(OUTPUT_DIR)
    .filter(file => file.endsWith('.md'));
  
  mainFiles.forEach(file => {
    processFile(path.join(OUTPUT_DIR, file));
  });
  
  // Process subdirectories
  const subdirs = fs.readdirSync(OUTPUT_DIR)
    .filter(item => fs.statSync(path.join(OUTPUT_DIR, item)).isDirectory());
  
  subdirs.forEach(dir => {
    const dirPath = path.join(OUTPUT_DIR, dir);
    const dirFiles = fs.readdirSync(dirPath)
      .filter(file => file.endsWith('.md'));
    
    dirFiles.forEach(file => {
      processFile(path.join(dirPath, file));
    });
  });
  
  console.log('\n✅ Post-processing complete!');
  console.log('Documentation organized into:');
  console.log('  - operations/ (Run and Artifact operations)');
  console.log('  - data-types/ (Type definitions)');
}

// Run the post-processor if called directly
if (require.main === module) {
  processAllFiles();
}

module.exports = { processFile, processAllFiles };