#!/usr/bin/env node

/**
 * Add links to referenced data types in all documentation
 */

const fs = require('fs');
const path = require('path');

const dataTypesDir = path.join(__dirname, '../content/en/ref/query-panel-new/data-types');

console.log('Adding links to referenced data types...\n');

// Map of type names to their documentation files
const typeLinks = {
  'ConfigDict': '[`ConfigDict`](../data-types/ConfigDict.md)',
  'SummaryDict': '[`SummaryDict`](../data-types/SummaryDict.md)',
  'Entity': '[`Entity`](../data-types/Entity.md)',
  'ArtifactType': '[`ArtifactType`](../data-types/ArtifactType.md)',
  'Artifact': '[`Artifact`](../data-types/Artifact.md)',
  'User': '[`User`](../data-types/User.md)',
  'Project': '[`Project`](../data-types/Project.md)',
  'Run': '[`Run`](../data-types/Run.md)',
  'Table': '[`Table`](../data-types/Table.md)',
  'ArtifactVersion': '[`ArtifactVersion`](../data-types/ArtifactVersion.md)'
};

// Process each file
const files = fs.readdirSync(dataTypesDir)
  .filter(file => file.endsWith('.md') && file !== '_index.md');

files.forEach(file => {
  const filePath = path.join(dataTypesDir, file);
  let content = fs.readFileSync(filePath, 'utf8');
  const originalContent = content;
  
  // Find the properties table
  const tableMatch = content.match(/\| Property \| Type \| Description \|\n\| :[-]+.*\n([\s\S]*?)(?:\n\n|$)/);
  
  if (tableMatch) {
    let tableContent = tableMatch[0];
    let modified = false;
    
    // Replace each type with its linked version in the table
    Object.entries(typeLinks).forEach(([typeName, linkedType]) => {
      // Don't link to self
      const currentFile = file.replace('.md', '');
      if (currentFile === typeName) return;
      
      // Match the type in backticks, making sure it's not already linked
      const pattern = new RegExp(`\\| \`${typeName}\` \\|`, 'g');
      if (tableContent.match(pattern)) {
        tableContent = tableContent.replace(pattern, `| ${linkedType} |`);
        modified = true;
      }
      
      // Also handle arrays
      const arrayPattern = new RegExp(`\\| \`Array<${typeName}>\` \\|`, 'g');
      if (tableContent.match(arrayPattern)) {
        tableContent = tableContent.replace(arrayPattern, `| \`Array<\`${linkedType}\`>\` |`);
        modified = true;
      }
    });
    
    if (modified) {
      content = content.replace(tableMatch[0], tableContent);
      fs.writeFileSync(filePath, content);
      console.log(`✅ Updated ${file}`);
    } else {
      console.log(`  - ${file} (no changes needed)`);
    }
  } else {
    console.log(`  - ${file} (no properties table)`);
  }
});

console.log('\n✨ Done! All type references are now properly linked.');
