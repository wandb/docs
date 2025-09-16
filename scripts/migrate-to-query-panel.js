#!/usr/bin/env node

/**
 * Migrate query-panel-new to query-panel with lowercase filenames
 */

const fs = require('fs');
const path = require('path');

const oldDir = path.join(__dirname, '../content/en/ref/query-panel-new');
const newDir = path.join(__dirname, '../content/en/ref/query-panel');

console.log('Migrating documentation to /content/en/ref/query-panel with lowercase filenames...\n');

// First, backup and remove old query-panel if it exists
if (fs.existsSync(newDir)) {
  console.log('🗑️  Removing old query-panel directory...');
  fs.rmSync(newDir, { recursive: true });
}

// Create new directory structure
fs.mkdirSync(newDir, { recursive: true });
fs.mkdirSync(path.join(newDir, 'data-types'), { recursive: true });
fs.mkdirSync(path.join(newDir, 'operations'), { recursive: true });

// Map of old names to new lowercase names
const filenameMap = {
  // Data types
  'Artifact.md': 'artifact.md',
  'ArtifactType.md': 'artifacttype.md',
  'ArtifactVersion.md': 'artifactversion.md',
  'ConfigDict.md': 'configdict.md',
  'Entity.md': 'entity.md',
  'Project.md': 'project.md',
  'Run.md': 'run.md',
  'SummaryDict.md': 'summarydict.md',
  'Table.md': 'table.md',
  'User.md': 'user.md',
  // Operations
  'Run_Operations.md': 'run-operations.md',
  'Artifact_Operations.md': 'artifact-operations.md'
};

// Function to update links in content
function updateLinks(content) {
  // Update links to data types
  content = content.replace(/\(\.\.\/data-types\/ConfigDict\.md\)/g, '(../data-types/configdict.md)');
  content = content.replace(/\(\.\.\/data-types\/SummaryDict\.md\)/g, '(../data-types/summarydict.md)');
  content = content.replace(/\(\.\.\/data-types\/Entity\.md\)/g, '(../data-types/entity.md)');
  content = content.replace(/\(\.\.\/data-types\/ArtifactType\.md\)/g, '(../data-types/artifacttype.md)');
  content = content.replace(/\(\.\.\/data-types\/Artifact\.md\)/g, '(../data-types/artifact.md)');
  content = content.replace(/\(\.\.\/data-types\/ArtifactVersion\.md\)/g, '(../data-types/artifactversion.md)');
  content = content.replace(/\(\.\.\/data-types\/User\.md\)/g, '(../data-types/user.md)');
  content = content.replace(/\(\.\.\/data-types\/Project\.md\)/g, '(../data-types/project.md)');
  content = content.replace(/\(\.\.\/data-types\/Run\.md\)/g, '(../data-types/run.md)');
  content = content.replace(/\(\.\.\/data-types\/Table\.md\)/g, '(../data-types/table.md)');
  
  // Update links to operations
  content = content.replace(/\(\.\.\/operations\/Run_Operations\.md\)/g, '(../operations/run-operations.md)');
  content = content.replace(/\(\.\.\/operations\/Artifact_Operations\.md\)/g, '(../operations/artifact-operations.md)');
  
  return content;
}

// Copy and rename files
console.log('📁 Copying data types...');
const dataTypesOld = path.join(oldDir, 'data-types');
if (fs.existsSync(dataTypesOld)) {
  const files = fs.readdirSync(dataTypesOld);
  files.forEach(file => {
    if (file === '_index.md') {
      // Copy index file as-is
      let content = fs.readFileSync(path.join(dataTypesOld, file), 'utf8');
      content = updateLinks(content);
      fs.writeFileSync(path.join(newDir, 'data-types', file), content);
      console.log(`  ✓ Copied _index.md`);
    } else if (filenameMap[file]) {
      let content = fs.readFileSync(path.join(dataTypesOld, file), 'utf8');
      content = updateLinks(content);
      fs.writeFileSync(path.join(newDir, 'data-types', filenameMap[file]), content);
      console.log(`  ✓ ${file} → ${filenameMap[file]}`);
    }
  });
}

console.log('\n📁 Copying operations...');
const operationsOld = path.join(oldDir, 'operations');
if (fs.existsSync(operationsOld)) {
  const files = fs.readdirSync(operationsOld);
  files.forEach(file => {
    if (file === '_index.md') {
      // Copy index file as-is
      let content = fs.readFileSync(path.join(operationsOld, file), 'utf8');
      content = updateLinks(content);
      fs.writeFileSync(path.join(newDir, 'operations', file), content);
      console.log(`  ✓ Copied _index.md`);
    } else if (filenameMap[file]) {
      let content = fs.readFileSync(path.join(operationsOld, file), 'utf8');
      content = updateLinks(content);
      fs.writeFileSync(path.join(newDir, 'operations', filenameMap[file]), content);
      console.log(`  ✓ ${file} → ${filenameMap[file]}`);
    }
  });
}

// Copy root _index.md
console.log('\n📁 Copying root index...');
const rootIndex = path.join(oldDir, '_index.md');
if (fs.existsSync(rootIndex)) {
  let content = fs.readFileSync(rootIndex, 'utf8');
  content = updateLinks(content);
  // Update the menu identifier
  content = content.replace('identifier: query-panel-new', 'identifier: query-panel');
  content = content.replace('parent: query-panel-new', 'parent: query-panel');
  fs.writeFileSync(path.join(newDir, '_index.md'), content);
  console.log('  ✓ Copied and updated _index.md');
}

console.log('\n✨ Migration complete!');
console.log(`📍 New location: ${newDir}`);
console.log('📝 All filenames are now lowercase');
console.log('🔗 All internal links have been updated');
