# Future Enhancements for TypeDoc Post-Processor

## Property Ordering Enhancement

### Problem
TypeDoc may generate properties in alphabetical order or based on source code declaration order, which doesn't match the logical order shown in examples.

### Proposed Solution
Enhance `postprocess-hugo.js` to automatically reorder properties based on the example:

```javascript
function reorderPropertiesByExample(content) {
  // 1. Parse the example code block
  const exampleMatch = content.match(/```typescript\nconst \w+: \w+ = \{([^}]+)\}/s);
  if (!exampleMatch) return content;
  
  // 2. Extract field names from example in order
  const exampleFields = [];
  const fieldPattern = /^\s*(\w+):/gm;
  let match;
  while ((match = fieldPattern.exec(exampleMatch[1])) !== null) {
    exampleFields.push(match[1]);
  }
  
  // 3. Extract all property sections
  const properties = extractPropertySections(content);
  
  // 4. Reorder properties to match example
  const reorderedProps = exampleFields
    .map(field => properties[field])
    .filter(Boolean);
  
  // 5. Add any remaining properties not in example
  for (const [name, prop of Object.entries(properties)) {
    if (!exampleFields.includes(name)) {
      reorderedProps.push(prop);
    }
  }
  
  // 6. Rebuild properties section
  return rebuildPropertiesSection(content, reorderedProps);
}
```

### Benefits
- Automatic consistency between examples and properties
- No manual intervention needed after regeneration
- Better user experience with logical property ordering

## Additional Enhancements

### 1. Property Completeness Check
Warn if properties in example aren't documented:
```javascript
function checkPropertyCompleteness(content) {
  const exampleFields = parseExampleFields(content);
  const documentedFields = parseDocumentedProperties(content);
  
  const missing = exampleFields.filter(f => !documentedFields.includes(f));
  if (missing.length > 0) {
    console.warn(`Missing properties: ${missing.join(', ')}`);
  }
}
```

### 2. Type Inference from Examples
Extract types from example values when TypeDoc doesn't provide them:
```javascript
function inferTypeFromExample(value) {
  if (value.startsWith('new Date')) return 'Date';
  if (value.startsWith('"') || value.startsWith("'")) return 'string';
  if (value === 'true' || value === 'false') return 'boolean';
  if (!isNaN(value)) return 'number';
  if (value.startsWith('{')) return 'object';
  if (value.startsWith('[')) return 'array';
  return 'unknown';
}
```

### 3. Cross-Reference Validation
Ensure all type references link to valid documentation:
```javascript
function validateTypeReferences(content) {
  const typeRefs = content.matchAll(/\[`(\w+)`\]\(\.\.\/data-types\/(\w+)\.md\)/g);
  for (const [full, type, file] of typeRefs) {
    if (!fs.existsSync(`../data-types/${file}.md`)) {
      console.warn(`Broken reference: ${full}`);
    }
  }
}
```

## Implementation Priority

1. **High Priority**: Property reordering based on examples
2. **Medium Priority**: Property completeness checking
3. **Low Priority**: Type inference and validation

These enhancements would make the documentation generation more robust and reduce manual intervention.
