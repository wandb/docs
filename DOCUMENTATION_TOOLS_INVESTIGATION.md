# Documentation Generation Tools Investigation

## Current Issues with Generated Documentation

Based on analysis of the `/content/en/ref/query-panel/` documentation generated from `generateDocs.ts`:

### Problems Identified

1. **Duplicate Content**: Operations are repeated under both "Chainable Ops" and "List Ops" sections with identical content
2. **Poor Formatting**: 
   - Mixed HTML tags (`<h3>`) with Markdown
   - Minimal table formatting with empty header columns
   - No proper styling or visual hierarchy
3. **Lack of Context**:
   - No meaningful descriptions or explanations
   - Missing code examples
   - No usage scenarios or best practices
   - Repetitive, unhelpful descriptions (e.g., "Returns the name of the artifact" for `artifact-name`)
4. **Poor Cross-referencing**: Basic links but no contextual navigation
5. **No Type Information**: Missing detailed type signatures and interfaces

## Alternative Documentation Generation Tools

### 1. **TypeDoc** ⭐ Recommended
**Best for**: TypeScript API documentation

**Pros**:
- Purpose-built for TypeScript with full type system support
- Generates beautiful, searchable HTML documentation
- Supports JSDoc/TSDoc comments
- Includes inheritance diagrams and type hierarchies
- Plugin ecosystem (themes, custom plugins)
- Supports MDX output for integration with docusaurus/nextra

**Cons**:
- Requires well-commented source code
- May need customization for complex query language documentation

**Example Setup**:
```bash
npm install --save-dev typedoc typedoc-plugin-markdown
```

### 2. **API Extractor + API Documenter**
**Best for**: Enterprise-grade API documentation

**Pros**:
- Microsoft's official tool for TypeScript APIs
- Generates detailed API reports
- Supports custom documentation transformers
- Can output to multiple formats (Markdown, YAML, JSON)
- Excellent for maintaining API contracts
- Built-in versioning support

**Cons**:
- Steeper learning curve
- More complex setup
- May be overkill for smaller projects

### 3. **TSDoc Parser + Custom Generator**
**Best for**: Maximum control over output format

**Pros**:
- Use TSDoc standard for comments
- Build custom Hugo-specific generator
- Full control over output format and structure
- Can generate exactly the markdown format needed

**Cons**:
- Requires development effort
- Maintenance burden

**Example Approach**:
```typescript
import { TSDocParser } from '@microsoft/tsdoc';
// Custom generator to output Hugo-compatible markdown
```

### 4. **Compodoc**
**Best for**: Angular-style documentation (but works for any TS)

**Pros**:
- Beautiful default themes
- Dependency graphs
- Coverage reports
- Search functionality
- Multiple export formats

**Cons**:
- Primarily designed for Angular
- May include unnecessary features

### 5. **Docusaurus with TypeDoc Plugin**
**Best for**: If migrating away from Hugo

**Pros**:
- Modern documentation platform
- Built-in search, versioning, i18n
- TypeDoc plugin for automatic API docs
- React-based for interactive components
- MDX support for dynamic content

**Cons**:
- Would require migration from Hugo
- Different architecture

### 6. **Doxide** (AI-Enhanced)
**Best for**: Automated documentation enhancement

**Pros**:
- Uses AI to generate meaningful descriptions
- Can improve existing documentation
- Understands code context
- Generates examples automatically

**Cons**:
- Requires API key
- May generate incorrect information
- Newer tool, less mature

## Recommended Approach

### Option 1: TypeDoc with Custom Theme (Quick Win)
1. Install TypeDoc with markdown plugin
2. Configure to output Hugo-compatible markdown
3. Create custom templates for query language specifics
4. Add proper TSDoc comments to source

**Benefits**: Fast implementation, maintains current workflow

### Option 2: Custom TSDoc-based Generator (Best Long-term)
1. Parse TypeScript using TSDoc
2. Build custom generator for Hugo format
3. Add query-specific documentation features
4. Generate rich examples and descriptions

**Benefits**: Perfect fit for requirements, full control

### Option 3: Hybrid Approach
1. Use TypeDoc for basic structure
2. Post-process output with custom script
3. Enhance with query-specific examples
4. Add interactive components where needed

## Implementation Recommendations

### Immediate Actions
1. **Add proper TSDoc comments** to source TypeScript files
2. **Test TypeDoc** with markdown output plugin
3. **Create proof-of-concept** with a few query types

### Documentation Improvements Needed
- Add usage examples for each operation
- Include type signatures and parameters
- Provide context and use cases
- Remove duplicate content
- Improve cross-referencing
- Add interactive examples where possible

### Example of Improved Documentation Format

```markdown
## run-config

Returns the configuration object for a run.

### Syntax
\`\`\`typescript
runConfig(run: Run): ConfigDict
\`\`\`

### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `run` | `Run` | The run object to extract configuration from |

### Returns
`ConfigDict` - A typed dictionary containing the run's configuration settings

### Examples

\`\`\`typescript
// Get configuration from a specific run
const config = runConfig(myRun);
console.log(config.learning_rate); // 0.001
\`\`\`

### See Also
- [run-summary](#run-summary) - For summary metrics
- [Run Type Reference](./run.md) - Complete run type documentation
```

## Conclusion

The current documentation lacks context, has formatting issues, and contains duplicated content. **TypeDoc** with custom templates would provide the quickest improvement, while a **custom TSDoc-based generator** would offer the best long-term solution for maintaining Hugo compatibility and adding query-language-specific features.
