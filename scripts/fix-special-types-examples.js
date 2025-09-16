#!/usr/bin/env node

/**
 * Add concrete examples for W&B special types in ConfigDict and SummaryDict
 */

const fs = require('fs');
const path = require('path');

const dataTypesDir = path.join(__dirname, '../content/en/ref/query-panel-new/data-types');

console.log('Adding concrete W&B special type examples...\n');

// Update ConfigDict
let configContent = fs.readFileSync(path.join(dataTypesDir, 'ConfigDict.md'), 'utf8');

// Find and replace the special types section
configContent = configContent.replace(
  /\*\*Special W&B Types\*\* \(automatically handled when logging\)\n- W&B Tables: When you log `wandb\.Table\(\)`, appears as table reference\n- W&B Artifacts: When referencing artifacts, appears as artifact link/,
  `**Special W&B Types** (automatically handled when logging)
- W&B Tables: Logged tables appear as references with metadata
- W&B Artifacts: Artifact references appear as links

\`\`\`typescript
// What you see after logging a W&B Table to config:
const config = {
  // ... your normal config ...
  
  // This appears automatically when you log wandb.Table() to config:
  "evaluation_results": {
    "_type": "table-file",
    "ncols": 5,
    "nrows": 100,
    "path": "media/table/eval_results_2_abc123.table.json"
  }
};

// What you see after referencing an artifact in config:
const config = {
  // ... your normal config ...
  
  // This appears when you use an artifact:
  "model_artifact": {
    "_type": "artifactVersion",
    "id": "QXJ0aWZhY3Q6MTIzNDU2",
    "version": "v3",
    "path": "model-weights:v3"
  }
};
\`\`\``
);

fs.writeFileSync(path.join(dataTypesDir, 'ConfigDict.md'), configContent);
console.log('✅ Updated ConfigDict.md with concrete W&B type examples');

// Update SummaryDict
let summaryContent = fs.readFileSync(path.join(dataTypesDir, 'SummaryDict.md'), 'utf8');

// Find and replace the special types section
summaryContent = summaryContent.replace(
  /\*\*Special W&B Types\*\* \(automatically handled when logging\)\n- W&B Histograms: Distribution visualizations\n- W&B Tables: Detailed results tables/,
  `**Special W&B Types** (automatically handled when logging)
- W&B Histograms: Distribution visualizations
- W&B Tables: Detailed results tables

\`\`\`typescript
// What you see after logging W&B special types to summary:
const summary = {
  // ... your normal metrics ...
  
  // This appears when you log wandb.Histogram():
  "weight_distribution": {
    "_type": "histogram",
    "bins": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "values": [10, 25, 45, 30, 15, 5]
  },
  
  // This appears when you log wandb.Table():
  "predictions_table": {
    "_type": "table-file", 
    "ncols": 4,
    "nrows": 1000,
    "path": "media/table/predictions_3_def456.table.json"
  },
  
  // This appears when you reference an artifact:
  "best_model": {
    "_type": "artifactVersion",
    "id": "QXJ0aWZhY3Q6OTg3NjU0",
    "version": "v12",
    "path": "model-checkpoint:v12"
  }
};
\`\`\``
);

fs.writeFileSync(path.join(dataTypesDir, 'SummaryDict.md'), summaryContent);
console.log('✅ Updated SummaryDict.md with concrete W&B type examples');

console.log('\n✨ Done! Added concrete examples showing how W&B special types actually appear.');
