#!/usr/bin/env node

/**
 * Fix ConfigDict and SummaryDict to clarify they're flexible dictionaries
 */

const fs = require('fs');
const path = require('path');

const dataTypesDir = path.join(__dirname, '../content/en/ref/query-panel-new/data-types');

console.log('Fixing ConfigDict and SummaryDict documentation...\n');

// ConfigDict - make it clear it's a flexible dictionary
const configDictContent = `---
title: ConfigDict
---

Flexible configuration dictionary for W&B runs. Can contain any JSON-serializable key-value pairs.

\`\`\`typescript
// Example with common configuration fields:
const config: ConfigDict = {
  learning_rate: 0.001,
  batch_size: 32,
  epochs: 10,
  optimizer: "adam",
  model: {
    layers: 12,
    hidden_dim: 768,
    dropout: 0.1
  }
};

// But any valid JSON structure is allowed:
const customConfig: ConfigDict = {
  experiment_name: "test-123",
  use_cuda: true,
  seed: 42,
  custom_params: {
    my_special_setting: "enabled"
  }
};
\`\`\`

**Note:** ConfigDict is a flexible dictionary type that accepts any JSON-serializable keys and values. The examples above show common patterns, but you can use any field names and structures that suit your needs.
`;

// SummaryDict - make it clear it's a flexible dictionary
const summaryDictContent = `---
title: SummaryDict
---

Flexible summary metrics dictionary for W&B runs. Can contain any JSON-serializable key-value pairs.

\`\`\`typescript
// Example with common summary metrics:
const summary: SummaryDict = {
  loss: 0.023,
  accuracy: 0.95,
  val_accuracy: 0.93,
  total_steps: 10000,
  best_epoch: 8
};

// But any metrics or nested structures are allowed:
const customSummary: SummaryDict = {
  train: {
    loss: 0.023,
    accuracy: 0.95
  },
  validation: {
    loss: 0.045,
    accuracy: 0.93
  },
  custom_metric: "completed",
  metadata: {
    checkpoint_saved: true,
    early_stopped: false
  }
};
\`\`\`

**Note:** SummaryDict is a flexible dictionary type that accepts any JSON-serializable keys and values. The examples above show common patterns, but you can log any metrics and structures that suit your needs.
`;

// Write the fixed files
fs.writeFileSync(path.join(dataTypesDir, 'ConfigDict.md'), configDictContent);
console.log('✅ Fixed ConfigDict.md - clarified it\'s a flexible dictionary');

fs.writeFileSync(path.join(dataTypesDir, 'SummaryDict.md'), summaryDictContent);  
console.log('✅ Fixed SummaryDict.md - clarified it\'s a flexible dictionary');

console.log('\n✨ Done! Both dictionary types now clearly explain they\'re flexible and can contain any JSON-serializable data.');
