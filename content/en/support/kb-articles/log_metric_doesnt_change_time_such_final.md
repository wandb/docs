---
url: /support/:filename
title: "How can I log a metric that doesn't change over time such as a final evaluation accuracy?"
toc_hide: true
type: docs
support:
   - runs
---
Using `run.log({'final_accuracy': 0.9})` updates the final accuracy correctly. By default, `run.log({'final_accuracy': <value>})` updates `run.settings['final_accuracy']`, which reflects the value in the runs table.