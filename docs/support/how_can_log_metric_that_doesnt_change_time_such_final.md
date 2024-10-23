---
title: "How can I log a metric that doesn't change over time such as a final evaluation accuracy?"
tags:
   - runs table
---
Using `wandb.log({'final_accuracy': 0.9})` updates the final accuracy correctly. By default, `wandb.log({'final_accuracy': <value>})` updates `wandb.settings['final_accuracy']`, which reflects the value in the runs table.