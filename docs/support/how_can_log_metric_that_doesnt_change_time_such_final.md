---
title: "How can I log a metric that doesn't change over time such as a final evaluation accuracy?"
displayed_sidebar: support
tags:
   - None
---
Using `wandb.log({'final_accuracy': 0.9})` updates the final accuracy correctly. By default, `wandb.log({'final_accuracy': <value>})` updates `wandb.settings['final_accuracy']`, which reflects the value in the runs table.