---
title: "How can I log a metric that doesn't change over time such as a final evaluation accuracy?"
tags:
   - 
---

Using `wandb.log({'final_accuracy': 0.9}` will work fine for this. By default `wandb.log({'final_accuracy'})` will update `wandb.settings['final_accuracy']`, which is the value shown in the runs table.