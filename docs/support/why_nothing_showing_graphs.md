---
title: "Why is nothing showing up in my graphs?"
tags:
   - experiments
---

If you're seeing "No visualization data logged yet" that means that we haven't gotten the first `wandb.log` call from your script yet. This could be because your run takes a long time to finish a step. If you're logging at the end of each epoch, you could log a few times per epoch to see data stream in more quickly.