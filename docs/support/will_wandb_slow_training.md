---
title: "Will wandb slow down my training?"
displayed_sidebar: support
tags:
   - experiments
---
W&B has a minimal impact on training performance under normal usage conditions. Normal use includes logging at a rate of less than once per second and limiting data to a few megabytes per step. W&B operates in a separate process with non-blocking function calls, ensuring that brief network outages or intermittent disk read/write issues do not disrupt performance. Excessive logging of large amounts of data may lead to disk I/O issues. For further inquiries, contact support.