---
title: "Will wandb slow down my training?"
tags:
   - 
---

W&B should have a negligible effect on your training performance if you use it normally. Normal use of wandb means logging less than once a second and logging less than a few megabytes of data at each step. W&B runs in a separate process and the function calls don't block, so if the network goes down briefly or there are intermittent read write issues on disk it should not affect your performance. It is possible to log a huge amount of data quickly, and if you do that you might create disk I/O issues. If you have any questions, please don't hesitate to contact us.