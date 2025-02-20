---
menu:
  support:
    identifier: ja-support-graphs_nothing_showing
tags:
- experiments
title: Why is nothing showing up in my graphs?
toc_hide: true
type: docs
---

If the message "No visualization data logged yet" appears, the script has not executed the first `wandb.log` call. This situation may occur if the run takes a long time to complete a step. To expedite data logging, log multiple times per epoch instead of only at the end.