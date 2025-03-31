---
menu:
  support:
    identifier: ko-support-kb-articles-graphs_nothing_showing
support:
- experiments
title: Why is nothing showing up in my graphs?
toc_hide: true
type: docs
url: /support/:filename
---

If the message "No visualization data logged yet" appears, the script has not executed the first `wandb.log` call. This situation may occur if the run takes a long time to complete a step. To expedite data logging, log multiple times per epoch instead of only at the end.