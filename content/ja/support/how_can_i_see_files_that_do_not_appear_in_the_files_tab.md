---
menu:
  support:
    identifier: ja-support-how_can_i_see_files_that_do_not_appear_in_the_files_tab
tags:
- experiments
title: How can I see files that do not appear in the Files tab?
toc_hide: true
type: docs
---

The Files tab shows a maximum of 10,000 files. To download all files, use the [public API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}):

```python
import wandb

api = wandb.Api()
run = api.run('<entity>/<project>/<run_id>')
run.file('<file>').download()

for f in run.files():
    if <condition>:
        f.download()
```