---
title: How can I see files that do not appear in the Files tab?
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_see_files_that_do_not_appear_in_the_files_tab
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

[ファイル] タブには、最大 10,000 個のファイルが表示されます。すべてのファイルをダウンロードするには、[パブリック API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使用します。

```python
import wandb

api = wandb.Api()
run = api.run('<entity>/<project>/<run_id>')
run.file('<file>').download()

for f in run.files():
    if <condition>:
        f.download()
```
