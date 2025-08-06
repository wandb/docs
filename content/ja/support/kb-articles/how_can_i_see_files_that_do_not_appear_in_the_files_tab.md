---
title: Files タブに表示されないファイルを確認するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_see_files_that_do_not_appear_in_the_files_tab
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

Filesタブでは最大10,000個のファイルが表示されます。すべてのファイルをダウンロードするには、[public API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) をご利用ください。

```python
import wandb

api = wandb.Api()
run = api.run('<entity>/<project>/<run_id>')
run.file('<file>').download()

for f in run.files():
    if <condition>:
        f.download()
```