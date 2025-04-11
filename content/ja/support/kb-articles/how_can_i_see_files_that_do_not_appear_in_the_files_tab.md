---
title: ファイルタブに表示されないファイルを見るにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_see_files_that_do_not_appear_in_the_files_tab
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

Files タブは最大 10,000 個のファイルを表示します。すべてのファイルをダウンロードするには、[パブリック API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使用します。

```python
import wandb

api = wandb.Api()
run = api.run('<entity>/<project>/<run_id>')
run.file('<file>').download()

for f in run.files():
    if <condition>:
        f.download()
```