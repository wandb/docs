---
title: Files タブに表示されないファイルを表示するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_see_files_that_do_not_appear_in_the_files_tab
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

Files タブには、最大 10,000 個のファイルが表示されます。すべてのファイルをダウンロードするには、[公開 API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使用してください:

```python
import wandb

api = wandb.Api()
run = api.run('<entity>/<project>/<run_id>')
run.file('<file>').download()

for f in run.files():
    if <condition>:
        f.download()
```