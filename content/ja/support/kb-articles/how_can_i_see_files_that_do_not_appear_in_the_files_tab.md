---
title: Files タブに表示されないファイルを確認するにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

Files タブでは最大 10,000 件のファイルが表示されます。すべてのファイルをダウンロードするには、[public API]({{< relref "/ref/python/public-api/api.md" >}}) をご利用ください。

```python
import wandb

api = wandb.Api()
run = api.run('<entity>/<project>/<run_id>')
run.file('<file>').download()

for f in run.files():
    if <condition>:
        f.download()
```