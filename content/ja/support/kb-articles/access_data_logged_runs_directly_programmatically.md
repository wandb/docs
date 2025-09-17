---
title: 自分の runs にログされたデータに直接、プログラムからアクセスするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-access_data_logged_runs_directly_programmatically
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

history オブジェクトは `wandb.log` でログしたメトリクスを追跡します。API を使って history オブジェクトにアクセスします:

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```