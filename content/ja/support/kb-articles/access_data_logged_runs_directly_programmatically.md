---
title: 自分の Run にログされたデータへ直接かつプログラムからアクセスするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-access_data_logged_runs_directly_programmatically
support:
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

history オブジェクトは、`wandb.log` で記録されたメトリクスを追跡します。API を使って history オブジェクトにアクセスできます。

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```