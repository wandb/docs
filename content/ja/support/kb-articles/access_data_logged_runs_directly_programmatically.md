---
title: 自分の Run にログされたデータへ直接プログラムからアクセスするにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

history オブジェクトは、`wandb.log` で記録されたメトリクスを追跡します。API を使って history オブジェクトにアクセスできます。

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```