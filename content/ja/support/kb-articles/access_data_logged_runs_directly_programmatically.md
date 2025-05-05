---
title: 自分の run にログされたデータに直接アクセスするにはどうしたらよいですか？また、プログラム経由でのアクセス方法は？
menu:
  support:
    identifier: ja-support-kb-articles-access_data_logged_runs_directly_programmatically
support:
  - experiments
toc_hide: true
type: docs
url: /ja/support/:filename
---
履歴オブジェクトは `wandb.log` でログされたメトリクスを追跡します。API を使用して履歴オブジェクトにアクセスします:

```python
api = wandb.Api()
run = api.run("username/project/run_id")
print(run.history())
```