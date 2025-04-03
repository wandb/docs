---
title: How do I fix the error `resume='must' but run (<run_id>) doesn't exist`?
menu:
  support:
    identifier: ja-support-kb-articles-how_do_i_fix_the_error_resume_must_but_run_run_id_doesnt_exist
support:
- resuming
- runs
toc_hide: true
type: docs
url: /support/:filename
---

`resume='must'` のエラーが発生したが、run（`<run_id>`）が存在しない場合、再開しようとしている run が project または Entity 内に存在しません。正しいインスタンスにログインしており、project と Entity が設定されていることを確認してください。

```python
wandb.init(entity=<entity>, project=<project>, id=<run-id>, resume='must')
```

認証されていることを確認するには、[`wandb login --relogin`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) を実行します。
