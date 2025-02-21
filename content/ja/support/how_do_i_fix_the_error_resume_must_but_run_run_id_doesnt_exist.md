---
title: How do I fix the error `resume='must' but run (<run_id>) doesn't exist`?
menu:
  support:
    identifier: ja-support-how_do_i_fix_the_error_resume_must_but_run_run_id_doesnt_exist
tags:
- resuming
- runs
toc_hide: true
type: docs
---

`resume='must'` のエラーが発生したが、run (<run_id>) が存在しない場合、再開しようとしている run が project または entity 内に存在しません。正しいインスタンスにログインしていること、および project と entity が設定されていることを確認してください。

```python
wandb.init(entity=<entity>, project=<project>, id=<run-id>, resume='must')
```

[`wandb login --relogin`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) を実行して、認証されていることを確認してください。
