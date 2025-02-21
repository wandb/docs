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

`resume='must' but run (<run_id>) doesn't exist` というエラーが発生した場合、再開しようとしている run がプロジェクトまたはエンティティ内に存在していません。正しいインスタンスにログインしており、プロジェクトとエンティティが設定されていることを確認してください。

```python
wandb.init(entity=<entity>, project=<project>, id=<run-id>, resume='must')
```

[`wandb login --relogin`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) を実行して、認証されていることを確認してください。