---
title: '`resume=''must'' but run (<run_id>) doesn''t exist` エラーをどのように修正しますか？'
menu:
  support:
    identifier: ja-support-kb-articles-how_do_i_fix_the_error_resume_must_but_run_run_id_doesnt_exist
support:
- 再開
- 'run '
toc_hide: true
type: docs
url: /ja/support/:filename
---

`resume='must' but run (<run_id>) doesn't exist` というエラーが発生した場合、再開しようとしている run が Project または Entity 内に存在しません。正しいインスタンスにログインし、Project と Entity が設定されていることを確認してください。

```python
wandb.init(entity=<entity>, project=<project>, id=<run-id>, resume='must')
```

[`wandb login --relogin`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) を実行して、認証されていることを確認してください。