---
title: '`resume=''must'' ですが run (<run_id>) が存在しません` というエラーをどう解決すればよいですか？'
menu:
  support:
    identifier: ja-support-kb-articles-how_do_i_fix_the_error_resume_must_but_run_run_id_doesnt_exist
support:
- 再開
- run
toc_hide: true
type: docs
url: /support/:filename
---

`resume='must' but run (<run_id>) doesn't exist` というエラーが発生した場合、再開しようとしている run が、その Project または Entity 内に存在していません。正しいインスタンスにログインし、Project と Entity が正しく設定されているかご確認ください。

```python
wandb.init(entity=<entity>, project=<project>, id=<run-id>, resume='must')
```

認証されているか確認するには、[`wandb login --relogin`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) を実行してください。