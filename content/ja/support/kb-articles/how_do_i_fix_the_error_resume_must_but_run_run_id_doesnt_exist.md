---
title: エラー `resume='must' but run (<run_id>) doesn't exist` をどのように修正すればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-how_do_i_fix_the_error_resume_must_but_run_run_id_doesnt_exist
support:
- 再開
- runs
toc_hide: true
type: docs
url: /support/:filename
---

もし `resume='must' but run (<run_id>) doesn't exist` というエラーが表示される場合、再開しようとしている run が、対象の project または entity 内に存在していません。正しいインスタンスにログインしており、project と entity が設定されていることを確認してください:

```python
wandb.init(entity=<entity>, project=<project>, id=<run-id>, resume='must')
```

認証済みであることを確認するには、[`wandb login --relogin`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) を実行してください。