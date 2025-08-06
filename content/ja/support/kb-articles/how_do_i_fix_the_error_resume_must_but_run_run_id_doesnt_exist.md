---
title: エラー `resume='must' but run (<run_id>) doesn't exist` をどうやって解決すればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 再開
- run
---

`resume='must' but run (<run_id>) doesn't exist` というエラーが発生した場合、再開しようとしている run が Project または Entity 内に存在していません。正しいインスタンスにログインしており、Project および Entity が設定されていることを確認してください。

```python
wandb.init(entity=<entity>, project=<project>, id=<run-id>, resume='must')
```

認証されているかを確認するには [`wandb login --relogin`]({{< relref "/ref/cli/wandb-login.md" >}}) を実行してください。