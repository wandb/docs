---
title: How do I switch between accounts on the same machine?
menu:
  support:
    identifier: ja-support-switch_accounts_same_machine
tags:
- environment variables
toc_hide: true
type: docs
---

2 つの W&B アカウントを同じマシンで管理するには、両方の APIキー をファイルに保存します。次のコードをリポジトリで使用して、キーを安全に切り替え、秘密キーがソース管理にチェックインされないようにします。

```python
# "~/keys.json" が存在する場合
if os.path.exists("~/keys.json"):
    # 環境変数 WANDB_API_KEY を設定
    os.environ["WANDB_API_KEY"] = json.loads("~/keys.json")["work_account"]
```