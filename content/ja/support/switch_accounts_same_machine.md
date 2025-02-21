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

同じマシンから2つの W&B アカウントを管理するには、両方の APIキー をファイルに保存します。次のコードをリポジトリで使用して、秘密キーがソース管理にチェックインされるのを防ぎ、キーを安全に切り替えます。

```python
if os.path.exists("~/keys.json"):
    os.environ["WANDB_API_KEY"] = json.loads("~/keys.json")["work_account"]
```
