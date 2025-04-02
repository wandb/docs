---
title: How do I switch between accounts on the same machine?
menu:
  support:
    identifier: ja-support-kb-articles-switch_accounts_same_machine
support:
- environment variables
toc_hide: true
type: docs
url: /support/:filename
---

同じマシンから2つの W&B アカウントを管理するには、両方の APIキー をファイルに保存します。次の コード をリポジトリで使用して、秘密 キー がソース管理にチェックインされるのを防ぎ、キー を安全に切り替えます。

```python
if os.path.exists("~/keys.json"):
    os.environ["WANDB_API_KEY"] = json.loads("~/keys.json")["work_account"]
```
