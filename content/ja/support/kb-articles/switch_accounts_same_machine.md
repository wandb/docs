---
title: 同じマシンでアカウントを切り替えるにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-switch_accounts_same_machine
support:
- environment variables
toc_hide: true
type: docs
url: /support/:filename
---

2つのW&Bアカウントを同じマシンから管理するには、両方のAPI キーをファイルに保存します。以下のコードをリポジトリで使用し、キーを安全に切り替えることで、秘密キーがソース管理にチェックインされるのを防ぎます。

```python
# キーを切り替えるコード例
if os.path.exists("~/keys.json"):
    os.environ["WANDB_API_KEY"] = json.loads("~/keys.json")["work_account"]
```