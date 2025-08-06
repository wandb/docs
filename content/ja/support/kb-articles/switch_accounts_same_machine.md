---
title: 同じマシンでアカウントを切り替えるにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-switch_accounts_same_machine
support:
- 環境変数
toc_hide: true
type: docs
url: /support/:filename
---

同じマシンで２つの W&B アカウントを管理するには、両方のAPIキーをファイルに保存してください。リポジトリで以下のコードを使うと、キーを安全に切り替えられ、秘密のキーがソース管理に誤って含まれることを防げます。

```python
# もし"~/keys.json"が存在すれば
if os.path.exists("~/keys.json"):
    # "work_account"用のAPIキーを環境変数に設定
    os.environ["WANDB_API_KEY"] = json.loads("~/keys.json")["work_account"]
```
