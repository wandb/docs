---
title: 同じマシン上でアカウントを切り替えるにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-switch_accounts_same_machine
support:
- 環境変数
toc_hide: true
type: docs
url: /support/:filename
---

同じマシンから 2 つの W&B アカウントを管理するには、両方の API キーをファイルに保存します。各リポジトリで次のコードを使えば、API キーを安全に切り替えられ、シークレットキーがバージョン管理にコミットされるのを防げます。

```python
if os.path.exists("~/keys.json"):
    os.environ["WANDB_API_KEY"] = json.loads("~/keys.json")["work_account"]
```