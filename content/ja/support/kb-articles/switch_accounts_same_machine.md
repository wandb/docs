---
title: 同じマシンでアカウントを切り替えるにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 環境変数
---

同じマシンで 2つの W&B アカウントを管理するには、両方の APIキー をファイルに保存します。リポジトリ内で次のコードを使ってキーを切り替えることで、秘密のキーがソース管理にチェックインされるのを防げます。

```python
# ファイルが存在するかを確認
if os.path.exists("~/keys.json"):
    # APIキー の環境変数を設定
    os.environ["WANDB_API_KEY"] = json.loads("~/keys.json")["work_account"]
```
