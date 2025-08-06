---
title: W&B Inference で Invalid Authentication (401) エラーが発生した場合の対処方法
url: /support/:filename
toc_hide: true
type: docs
support:
- 推論
---

401 Invalid Authentication エラーは、APIキーが無効であるか、W&B の project entity/name が間違っている場合に発生します。

## APIキーを確認する

1. [https://wandb.ai/authorize](https://wandb.ai/authorize) で新しい APIキーを取得してください
2. コピー時に余分なスペースや文字抜けがないか確認してください
3. APIキーは安全に保管してください

## project の設定を確認する

project が `<your-team>/<your-project>` の形式になっているか確認してください:

**Python の例:**
```python
client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="<your-api-key>",
    project="<your-team>/<your-project>",  # W&B の team と project 名が一致している必要があります
)
```

**Bash の例:**
```bash
curl https://api.inference.wandb.ai/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "OpenAI-Project: <your-team>/<your-project>"
```

## よくあるミス

- team 名の代わりに personal entity を使用している
- team や project 名のスペルミス
- team と project の間にスラッシュが抜けている
- 期限切れまたは削除された APIキーを使っている

## それでも解決しない場合

- W&B アカウントにて team と project が存在するか確認してください
- 指定された team へのアクセス権があるか確認してください
- 今の APIキーが使えない場合は、新しい APIキーを作成してみてください