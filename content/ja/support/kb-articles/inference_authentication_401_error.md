---
title: W&B Inference で無効な認証 (401) エラーを解決するには？
menu:
  support:
    identifier: ja-support-kb-articles-inference_authentication_401_error
support:
- 推論
toc_hide: true
type: docs
url: /support/:filename
---

401 Invalid Authentication エラーは、API キーが無効であるか、W&B の project entity/name が正しくないことを意味します。

## API キーを確認する

1. [https://wandb.ai/authorize](https://wandb.ai/authorize) で新しい API キーを取得する
2. コピー時に余分なスペースや文字抜けがないか確認する
3. API キーを安全に保管する

## project の設定を確認する

project が `<your-team>/<your-project>` の形式になっていることを確認してください:

**Python の例:**
```python
client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="<your-api-key>",
    project="<your-team>/<your-project>",  # W&B の team と project に一致している必要があります
)
```

**Bash の例:**
```bash
curl https://api.inference.wandb.ai/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "OpenAI-Project: <your-team>/<your-project>"
```

## よくあるミス

- team 名の代わりに個人の entity を使用している
- team または project 名のスペルミス
- team と project の間のスラッシュがない
- 期限切れまたは削除済みの API キーを使用している

## まだ問題がありますか？

- W&B アカウントに該当の team と project が存在することを確認する
- 指定された team へアクセスできることを確認する
- 現在のものが動作しない場合は、新しい API キーを作成してみてください