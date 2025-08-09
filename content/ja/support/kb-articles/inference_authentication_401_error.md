---
title: W&B Inference で「無効な認証（401）」エラーが発生した場合の対処方法
menu:
  support:
    identifier: ja-support-kb-articles-inference_authentication_401_error
support:
- 推論
toc_hide: true
type: docs
url: /support/:filename
---

401 Invalid Authentication エラーは、APIキーが無効であるか、W&B の project entity/name が間違っている場合に発生します。

## APIキーを確認する

1. 新しい APIキーを [https://wandb.ai/authorize](https://wandb.ai/authorize) で取得してください
2. コピー時に余分なスペースや抜けている文字がないか確認してください
3. APIキーを安全に保管してください

## Project 設定を確認する

project のフォーマットが正しいことを確認してください（`<your-team>/<your-project>` の形式）:

**Python 例:**
```python
client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="<your-api-key>",
    project="<your-team>/<your-project>",  # W&B の team と project 名が一致している必要があります
)
```

**Bash 例:**
```bash
curl https://api.inference.wandb.ai/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "OpenAI-Project: <your-team>/<your-project>"
```

## よくある間違い

- 個人 entity を team 名の代わりに使用している
- team 名や project 名をスペルミスしている
- team と project の間のスラッシュが抜けている
- 期限切れまたは削除された APIキーを使用している

## それでも問題が解決しない場合

- 該当の team と project が W&B アカウントに存在しているか確認してください
- 指定した team へのアクセス権があるか確認してください
- 現在の APIキーでうまくいかない場合、新しい APIキーを作成して試してください