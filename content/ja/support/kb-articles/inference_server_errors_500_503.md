---
title: W&B Inference での サーバー エラー (500、503) はどのように対処すればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-inference_server_errors_500_503
support:
- 推論
toc_hide: true
type: docs
url: /support/:filename
---

サーバー エラーは W&B Inference サービスに一時的な問題が発生していることを示します。

## エラーの種類

### 500 - Internal Server Error
**メッセージ:** "The server had an error while processing your request"

これはサーバー 側の一時的な内部エラーです。

### 503 - Service Overloaded
**メッセージ:** "The engine is currently overloaded, please try again later"

サービスでトラフィックが増加しています。

## サーバー エラーの対処方法

1. **再試行の前に待機する**
   - 500 エラー: 30〜60 秒待つ
   - 503 エラー: 60〜120 秒待つ

2. **指数バックオフを使う**
   ```python
   import time
   import openai
   
   def call_with_retry(client, messages, model, max_retries=5):
       for attempt in range(max_retries):
           try:
               return client.chat.completions.create(
                   model=model,
                   messages=messages
               )
           except Exception as e:
               if "500" in str(e) or "503" in str(e):
                   if attempt < max_retries - 1:
                       wait_time = min(60, (2 ** attempt))
                       time.sleep(wait_time)
                   else:
                       raise
               else:
                   raise
   ```

3. **適切なタイムアウトを設定する**
   - HTTP クライアントのタイムアウト値を増やす
   - より良いハンドリングのために非同期処理を検討する

## サポートに連絡するタイミング

次の場合はサポートに連絡してください:
- エラーが 10 分以上継続する
- 特定の時間帯に失敗のパターンが見られる
- エラー メッセージに追加の詳細が含まれている

次を提供してください:
- エラー メッセージとコード
- エラーが発生した時刻
- あなたのコードスニペット（API キーを削除）
- W&B の entity と project 名