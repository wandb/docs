---
title: W&B Inference でサーバーエラー (500, 503) が発生した場合の対処法
url: /support/:filename
toc_hide: true
type: docs
support:
- 推論
---

サーバーエラーは、一時的な W&B Inference サービスの問題を示します。

## エラーの種類

### 500 - Internal Server Error
**メッセージ:** 「サーバーがリクエストの処理中にエラーが発生しました」

これはサーバー側の一時的な内部エラーです。

### 503 - Service Overloaded
**メッセージ:** 「エンジンが現在混雑しています。しばらくしてから再度お試しください」

サービスが高いトラフィックを経験しています。

## サーバーエラーの対処方法

1. **再試行前に待機する**
   - 500 エラー: 30〜60秒待ってから再試行
   - 503 エラー: 60〜120秒待ってから再試行

2. **指数バックオフを使用する**
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
               # 500 または 503 エラーの場合、再試行する
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
   - 非同期処理を検討し、より柔軟に対応できるようにする

## サポートへの連絡タイミング

以下の場合はサポートへご連絡ください:
- エラーが10分以上続く場合
- 特定の時間帯で失敗パターンが見られる場合
- エラーメッセージに追加情報が含まれている場合

ご提供いただく情報：
- エラーメッセージとコード
- エラーが発生した日時
- ご利用のコードスニペット（APIキーは削除してください）
- W&B の entity 名と project 名