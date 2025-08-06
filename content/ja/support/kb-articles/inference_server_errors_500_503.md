---
title: W&B Inference でサーバーエラー（500、503）が発生した場合の対処方法
menu:
  support:
    identifier: ja-support-kb-articles-inference_server_errors_500_503
support:
- 推論
toc_hide: true
type: docs
url: /support/:filename
---

サーバーエラーは、一時的に W&B Inference サービスで問題が発生していることを示します。

## エラータイプ

### 500 - Internal Server Error
**メッセージ:** 「サーバーでリクエスト処理中にエラーが発生しました」

これはサーバー側で発生した一時的な内部エラーです。

### 503 - Service Overloaded
**メッセージ:** 「現在エンジンが過負荷のため、しばらくしてから再試行してください」

サービスが高いトラフィックを処理している状態です。

## サーバーエラーの対処法

1. **リトライ前に待機する**
   - 500エラーの場合: 30～60秒待つ
   - 503エラーの場合: 60～120秒待つ

2. **指数バックオフを利用する**
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
               # 500または503エラーの場合、リトライ回数に応じて待機して再試行します
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
   - HTTPクライアントのタイムアウト値を増やす
   - より良いハンドリングのために非同期処理を検討する

## サポートに連絡するタイミング

以下の場合はサポートまでご連絡ください:

- 10分以上エラーが継続する場合
- 特定の時間帯に繰り返し失敗が発生している場合
- エラーメッセージに追加の詳細情報が含まれている場合

ご提供いただきたい情報:

- エラーメッセージおよびエラーコード
- エラーが発生した時間
- ご利用のコードスニペット（APIキーは削除してください）
- W&B の entity 名と project 名