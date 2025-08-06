---
title: W&B Inference でレート制限エラー（429）が発生するのはなぜですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 推論
---

レート制限エラー（429）は、同時実行数の上限を超えた場合やクレジットが不足した場合に発生します。

## 429 エラーの種類

### 同時実行制限の到達
**エラー:** "Concurrency limit reached for requests"

**解決方法:**
- 同時に送信するリクエストの数を減らす
- リクエストの間に遅延を挟む
- エクスポネンシャルバックオフを実装する
- 注意: レートリミットは各 W&B Project ごとに適用されます

### クォータ上限超過
**エラー:** "You exceeded your current quota, please check your plan and billing details"

**解決方法:**
- W&B Billing ページでクレジット残高を確認する
- クレジットを追加購入するかプランをアップグレードする
- サポートにリミット増加をリクエストする

### 個人アカウントの制限
**エラー:** "W&B Inference isn't available for personal accounts"

**解決方法:**
- 個人アカウント以外へ切り替える
- Team を作成して W&B Inference にアクセスする
- Personal Entity は 2024年5月に廃止されました

## レートリミットを回避するためのベストプラクティス

1. **エクスポネンシャルバックオフ付きのリトライロジックを実装する:**
   ```python
   import time

   def retry_with_backoff(func, max_retries=3):
       for i in range(max_retries):
           try:
               return func()
           except Exception as e:
               # "429" エラーの場合はリトライし、それ以外は例外を上げる
               if "429" in str(e) and i < max_retries - 1:
                   time.sleep(2 ** i)
               else:
                   raise
   ```

2. **並列リクエストの代わりにバッチプロセッシングを利用する**

3. **W&B Billing ページでご利用状況を監視する**

## デフォルトの利用上限

- **Pro アカウント:** $6,000/月
- **Enterprise アカウント:** $700,000/年

上限の調整をご希望の場合は、アカウント担当者またはサポートまでご連絡ください。