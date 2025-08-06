---
title: W&B Inference でレートリミットエラー（429）が発生するのはなぜですか？
menu:
  support:
    identifier: ja-support-kb-articles-inference_rate_limit_429_error
support:
- 推論
toc_hide: true
type: docs
url: /support/:filename
---

レート制限エラー（429）は、同時実行制限を超えた場合やクレジットが不足した場合に発生します。

## 429 エラーの種類

### 同時実行制限に到達
**エラー:** "Concurrency limit reached for requests"

**解決方法:**
- 並列リクエスト数を減らす
- リクエスト間に遅延を加える
- 指数的なバックオフを実装する
- 注意: レート制限は各 W&B Project ごとに適用されます

### クォータ超過
**エラー:** "You exceeded your current quota, please check your plan and billing details"

**解決方法:**
- W&B の課金ページでクレジット残高を確認する
- クレジットを追加購入するか、プランをアップグレードする
- サポートから制限の引き上げをリクエストする

### 個人アカウントの制限
**エラー:** "W&B Inference isn't available for personal accounts"

**解決方法:**
- パーソナルアカウント以外に切り替える
- W&B Inference を利用するために Team を作成する
- パーソナル Entities は 2024年5月に廃止されました

## レート制限を回避するベストプラクティス

1. **指数的バックオフを使ったリトライロジックを実装する:**
   ```python
   import time
   
   def retry_with_backoff(func, max_retries=3):
       for i in range(max_retries):
           try:
               return func()
           except Exception as e:
               if "429" in str(e) and i < max_retries - 1:
                   time.sleep(2 ** i)
               else:
                   raise
   ```

2. **並列リクエストではなくバッチプロセッシングを使用する**

3. **W&B の課金ページで利用状況をモニタリングする**

## デフォルトの利用上限

- **Pro アカウント:** $6,000/月
- **Enterprise アカウント:** $700,000/年

上限の調整については、アカウント担当者またはサポートまでご連絡ください。