---
title: W&B Inference でレート制限エラー (429) が発生するのはなぜですか？
menu:
  support:
    identifier: ja-support-kb-articles-inference_rate_limit_429_error
support:
- 推論
toc_hide: true
type: docs
url: /support/:filename
---

Rate limit エラー (429) は、同時実行制限を超えるか、クレジットを使い切ったときに発生します。

## 429 エラーの種類

### 同時実行上限に到達
**エラー:** "Concurrency limit reached for requests"

**解決策:**
- 並列リクエストの数を減らす
- リクエスト間に遅延を入れる
- 指数バックオフを実装する
- 注意: レート制限は W&B Projects 単位で適用されます

### クォータ超過
**エラー:** "You exceeded your current quota, please check your plan and billing details"

**解決策:**
- W&B Billing ページでクレジット残高を確認する
- クレジットを追加購入するか、プランをアップグレードする
- サポートに上限引き上げを依頼する

### 個人アカウントの制限
**エラー:** "W&B Inference isn't available for personal accounts"

**解決策:**
- 個人以外のアカウントに切り替える
- W&B Inference にアクセスするには Team を作成する
- Personal Entities は 2024 年 5 月に非推奨になりました

## レート制限を回避するためのベストプラクティス

1. **指数バックオフ付きのリトライロジックを実装する:**
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

2. **並列リクエストの代わりにバッチプロセッシングを使用する**

3. **W&B Billing ページで使用量を監視する**

## デフォルトの支出上限

- **Pro アカウント:** 月 $6,000
- **Enterprise アカウント:** 年 $700,000

上限の調整が必要な場合は、営業担当者またはサポートにお問い合わせください。