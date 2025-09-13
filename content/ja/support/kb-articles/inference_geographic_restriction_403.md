---
title: なぜ W&B Inference では、私の国または地域がサポートされていないと表示されるのですか？
menu:
  support:
    identifier: ja-support-kb-articles-inference_geographic_restriction_403
support:
- 推論
toc_hide: true
type: docs
url: /support/:filename
---

"Country, region, or territory not supported" と表示される 403 エラーは、サポート対象外の場所から W&B Inference に アクセスしていることを意味します。

## このエラーが発生する理由

W&B Inference にはコンプライアンスや規制上の要件により地理的な制限があります。サービスはサポート対象の地域からのみ アクセスできます。

## 対処方法

1. **利用規約を確認する**
   - 現在のサポート対象地域の一覧は [利用規約](https://docs.coreweave.com/docs/policies/terms-of-service/terms-of-use#geographic-restrictions) を確認

2. **サポート対象の場所から利用する**
   - サポート対象の国や地域にいるときにサービスに アクセスする
   - 組織がサポート対象地域に持つリソースの利用を検討する

3. **アカウント担当に連絡する**
   - エンタープライズのお客様は担当営業と対応策を相談可能
   - 一部の組織には特別な取り決めがある場合がある

## エラーの詳細

このエラーが表示された場合:
```
{
  "error": {
    "code": 403,
    "message": "Country, region, or territory not supported"
  }
}
```

これは API リクエスト時点の IP アドレスの位置情報によって判定されます。