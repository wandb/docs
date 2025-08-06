---
title: なぜ W&B Inference で「お住まいの国または地域はサポートされていません」と表示されるのですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 推論
---

「Country, region, or territory not supported（国・地域・領域がサポート対象外です）」という 403 エラーは、W&B Inference にサポート対象外の場所から アクセス している場合に表示されます。

## なぜこのエラーが発生するのか

W&B Inference では、コンプライアンスや規制要件により地理的な制限が設けられています。このサービスは、サポートされている地域からのみ アクセス 可能です。

## できること

1. **利用規約を確認する**
   - 現在サポートされている場所の一覧について、[利用規約](https://docs.coreweave.com/docs/policies/terms-of-service/terms-of-use#geographic-restrictions) をご確認ください。

2. **サポートされている場所から利用する**
   - サポートされている国や地域にいるときにサービスへ アクセス してください。
   - サポート対象の場所で、所属する組織のリソース利用をご検討ください。

3. **アカウント担当チームへ連絡する**
   - Enterprise のお客様は、担当のアカウントエグゼクティブとご相談いただけます。
   - 一部の組織では特別な取り決めがある場合があります。

## エラーの詳細

このようなエラーが表示される場合:

```
{
  "error": {
    "code": 403,
    "message": "Country, region, or territory not supported"
  }
}
```

これは、APIリクエスト時のあなたの IP アドレスの位置によって判断されています。