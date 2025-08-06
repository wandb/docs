---
title: W&B Inference で「お住まいの国や地域はサポートされていません」と表示されるのはなぜですか？
menu:
  support:
    identifier: ja-support-kb-articles-inference_geographic_restriction_403
support:
- 推論
toc_hide: true
type: docs
url: /support/:filename
---

「Country, region, or territory not supported」というメッセージ付きの 403 エラーは、サポートされていない地域から W&B Inference にアクセスしようとしていることを示しています。

## なぜこの問題が発生するか

W&B Inference は、コンプライアンスおよび規制要件により地理的な制限があります。サポートされている地域からのみサービスに アクセスできます。

## どのように対応できるか

1. **利用規約を確認する**
   - 現在サポートされている地域の一覧は [Terms of Service](https://docs.coreweave.com/docs/policies/terms-of-service/terms-of-use#geographic-restrictions) をご確認ください

2. **サポートされている地域から利用する**
   - サポート対象の国または地域にいるときのみサービスに アクセスしてください
   - 組織のリソースがサポート地域にある場合はそちらを検討してください

3. **アカウントチームに相談する**
   - エンタープライズのお客様はアカウント担当者にオプションについてご相談いただけます
   - 一部の組織には特別な取り決めがある場合があります

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

これは API リクエスト時の IP アドレスの場所によって判定されています。