---
title: 各 Artifact のバージョンはどれくらいのストレージを使用しますか？
menu:
  support:
    identifier: ja-support-kb-articles-artifact_storage_version
support:
- artifacts
- ストレージ
toc_hide: true
type: docs
url: /support/:filename
---

2 つの artifact バージョン間で変更されたファイルのみがストレージ費用の対象になります。
{{< img src="/images/artifacts/artifacts-dedupe.PNG" alt="Artifact の重複排除" >}}
`cat.png` と `dog.png` の 2 つの画像ファイルを含む、`animals` という名前の画像 artifact を考えます:
```
images
|-- cat.png (2MB) # `v0` で追加
|-- dog.png (1MB) # `v0` で追加
```
この artifact には バージョン `v0` が割り当てられます。
新しい画像 `rat.png` を追加すると、新しい artifact バージョン `v1` が次の内容で作成されます:
```
images
|-- cat.png (2MB) # `v0` で追加
|-- dog.png (1MB) # `v0` で追加
|-- rat.png (3MB) # `v1` で追加
```
バージョン `v1` は合計 6MB を追跡しますが、残りの 3MB を `v0` と共有するため、実際に占有する容量は 3MB のみです。`v1` を削除すると、`rat.png` に関連する 3MB のストレージが解放されます。`v0` を削除すると、`cat.png` と `dog.png` のストレージコストは `v1` に引き継がれ、`v1` のストレージサイズは 6MB に増加します。