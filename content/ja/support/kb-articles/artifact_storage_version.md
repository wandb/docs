---
title: 各 Artifacts バージョンはどれくらいのストレージを使用しますか?
menu:
  support:
    identifier: ja-support-kb-articles-artifact_storage_version
support:
- アーティファクト
- 記憶装置
toc_hide: true
type: docs
url: /support/:filename
---

2 つの Artifacts のバージョン間で変更されたファイルのみがストレージコストの対象となります。

{{< img src="/images/artifacts/artifacts-dedupe.PNG" alt="Artifacts 'dataset' の v1 では、異なる画像は 5 枚中 2 枚のみで、そのため占有率は 40% にとどまります。" >}}

2 つの画像ファイル `cat.png` と `dog.png` を含む画像 Artifact `animals` を考えてみましょう:

images
|-- cat.png (2MB) # `v0` で追加
|-- dog.png (1MB) # `v0` で追加

この Artifact にはバージョン `v0` が割り当てられます。

新しい画像 `rat.png` を追加すると、新しい Artifact のバージョン `v1` が次の内容で作成されます:

images
|-- cat.png (2MB) # `v0` で追加
|-- dog.png (1MB) # `v0` で追加
|-- rat.png (3MB) # `v1` で追加

バージョン `v1` は合計 6MB をトラックしますが、`v0` と 3MB を共有しているため、占有するスペースは 3MB のみです。`v1` を削除すると、`rat.png` に関連する 3MB のストレージが回収されます。`v0` を削除すると、`cat.png` と `dog.png` のストレージコストは `v1` に移され、ストレージサイズは 6MB に増加します。