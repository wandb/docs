---
title: 各アーティファクト バージョンはどれくらいのストレージを使用しますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- アーティファクト
- ストレージ
---

2 つのアーティファクトバージョン間で変更されたファイルのみがストレージコストの対象となります。

{{< img src="/images/artifacts/artifacts-dedupe.PNG" alt="Artifact deduplication" >}}

`animals` という名前の画像アーティファクトがあり、その中に `cat.png` と `dog.png` という 2 つの画像ファイルが含まれているとしましょう。

```
images
|-- cat.png (2MB) # `v0` で追加
|-- dog.png (1MB) # `v0` で追加
```

このアーティファクトはバージョン `v0` となります。

新しい画像 `rat.png` を追加すると、新しいアーティファクトバージョン `v1` が作成され、内容は次のようになります。

```
images
|-- cat.png (2MB) # `v0` で追加
|-- dog.png (1MB) # `v0` で追加
|-- rat.png (3MB) # `v1` で追加
```

バージョン `v1` は合計 6MB を管理していますが、実際に占有するストレージは 3MB のみで、残りの 3MB は `v0` と共有しています。`v1` を削除すると、`rat.png` に割り当てられた 3MB のストレージが解放されます。`v0` を削除すると、`cat.png` と `dog.png` のストレージコストが `v1` に引き継がれ、`v1` のストレージサイズは 6MB に増加します。