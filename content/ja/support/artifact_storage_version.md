---
title: How much storage does each artifact version use?
menu:
  support:
    identifier: ja-support-artifact_storage_version
tags:
- artifacts
- storage
toc_hide: true
type: docs
---

2 つのアーティファクト バージョン間で変更されるファイルのみがストレージ コストの対象となります。

{{< img src="/images/artifacts/artifacts-dedupe.PNG" alt="アーティファクト 'dataset' の v1 では、異なる画像が 5 枚中 2 枚しかないため、スペースの 40% しか占有しません。" >}}

`animals` という名前の画像アーティファクトを考えてみましょう。このアーティファクトには、2 つの画像ファイル `cat.png` と `dog.png` が含まれています。

```
images
|-- cat.png (2MB) # `v0` で追加
|-- dog.png (1MB) # `v0` で追加
```

このアーティファクトはバージョン `v0` を受け取ります。

新しい画像 `rat.png` を追加すると、新しいアーティファクト バージョン `v1` が次の内容で作成されます。

```
images
|-- cat.png (2MB) # `v0` で追加
|-- dog.png (1MB) # `v0` で追加
|-- rat.png (3MB) # `v1` で追加
```

バージョン `v1` は合計 6MB を追跡しますが、`v0` と 3MB を共有しているため、3MB のスペースしか占有しません。`v1` を削除すると、`rat.png` に関連付けられた 3MB のストレージが回収されます。`v0` を削除すると、`cat.png` と `dog.png` のストレージ コストは `v1` に移行され、そのストレージ サイズは 6MB に増加します。