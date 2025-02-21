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

2 つの artifact のバージョン間で変更されたファイルのみがストレージコストを発生させます。

{{< img src="/images/artifacts/artifacts-dedupe.PNG" alt="artifact 'dataset' の v1 は、5 つの画像のうち 2 つだけが異なっているので、スペースの 40% のみを占有しています。" >}}

`animals` という名前の画像 artifact を考えてみましょう。これには、2 つの画像ファイル、`cat.png` と `dog.png` が含まれています。

```
images
|-- cat.png (2MB) # `v0` で追加
|-- dog.png (1MB) # `v0` で追加
```

この artifact はバージョン `v0` を受け取ります。

新しい画像 `rat.png` を追加すると、次の内容で新しい artifact バージョン `v1` が作成されます。

```
images
|-- cat.png (2MB) # `v0` で追加
|-- dog.png (1MB) # `v0` で追加
|-- rat.png (3MB) # `v1` で追加
```

バージョン `v1` は合計 6MB を追跡しますが、残りの 3MB を `v0` と共有するため、3MB のスペースしか占有しません。`v1` を削除すると、`rat.png` に関連付けられた 3MB のストレージが再利用されます。`v0` を削除すると、`cat.png` と `dog.png` のストレージコストが `v1` に転送され、ストレージサイズが 6MB に増加します。
