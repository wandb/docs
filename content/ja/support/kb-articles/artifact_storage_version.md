---
title: 各 artifact バージョンはどれくらいのストレージを使用しますか？
menu:
  support:
    identifier: ja-support-kb-articles-artifact_storage_version
support:
- アーティファクト
- ストレージ
toc_hide: true
type: docs
url: /support/:filename
---

2 つの artifact バージョン間で変更されたファイルのみがストレージコストの対象となります。

{{< img src="/images/artifacts/artifacts-dedupe.PNG" alt="Artifact の重複排除" >}}

`animals` という名前の画像 artifact があり、2 つの画像ファイル `cat.png` と `dog.png` を含んでいるとします。

```
images
|-- cat.png (2MB) # `v0` で追加
|-- dog.png (1MB) # `v0` で追加
```

この artifact はバージョン `v0` となります。

新たに `rat.png` という画像を追加すると、新しい artifact バージョンである `v1` が以下の内容で作成されます。

```
images
|-- cat.png (2MB) # `v0` で追加
|-- dog.png (1MB) # `v0` で追加
|-- rat.png (3MB) # `v1` で追加
```

バージョン `v1` は合計 6MB を追跡していますが、`v0` と残り 3MB を共有しているため、実際の使用容量は 3MB のみです。`v1` を削除すると、`rat.png` に関連する 3MB のストレージが解放されます。`v0` を削除すると、`cat.png` と `dog.png` のストレージコストが `v1` に移行し、`v1` のストレージサイズは 6MB となります。