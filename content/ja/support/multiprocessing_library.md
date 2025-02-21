---
title: Does W&B use the `multiprocessing` library?
menu:
  support:
    identifier: ja-support-multiprocessing_library
tags:
- experiments
toc_hide: true
type: docs
---

はい、W&Bは `multiprocessing` ライブラリを使用しています。次のようなエラーメッセージが表示される場合、問題が発生している可能性があります。

```
現在のプロセスがブートストラップフェーズを終了する前に、新しいプロセスを開始しようとしました。
```

これを解決するには、`if __name__ == "__main__":` を使用してエントリポイントの保護を追加します。この保護は、スクリプトからW&Bを直接実行するときに必要です。