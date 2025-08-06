---
title: W&B は `multiprocessing` ライブラリを使用していますか？
menu:
  support:
    identifier: ja-support-kb-articles-multiprocessing_library
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

はい、W&B は `multiprocessing` ライブラリを使用しています。次のようなエラーメッセージが表示される場合、何らかの問題が発生している可能性があります。

```
An attempt has been made to start a new process before the current process 
has finished its bootstrapping phase.
```

この問題を解決するには、`if __name__ == "__main__":` によるエントリーポイントの保護を追加してください。W&B をスクリプトから直接実行する場合、この保護が必要です。