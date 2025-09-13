---
title: W&B は `multiprocessing` ライブラリを使用しますか？
menu:
  support:
    identifier: ja-support-kb-articles-multiprocessing_library
support:
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

はい、W&B は `multiprocessing` ライブラリを使用しています。次のようなエラーメッセージは、問題が発生している可能性を示します:

```
An attempt has been made to start a new process before the current process 
has finished its bootstrapping phase.
```

これを解決するには、`if __name__ == "__main__":` によるエントリーポイントガードを追加してください。これは、スクリプトから W&B を直接実行する場合に必要です。