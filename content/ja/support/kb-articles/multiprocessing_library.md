---
title: W&B は `multiprocessing` ライブラリを使用していますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

はい、W&B は `multiprocessing` ライブラリを使用しています。以下のようなエラーメッセージが表示される場合、何らかの問題が発生している可能性があります。

```
An attempt has been made to start a new process before the current process 
has finished its bootstrapping phase.
```

これを解決するには、`if __name__ == "__main__":` でエントリーポイントの保護を追加してください。この保護は、スクリプトから直接 W&B を実行する際に必要です。