---
title: W&B は `multiprocessing` ライブラリを使用していますか？
menu:
  support:
    identifier: ja-support-kb-articles-multiprocessing_library
support:
- 実験管理
toc_hide: true
type: docs
url: /ja/support/:filename
---

はい、W&B は `multiprocessing` ライブラリを使用しています。以下のようなエラーメッセージが表示される場合、問題がある可能性があります。

```
An attempt has been made to start a new process before the current process 
has finished its bootstrapping phase.
```

これを解決するには、`if __name__ == "__main__":` でエントリポイント保護を追加してください。これは W&B をスクリプトから直接実行する場合に必要な保護です。