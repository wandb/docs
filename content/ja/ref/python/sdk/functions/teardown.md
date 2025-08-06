---
title: teardown()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-teardown
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




### <kbd>function</kbd> `teardown`

```python
teardown(exit_code: 'int | None' = None) → None
```

W&B の処理が完了するのを待ち、リソースを解放します。

`run.finish()` で明示的に終了されていない Run をすべて完了させ、すべてのデータがアップロードされるまで待機します。

`wandb.setup()` を使用したセッションの最後にこれを呼び出すことを推奨します。これは `atexit` フック内で自動的に呼び出されますが、Python の `multiprocessing` モジュールなど一部の環境では確実ではありません。