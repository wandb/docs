---
title: teardown()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-teardown
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




### <kbd>関数</kbd> `teardown`

```python
teardown(exit_code: 'int | None' = None) → None
```

W&B の完了を待ち、リソースを解放します。

`run.finish()` で明示的に終了されていない run を完了し、すべてのデータがアップロードされるまで待機します。

`wandb.setup()` を使用したセッションの最後にこれを呼び出すことを推奨します。`atexit` フックで自動的に呼び出されますが、Python の `multiprocessing` モジュールを使用している場合など、特定のセットアップでは確実ではありません。