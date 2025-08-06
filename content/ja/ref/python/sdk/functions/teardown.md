---
title: teardown()
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




### <kbd>function</kbd> `teardown`

```python
teardown(exit_code: 'int | None' = None) → None
```

W&B の処理が完了し、リソースを解放するまで待ちます。

`run.finish()` を明示的に呼び出していない Run もすべて完了させ、すべてのデータのアップロードが終わるまで待機します。

`wandb.setup()` を使ったセッションの終了時にこの関数を呼ぶことを推奨します。`atexit` フック内でも自動的に呼ばれますが、Python の `multiprocessing` モジュールを利用している場合など、一部の環境では確実に動作しないことがあります。