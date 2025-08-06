---
title: "finish()  \n完了（finish）"
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-finish
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py >}}




### <kbd>function</kbd> `finish`

```python
finish(exit_code: 'int | None' = None, quiet: 'bool | None' = None) → None
```

run を終了し、残りのデータをアップロードします。

W&B run の完了を示し、すべてのデータがサーバーに同期されていることを保証します。run の最終的な状態は、終了条件と同期ステータスによって決まります。

run の状態:
- Running: データのログやハートビートの送信がアクティブな run。
- Crashed: ハートビートの送信が予期せず停止した run。
- Finished: 正常に完了し（`exit_code=0`）、すべてのデータが同期された run。
- Failed: エラーが発生して完了した run（`exit_code!=0`）。

**引数:**

 - `exit_code`:  run の終了ステータスを示す整数値。成功時は 0、その他の値は run が失敗したことを示します。
 - `quiet`:  非推奨。ログ出力レベルは `wandb.Settings(quiet=...)` で設定してください。
```