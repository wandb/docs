---
title: 'finish()

  '
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py >}}




### <kbd>function</kbd> `finish`

```python
finish(exit_code: 'int | None' = None, quiet: 'bool | None' = None) → None
```

run を終了し、残りのデータをすべてアップロードします。

W&B run の完了をマークし、すべてのデータがサーバーに同期されることを保証します。run の最終的な状態は、exit 条件と同期ステータスによって決まります。

Run 状態:
- Running: データのログやハートビート送信を行っているアクティブな run。
- Crashed: ハートビートの送信が予期せず停止した run。
- Finished: すべてのデータが同期された上で正常に完了した run（`exit_code=0`）。
- Failed: エラーが発生して終了した run（`exit_code!=0`）。



**引数:**
 
 - `exit_code`: run の終了ステータスを示す整数。0 は成功、それ以外の値は run が失敗したことを示します。
 - `quiet`: 廃止されました。ログの出力レベルは `wandb.Settings(quiet=...)` で設定してください。
```