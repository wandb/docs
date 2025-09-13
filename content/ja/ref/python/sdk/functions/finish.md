---
title: finish()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-finish
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py >}}




### <kbd>関数</kbd> `finish`

```python
finish(exit_code: 'int | None' = None, quiet: 'bool | None' = None) → None
```

run を終了し、残っているデータをアップロードします。 

W&B run の完了を示し、すべてのデータがサーバーに同期されていることを保証します。run の最終状態は、終了条件と同期ステータスによって決まります。 

run の状態: 
- Running: データをログしたり、ハートビートを送信しているアクティブな run。 
- Crashed: 予期せずハートビート送信が止まった run。 
- Finished: すべてのデータが同期されたうえで正常終了した run（`exit_code=0`）。 
- Failed: エラーで終了した run（`exit_code!=0`）。 



**引数:**
 
 - `exit_code`:  run の終了ステータスを示す整数。成功は 0、その他の値は run を失敗として扱います。 
 - `quiet`:  非推奨。ログの詳細度は `wandb.Settings(quiet=...)` で設定してください。