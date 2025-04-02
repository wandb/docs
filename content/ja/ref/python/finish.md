---
title: finish
menu:
  reference:
    identifier: ja-ref-python-finish
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L4109-L4130 >}}

run を終了し、残りの data をすべてアップロードします。

```python
finish(
    exit_code: (int | None) = None,
    quiet: (bool | None) = None
) -> None
```

W&B の run の完了をマークし、すべての data が サーバー に同期されるようにします。
run の最終状態は、その終了条件と同期ステータスによって決まります。

#### run の状態:

- Running: data を ログ 記録している、またはハートビートを送信しているアクティブな run 。
- Crashed: 予期せずハートビートの送信を停止した run 。
- Finished: すべての data が同期され、run が正常に完了しました ( `exit_code=0` )。
- Failed: エラーで run が完了しました ( `exit_code!=0` )。

| Arg |  |
| :--- | :--- |
|  `exit_code` |  run の終了ステータスを示す整数。成功の場合は 0 を使用し、他の 値 は run を失敗としてマークします。 |
|  `quiet` |  非推奨。`wandb.Settings(quiet=...)` を使用して、ログ の冗長性を 設定 します。 |
