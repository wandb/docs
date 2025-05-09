---
title: 申し訳ありませんが、提供されたテキストが空白であるため、翻訳が必要なコンテンツが表示されません。それを解決していただければ、翻訳を提供いたします。
menu:
  reference:
    identifier: ja-ref-python-finish
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L4109-L4130 >}}

run を終了し、残りのデータをアップロードします。

```python
finish(
    exit_code: (int | None) = None,
    quiet: (bool | None) = None
) -> None
```

W&B run の完了をマークし、すべてのデータがサーバーに同期されていることを確認します。run の最終状態は、その終了条件と同期ステータスによって決まります。

#### Run 状態:

- Running: データをログし、またはハートビートを送信しているアクティブな run。
- Crashed: ハートビートの送信が予期せず停止した run。
- Finished: データがすべて同期された状態で正常に完了した run (`exit_code=0`)。
- Failed: エラーがある状態で完了した run (`exit_code!=0`)。

| Args |  |
| :--- | :--- |
|  `exit_code` |  run の終了ステータスを示す整数。成功には 0 を使用し、他の値は run を失敗とマークします。 |
|  `quiet` |  廃止予定。`wandb.Settings(quiet=...)` を使用してログの冗長性を設定します。 |