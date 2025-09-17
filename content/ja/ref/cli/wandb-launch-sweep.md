---
title: wandb launch-sweep
menu:
  reference:
    identifier: ja-ref-cli-wandb-launch-sweep
---

**使い方**

`wandb launch-sweep [OPTIONS] [CONFIG]`

**概要**

W&B Launch sweep を実行します（実験的）。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-q, --queue` | sweep を投入するキューの名前 |
| `-p, --project` | エージェントが監視する project の名前。指定した場合は、設定ファイルで指定された project の値を上書きします |
| `-e, --entity` | 使用する entity。デフォルトは現在ログインしている user です |
| `-r, --resume_id` | 8 文字の sweep ID を渡して Launch sweep を再開します。キューが必要です |
| `--prior_run` | この sweep に追加する既存の run の ID |