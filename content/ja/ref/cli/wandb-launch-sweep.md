---
title: wandb launch-sweep
---

**使用方法**

`wandb launch-sweep [OPTIONS] [CONFIG]`

**概要**

W&B の Launch sweep を実行します（実験的機能）。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-q, --queue` | sweep をプッシュするキュー名 |
| `-p, --project` | エージェントが監視する Project 名。指定した場合、config ファイルで設定した project の値を上書きします |
| `-e, --entity` | 使用する Entity。デフォルトは現在ログイン中の User です |
| `-r, --resume_id` | 8文字の sweep ID を指定して Launch sweep を再開します。キューの指定が必要です |
| `--prior_run` | この sweep に追加する既存の Run の ID |