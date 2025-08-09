---
title: wandb launch-sweep
menu:
  reference:
    identifier: ja-ref-cli-wandb-launch-sweep
---

**使い方**

`wandb launch-sweep [OPTIONS] [CONFIG]`

**概要**

W&B Launch sweep を実行します（実験的機能）。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-q, --queue` | sweep を投入するキューの名前 |
| `-p, --project` | エージェントが監視するプロジェクト名。指定した場合、設定ファイルで指定した project の値を上書きします |
| `-e, --entity` | 使用する entity。デフォルトは現在ログイン中のユーザー |
| `-r, --resume_id` | 8文字の sweep id を渡して Launch sweep を再開します。キューが必要です |
| `--prior_run` | この sweep に追加する既存 run の ID |