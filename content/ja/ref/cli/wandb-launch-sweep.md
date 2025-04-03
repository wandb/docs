---
title: wandb launch-sweep
menu:
  reference:
    identifier: ja-ref-cli-wandb-launch-sweep
---

**使用方法**

`wandb launch-sweep [OPTIONS] [CONFIG]`

**概要**

W&B Launch sweep（実験的機能）を実行します。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-q, --queue` | sweep を投入するキューの名前 |
| `-p, --project` | エージェント が監視する プロジェクト の名前。渡された場合、設定ファイルを使用して渡された プロジェクト の 値 を上書きします |
| `-e, --entity` | 使用する エンティティ。デフォルトは現在ログインしている ユーザー |
| `-r, --resume_id` | 8 文字の sweep ID を渡して Launch sweep を再開します。キューが必要です |
| `--prior_run` | この sweep に追加する既存の run の ID |
