---
title: wandb ローンンチ-sweep
menu:
  reference:
    identifier: ja-ref-cli-wandb-launch-sweep
---

**使用方法**

`wandb launch-sweep [OPTIONS] [CONFIG]`

**概要**

W&B Launch Sweep を実行します (実験的機能)。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-q, --queue` | sweep をプッシュするキューの名前 |
| `-p, --project` | エージェントが監視するプロジェクトの名前。指定した場合、設定ファイルで渡されたプロジェクトの値を上書きします |
| `-e, --entity` | 使用するエンティティ。デフォルトは現在ログインしているユーザーです |
| `-r, --resume_id` | 8文字のsweep IDを渡してlaunch sweep を再開します。キューが必要です |
| `--prior_run` | このsweep に追加する既存のrun のID |
