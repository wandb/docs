---
title: wandb launch-agent
menu:
  reference:
    identifier: ja-ref-cli-wandb-launch-agent
---

**使用方法**

`wandb launch-agent [OPTIONS]`

**概要**

W&B Launch エージェント を実行します。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-q, --queue` | エージェント が監視するキューの名前。複数の -q フラグがサポートされています。 |
| `-e, --entity` | 使用するエンティティ。デフォルトは現在ログインしている ユーザー |
| `-l, --log-file` | 内部 エージェント ログの出力先。stdout には - を使用します。デフォルトでは、すべての エージェント ログは wandb/ サブディレクトリーまたは WANDB_DIR （設定されている場合）の debug.log に出力されます。 |
| `-j, --max-jobs` | この エージェント が並行して実行できる Launch jobs の最大数。デフォルトは 1 です。上限なしの場合は -1 に設定します。 |
| `-c, --config` | 使用する エージェント 設定 yaml へのパス |
| `-v, --verbose` | 詳細な出力を表示 |
