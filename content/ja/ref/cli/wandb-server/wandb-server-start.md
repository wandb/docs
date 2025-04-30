---
title: wandb サーバー start
menu:
  reference:
    identifier: ja-ref-cli-wandb-server-wandb-server-start
---

**使用方法**

`wandb server start [OPTIONS]`

**概要**

ローカル W&B サーバーを開始します

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --port` | バインドする W&B サーバーのホストポート |
| `-e, --env` | wandb/local に渡す環境変数 |
| `--daemon / --no-daemon` | デーモンモードで run またはデーモンモードで run しない |