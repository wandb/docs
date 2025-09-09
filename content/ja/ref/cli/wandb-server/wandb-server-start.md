---
title: wandb server start
menu:
  reference:
    identifier: ja-ref-cli-wandb-server-wandb-server-start
---

**使い方**

`wandb server start [OPTIONS]`

**概要**

ローカルの W&B サーバーを起動する

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --port` | W&B サーバーをバインドするホストのポート |
| `-e, --env` | wandb/local に渡す環境変数 |
| `--daemon / --no-daemon` | デーモンモードで実行する／しない |