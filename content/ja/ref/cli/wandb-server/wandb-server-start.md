---
title: wandb サーバー start
---

**使い方**

`wandb server start [OPTIONS]`

**概要**

ローカルの W&B サーバーを起動します

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --port` | W&B サーバーをバインドするホストポートを指定 |
| `-e, --env` | wandb/local に渡す環境変数 |
| `--daemon / --no-daemon` | デーモンモードで実行するかしないか |