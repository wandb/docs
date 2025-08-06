---
title: wandb login
menu:
  reference:
    identifier: ja-ref-cli-wandb-login
---

**使用方法**

`wandb login [OPTIONS] [KEY]...`

**概要**

Weights & Biases にログインします。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--cloud` | ローカルではなくクラウドにログインします |
| `--host, --base-url` | 特定の W&B インスタンスにログインします |
| `--relogin` | すでにログイン済みの場合でも再ログインを強制します。 |
| `--anonymously` | 匿名でログインします |
| `--verify / --no-verify` | ログイン認証情報を検証します |