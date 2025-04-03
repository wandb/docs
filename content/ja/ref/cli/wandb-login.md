---
title: wandb login
menu:
  reference:
    identifier: ja-ref-cli-wandb-login
---

**利用方法**

`wandb login [OPTIONS] [KEY]...`

**概要**

Weights & Biases にログインします。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `--cloud` | ローカルではなく クラウド にログインします。 |
| `--host, --base-url` | W&B の特定のインスタンスにログインします。 |
| `--relogin` | すでにログインしている場合に、強制的に再ログインします。 |
| `--anonymously` | 匿名でログインします。 |
| `--verify / --no-verify` | ログイン認証情報を確認します。 |
