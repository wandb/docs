---
title: wandb job create
menu:
  reference:
    identifier: ja-ref-cli-wandb-job-wandb-job-create
---

**使用法**

`wandb job create [OPTIONS] {git|code|image} PATH`

**概要**

wandb の run なしで、ソースから Job を作成します。

Job には、git、code、または image の3つのタイプがあります。

git: git ソース。パス内または明示的に指定されたエントリポイントが、メインの Python 実行可能ファイルを指します。 code: requirements.txt ファイルを含むコードパス。 image: Docker イメージ。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --project` | Job をリストする Project。 |
| `-e, --entity` | Job が属する Entity。 |
| `-n, --name` | Job の名前。 |
| `-d, --description` | Job の説明。 |
| `-a, --alias` | Job のエイリアス。 |
| `--entry-point` | 実行可能ファイルとエントリポイントファイルを含む、スクリプトのエントリポイント。 code または repo Job に必要です。 --build-context が指定されている場合、エントリポイントコマンドのパスはビルドコンテキストからの相対パスになります。 |
| `-g, --git-hash` | git Job のソースとして使用するコミット参照。 |
| `-r, --runtime` | Job を実行する Python ランタイム。 |
| `-b, --build-context` | Job ソースコードのルートからのビルドコンテキストへのパス。 これが指定されている場合、Dockerfile とエントリポイントのベースパスとして使用されます。 |
| `--base-image` | Job に使用するベースイメージ。image Job と互換性がありません。 |
| `--dockerfile` | Job の Dockerfile へのパス。 --build-context が指定されている場合、Dockerfile のパスはビルドコンテキストからの相対パスになります。 |
