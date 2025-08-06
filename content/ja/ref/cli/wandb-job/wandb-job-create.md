---
title: wandb job create
---

**使い方**

`wandb job create [OPTIONS] {git|code|image} PATH`

**概要**

wandb run を作成せずに、ソースからジョブを作成します。

ジョブの種類は、git、code、image の 3 種類があります。

- git: エントリーポイントがパスに含まれているか、明示的に指定された git ソース。これはメインの Python 実行ファイルを指します。
- code: requirements.txt ファイルを含むコードパス。
- image: Docker イメージ。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --project` | ジョブを一覧表示したい Project を指定します。 |
| `-e, --entity` | ジョブが属する Entity を指定します。 |
| `-n, --name` | ジョブの名前を指定します。 |
| `-d, --description` | ジョブの説明を記述します。 |
| `-a, --alias` | ジョブのエイリアスを指定します。 |
| `--entry-point` | 実行ファイルおよびエントリーポイントファイルを含むスクリプトのエントリーポイント。code または repo ジョブの場合は必須です。--build-context が指定された場合、エントリーポイント コマンド内のパスはビルドコンテキストからの相対パスになります。 |
| `-g, --git-hash` | git ジョブのソースとして使用するコミット参照を指定します。 |
| `-r, --runtime` | ジョブの実行に使用する Python ランタイムを指定します。 |
| `-b, --build-context` | ジョブのソースコードのルートからビルドコンテキストへのパス。指定した場合、このパスが Dockerfile やエントリーポイントの基準パスとなります。 |
| `--base-image` | ジョブで使用するベースイメージを指定します。image ジョブとの併用はできません。 |
| `--dockerfile` | ジョブ用の Dockerfile のパス。--build-context が指定されている場合、Dockerfile のパスはビルドコンテキストからの相対パスとなります。 |