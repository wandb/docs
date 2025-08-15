---
title: wandb job create
menu:
  reference:
    identifier: ja-ref-cli-wandb-job-wandb-job-create
---

**使い方**

`wandb job create [OPTIONS] {git|code|image} PATH`

**概要**

wandb run を行わずに、ソースから job を作成します。

Jobs は git、code、image のいずれかのタイプになります。

git: エントリポイントがパス内にあるか、または明示的に指定されている Git ソース。メインの Python 実行ファイルを指します。  
code: requirements.txt ファイルを含むコードのパス。  
image: docker イメージ。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --project` | Jobs を一覧表示したい Project を指定します。 |
| `-e, --entity` | Jobs が属している Entity を指定します。|
| `-n, --name` | Job の名前を指定します。|
| `-d, --description` | Job の説明を入力します。|
| `-a, --alias` | Job のエイリアスを指定します。|
| `--entry-point` | スクリプトのエントリポイント（実行ファイルとエントリポイントファイルを含みます）。code または repo ジョブには必須です。--build-context が指定された場合、エントリポイントコマンド内のパスはビルドコンテキストからの相対パスになります。|
| `-g, --git-hash` | git ジョブのソースとして使用するコミットリファレンスを指定します。|
| `-r, --runtime` | Job を実行する Python ランタイムを指定します。|
| `-b, --build-context` | Job ソースコードのルートからビルドコンテキストへのパスを指定します。指定した場合、このパスが Dockerfile やエントリポイントの基準パスとして使われます。|
| `--base-image` | Job で使用するベースイメージを指定します。image ジョブとは併用できません。|
| `--dockerfile` | Job 用の Dockerfile のパスを指定します。--build-context が指定されている場合、Dockerfile のパスはビルドコンテキストからの相対パスになります。|