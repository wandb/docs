---
title: wandb job create
menu:
  reference:
    identifier: ja-ref-cli-wandb-job-wandb-job-create
---

**使用方法**

`wandb job create [OPTIONS] {git|code|image} PATH`

**概要**

wandb run なしで、ソースからジョブを作成します。

ジョブには git、code、image の 3 種類があります。

git: git ソース。エントリポイントは、パス内にあるか、メインの Python 実行可能ファイルを指すように明示的に指定します。code: requirements.txt ファイルを含むコード パス。image: Docker イメージ。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --project` | ジョブを一覧表示する対象の プロジェクト。 |
| `-e, --entity` | ジョブが所属する エンティティ |
| `-n, --name` | ジョブの名前 |
| `-d, --description` | ジョブの説明 |
| `-a, --alias` | ジョブの エイリアス |
| `--entry-point` | スクリプトのエントリポイント。実行可能ファイルとエントリポイント ファイルを含みます。code または リポジトリ ジョブで必須です。--build-context が指定された場合、エントリポイント コマンド内のパスはビルド コンテキストからの相対パスになります。 |
| `-g, --git-hash` | git ジョブのソースとして使用するコミット参照 |
| `-r, --runtime` | ジョブを実行するための Python ランタイム |
| `-b, --build-context` | ジョブのソース コードのルートからのビルド コンテキストへのパス。指定した場合、Dockerfile とエントリポイントの基準パスとして使用されます。 |
| `--base-image` | ジョブで使用するベース イメージ。image ジョブとは併用できません。 |
| `--dockerfile` | ジョブの Dockerfile へのパス。--build-context が指定された場合、Dockerfile のパスはビルド コンテキストからの相対パスになります。 |