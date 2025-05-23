---
title: wandb job create
menu:
  reference:
    identifier: ja-ref-cli-wandb-job-wandb-job-create
---

**使用方法**

`wandb job create [OPTIONS] {git|code|image} PATH`

**概要**

wandb run を使用せずにソースからジョブを作成します。

ジョブには、git、code、または image の3種類があります。

git: パス内またはメインの Python 実行可能ファイルを指すエントリポイントを明示的に指定した git ソースです。code: requirements.txt ファイルを含むコードパスです。image: Docker イメージです。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| `-p, --project` | ジョブをリストするプロジェクトを指定します。 |
| `-e, --entity` | ジョブが属するエンティティ |
| `-n, --name` | ジョブの名前 |
| `-d, --description` | ジョブの説明 |
| `-a, --alias` | ジョブのエイリアス |
| `--entry-point` | スクリプトのエントリポイントで、実行可能ファイルとエントリポイントファイルを含みます。code または repo ジョブでは必須です。--build-context が提供されている場合、エントリポイントコマンド内のパスはビルドコンテキストに対して相対的なものになります。 |
| `-g, --git-hash` | git ジョブのソースとして使用するコミット参照 |
| `-r, --runtime` | ジョブを実行する Python ランタイム |
| `-b, --build-context` | ジョブのソースコードのルートからビルドコンテキストへのパスです。提供されている場合、これが Dockerfile とエントリポイントのベースパスとして使用されます。 |
| `--base-image` | ジョブに使用するベースイメージ。image ジョブとは互換性がありません。 |
| `--dockerfile` | ジョブの Dockerfile へのパス。--build-context が提供されている場合、Dockerfile のパスはビルドコンテキストに対して相対的になります。 |