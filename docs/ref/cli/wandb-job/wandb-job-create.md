
# wandb job create

**使用方法**

`wandb job create [OPTIONS] {git|code|image} PATH`

**概要**

wandb run を使わずにソースからジョブを作成します。

ジョブは git、code、image の3種類から選べます。

git: パス内またはメインの python 実行ファイルを明示的に指示するエントリーポイントを持つ git ソース。code: requirements.txt ファイルを含むコードパス。image: Docker イメージ。

**オプション**

| **オプション** | **説明** |
| :--- | :--- |
| -p, --project | ジョブをリストするプロジェクト。 |
| -e, --entity | ジョブが所属するエンティティ。 |
| -n, --name | ジョブの名前。 |
| -d, --description | ジョブの説明。 |
| -a, --alias | ジョブのエイリアス。 |
| --entry-point | スクリプトのエントリーポイント（実行可能ファイルとエントリーポイント ファイルを含む）。code または repo ジョブでは必須。--build-context が指定されている場合、エントリーポイント コマンドのパスはビルドコンテキストに対して相対的になります。 |
| -g, --git-hash | git ジョブのソースとして使用するコミットリファレンス。 |
| -r, --runtime | ジョブを実行する Python ランタイム。 |
| -b, --build-context | ジョブソースコードのルートからビルドコンテキストへのパス。指定されている場合、これが Dockerfile とエントリーポイントのベースパスとして使用されます。 |
| --dockerfile | ジョブ用の Dockerfile へのパス。--build-context が指定されている場合、Dockerfile のパスはビルドコンテキストに対して相対的になります。 |