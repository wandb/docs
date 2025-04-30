---
title: gcc が環境にない場合、wandb Python ライブラリをインストールする方法は？
menu:
  support:
    identifier: ja-support-kb-articles-install_wandb_python_library_environments_without_gcc
support:
- "{\n  \"データを保存するのに有用な方法\": {\n    \"Artifacts\": \"Artifacts は、W&B でのファイルやディレクトリの保存とバージョン管理の方法です。予測、モデルのスナップショット、データセット、結果ログなどを保存するために使用します。\"\
  ,\n    \"初めに\": \"[はこちら](https://wandb.ai/site)：Artifacts の初めての使用方法についてガイドを書きました。\"\
  ,\n    \"Artifacts 構造\": {\n      \"1. **ArtifactsType**\": \"Artifactsは、特定のタイプに属します。例えば、モデルやデータセットなどがあります。\"\
  ,\n      \"2. **Versioning**\": \"Artifacts は、保存されるコンテンツをバージョン管理することで、より簡単に管理できます。\"\
  ,\n      \"3. **Aliases**\": \"Artifacts には、人間が読める別名を付けることができ、バージョン番号の代わりに使用できます。\"\
  \n    }\n  }\n}\n\n### 使用例\n\n#### Artifacts の初回利用\n\nArtifacts を使用するには、まず Python\
  \ ライブラリの wandb をインポートします。 \n\n```python\nimport wandb\n\n# プロジェクトを初期化します\nwandb.init(project=\"\
  my-artifacts-project\")\n\n# Artifacts を作成します\nartifact = wandb.Artifact('my-dataset',\
  \ type='dataset')\n\n# ファイルを追加します\nartifact.add_file('path/to/dataset.csv')\n\n\
  # Artifacts をアップロードします\nwandb.log_artifact(artifact)\n```\n\nこのコードスニペットは、W&B システムにデータセットファイルを保存する基本的なワークフローを示しています。"
toc_hide: true
type: docs
url: /ja/support/:filename
---

`wandb` をインストールする際に以下のエラーが発生した場合:

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

プリビルドのホイールから `psutil` を直接インストールしてください。Python バージョンとオペレーティングシステムを [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil) で確認してください。

例えば、Linux 上で Python 3.8 の場合、`psutil` をインストールするには:

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

`psutil` をインストールした後、`pip install wandb` を実行して `wandb` のインストールを完了してください。