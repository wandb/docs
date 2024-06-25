---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# セットアップ

### トレーニングコードでrunの名前を設定するにはどうすればいいですか？

トレーニングスクリプトの冒頭で `wandb.init` を呼び出す際に、例えば次のように実験名を渡します: `wandb.init(name="my_awesome_run")`。

### wandb をオフラインで実行できますか？

オフラインのマシンでトレーニングを行い、後で結果をサーバーにアップロードしたい場合に利用できる機能があります。

1. 環境変数 `WANDB_MODE=offline` を設定し、メトリクスをローカルに保存します。この際、インターネットは不要です。
2. 準備が整ったら `wandb init` をディレクトリーで実行し、プロジェクト名を設定します。
3. `wandb sync YOUR_RUN_DIRECTORY` を実行してメトリクスをクラウドサービスにプッシュし、ホストされているウェブアプリで結果を確認します。

`wandb.init()` の後に `run.settings._offline` もしくは `run.settings.mode` を使用してrunがオフラインかどうかをAPIで確認できます。

#### [`wandb sync`](../../ref/cli/wandb-sync.md) を使用するいくつかのユースケース

* インターネットがない場合。
* すべてを完全に無効にする必要がある場合。
* 理由があって後でrunを同期したい場合。例えば、トレーニングマシンのリソース使用を避けたい場合。

### これはPythonだけで動作しますか？

現在、このライブラリはPython 2.7+ & 3.6+ プロジェクトでのみ動作します。上記のアーキテクチャーにより、他の言語とも簡単に統合できるはずです。もし他の言語のモニタリングが必要な場合は、[contact@wandb.com](mailto:contact@wandb.com) までご連絡ください。

### anacondaパッケージはありますか？

はい！`pip` または `conda` でインストールできます。後者の場合、[conda-forge](https://conda-forge.org) チャンネルからパッケージを取得する必要があります。

<Tabs
  defaultValue="pip"
  values={[
    {label: 'pip', value: 'pip'},
    {label: 'conda', value: 'conda'},
  ]}>
  <TabItem value="pip">

```bash
# conda環境を作成
conda create -n wandb-env python=3.8 anaconda
# 作成した環境をアクティベート
conda activate wandb-env
# このconda環境にpipでwandbをインストール
pip install wandb
```

  </TabItem>
  <TabItem value="conda">

```
conda activate myenv
conda install wandb --channel conda-forge
```

  </TabItem>
</Tabs>

このインストールで問題が発生した場合はお知らせください。このAnacondaの[パッケージ管理に関するドキュメント](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)に役立つガイダンスがあります。

### gccがない環境でwandb Pythonライブラリをインストールするには？

`wandb` をインストールしようとして次のエラーが表示される場合：

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

事前にビルドされたホイールから直接 `psutil` をインストールできます。PythonバージョンとOSに合ったものをここから見つけてください: [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil)

例えば、LinuxでPython 3.8に `psutil` をインストールするには：

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

`psutil` がインストールされた後、`pip install wandb` を使用してwandbをインストールできます。

### W&BクライアントはPython 2をサポートしていますか？ <a href="#eol-python27" id="eol-python27"></a>

W&Bクライアントライブラリはバージョン0.10までPython 2.7とPython 3の両方をサポートしていました。Python 2のサポート終了に伴い、バージョン0.11からPython 2.7のサポートは終了しました。Python 2.7システムで `pip install --upgrade wandb` を実行するユーザーは0.10.xシリーズの新しいリリースのみを取得します。0.10.xシリーズのサポートは重大なバグ修正とパッチに限定されます。現在、バージョン0.10.33がPython 2.7をサポートする0.10.xシリーズの最後のバージョンです。

### W&BクライアントはPython 3.5をサポートしていますか？ <a href="#eol-python35" id="eol-python35"></a>

W&Bクライアントライブラリはバージョン0.11までPython 3.5をサポートしていました。Python 3.5のサポート終了に伴い、[バージョン0.12](https://github.com/wandb/wandb/releases/tag/v0.12.0) からサポートは終了しました。