---
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# セットアップ

### トレーニングコードで実行の名前をどのように設定できますか？

トレーニングスクリプトの最初に`wandb.init`を呼び出すときに、実験名を指定してください。次のようにします：`wandb.init(name="my_awesome_run")`。

### wandbをオフラインで実行できますか？

オフラインマシンでトレーニングして、後で結果を当社のサーバーにアップロードしたい場合は、以下の機能をご利用ください！

1. 環境変数 `WANDB_MODE=offline` を設定して、インターネットが不要なローカルにメトリクスを保存します。
2. ディレクトリーで`wandb init`を実行して、プロジェクト名を設定します。
3. `wandb sync YOUR_RUN_DIRECTORY` を実行して、メトリクスをクラウドサービスにプッシュし、ホストされたWebアプリで結果を表示します。

APIを使用して、`run.settings._offline` または `run.settings.mode` をwandb.init()の後に実行することで、実行がオフラインかどうかを確認できます。

#### [`wandb sync`](../../ref/cli/wandb-sync.md)を使用できるいくつかのユースケース

* インターネットがない場合。
* 完全に機能を無効にする必要がある場合。
* 何らかの理由で後で実行を同期することが望ましい場合。例：トレーニングマシンのリソースを使用しないようにするため。

### これはPython専用ですか？

現在、このライブラリはPython 2.7+および3.6+のプロジェクトでのみ動作します。上記のアーキテクチャーは、他の言語との統合を容易にするはずです。他の言語のモニタリングが必要な場合は、[contact@wandb.com](mailto:contact@wandb.com) までご連絡ください。

### Anacondaパッケージはありますか？
はい！`pip`または`conda`でインストールできます。後者の場合、[conda-forge](https://conda-forge.org)チャンネルからパッケージを取得する必要があります。

<Tabs
  defaultValue="pip"
  values={[
    {label: 'pip', value: 'pip'},
    {label: 'conda', value: 'conda'},
  ]}>
  <TabItem value="pip">

```bash
# conda envを作成
conda create -n wandb-env python=3.8 anaconda
# 作成したenvをアクティブ化
conda activate wandb-env
# このconda envでwandbをpipでインストール
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
このインストールで問題が発生した場合は、お知らせください。このAnaconda [パッケージの管理に関するドキュメント](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)には、役立つガイドがいくつかあります。

### gccがない環境でwandb Pythonライブラリをインストールするにはどうすればよいですか？

`wandb`のインストールを試みて以下のエラーが表示された場合：

```
unable to execute 'gcc': No such file or directory

error: command 'gcc' failed with exit status 1
```
`psutil`を、事前にビルドされたwheelから直接インストールできます。こちらからPythonのバージョンとOSを探してください: [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil)

例えば、LinuxのPython 3.8で`psutil`をインストールする場合：

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d

pip install $WHEEL_URL
```

`psutil`がインストールされた後、`pip install wandb`でwandbをインストールできます。

### W&BクライアントはPython 2をサポートしていますか？ <a href="#eol-python27" id="eol-python27"></a>

W&Bクライアントライブラリは、バージョン0.10までPython 2.7とPython 3の両方をサポートしていました。Python 2のサポートは終了したため、バージョン0.11以降はPython 2.7のサポートが中止されました。Python 2.7システムで`pip install --upgrade wandb`を実行するユーザーは、0.10.xシリーズの新しいリリースのみを受け取ります。0.10.xシリーズのサポートは、重要なバグ修正とパッチに限定されます。現在、バージョン0.10.33がPython 2.7をサポートする0.10.xシリーズの最後のバージョンです。

### W&BクライアントはPython 3.5をサポートしていますか？ <a href="#eol-python35" id="eol-python35"></a>

W&Bクライアントライブラリは、バージョン0.11までPython 3.5をサポートしていました。Python 3.5のサポートが終了したため、[バージョン0.12](https://github.com/wandb/wandb/releases/tag/v0.12.0)以降はサポートが中止されました。