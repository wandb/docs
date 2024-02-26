---
description: Run Weights and Biases on your own machines using Docker
displayed_sidebar: default
---

# 基本セットアップ

Dockerを使ってWeights and Biasesを自分のマシンで実行しましょう。

### インストール

- [Docker](https://www.docker.com) と [Python](https://www.python.org) がインストールされている任意のマシンで以下を実行します:

```
pip install wandb
wandb server start
```

### ログイン

初めてログインする場合は、ローカルのW&Bサーバーアカウントを作成し、APIキーを認証する必要があります。

複数のマシンで`wandb`を実行している場合や、プライベートインスタンスとwandbクラウド間で切り替える場合は、runsを記録する場所を制御する方法がいくつかあります。共有プライベートインスタンスにメトリクスを送信し、DNSを設定した場合は、

- ログインするたびに、ホストフラグをプライベートインスタンスのアドレスに設定します:

```
 wandb login --host=http://wandb.your-shared-local-host.com
```

- 環境変数`WANDB_BASE_URL`をローカルインスタンスのアドレスに設定します:
```python
export WANDB_BASE_URL="http://wandb.your-shared-local-host.com"
```

自動化された環境では、`WANDB_API_KEY` を [wandb.your-shared-local-host.com/authorize](http://wandb.your-shared-local-host.com/authorize) でアクセスできるように設定できます。

wandbの公開されている**クラウド**インスタンスにログを記録するように切り替えるには、ホストを `api.wandb.ai` に設定します:

```
wandb login --cloud
```

または

```python
export WANDB_BASE_URL="https://api.wandb.ai"
```

また、ブラウザでクラウド上のwandbアカウントにログインしている場合、[https://wandb.ai/settings](https://wandb.ai/settings) でクラウドAPIキーに切り替えることもできます。

### 無料ライセンスを生成する

W&Bサーバーの設定を完了させるには、ライセンスが必要です。[**デプロイマネージャーを開く**](https://deploy.wandb.ai/deploy) から無料ライセンスを生成してください。まだクラウドアカウントを持っていない場合は、無料ライセンスを生成するために作成する必要があります。2つのオプションがあります:

1. [**個人用ライセンス ->**](https://deploy.wandb.ai/deploy) は個人の作業に対して永久に無料です: ![](/images/hosting/personal_license.png)
2. [**チーム試用ライセンス ->**](https://deploy.wandb.ai/deploy) は無料であり、30日間有効で、チームを設定し、スケーラブルなバックエンドに接続することができます: ![](/images/hosting/team_trial_license.png)

### ローカルホストにライセンスを追加する

1. デプロイメントからライセンスをコピーして、W&Bサーバーのローカルホストに戻ります: ![](/images/hosting/add_license_local_host.png)
2. ローカルの設定に追加するために、ローカルホストの `/system-admin` ページに貼り付けます:
   ![](@site/static/images/hosting/License.gif)
### アップグレード

定期的に、_wandb/local_ の新バージョンがDockerHubにプッシュされています。アップグレードするには、以下のように実行できます：

```shell
$ wandb server start --upgrade
```

手動でインスタンスをアップグレードするには、以下のように実行します

```shell
$ docker pull wandb/local
$ docker stop wandb-local
$ docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local
```

### 永続性とスケーラビリティ

- すべてのメタデータやファイルは、W&Bサーバーの`/vol`ディレクトリーに保存されます。この場所に永続的なボリュームをマウントしない場合、dockerプロセスが終了するとすべてのデータが失われます。
- このソリューションは[プロダクション](/guides/hosting/hosting-options)ワークロードには適していません。
- メタデータは外部のMySQLデータベースに、ファイルは外部のストレージバケットに保存することができます。
- 根底にあるファイルストアはリサイズ可能である必要があります。最小ストレージしきい値を超えた際に、根底にあるファイルシステムをリサイズするように警告を設定してください。
- エンタープライズ試験では、画像/ビデオ/オーディオのヘビーワークロードではない場合、少なくとも100GBの空き容量を根底にあるボリュームに推奨します。

#### wandbはどのようにユーザーアカウントデータを永続化しますか？

Kubernetesインスタンスが停止されると、wandbアプリケーションはすべてのユーザーアカウントデータをtarballにまとめて、S3オブジェクトストアにアップロードします。インスタンスを再起動し、`BUCKET`環境変数を指定すると、wandbは以前にアップロードされたtarballを取得し、新しく開始されたKubernetes展開にユーザーアカウント情報を読み込みます。

Wandbは、外部バケットが設定されているときにインスタンス設定を保持します。また、バケットに証明書やシークレットも保持していますが、適切なシークレットストアに移動するか、少なくとも暗号化のレイヤーを追加するべきです。外部オブジェクトストアが有効になっている場合、すべてのユーザーデータを含むため、強力なアクセス制御を適用する必要があります。
#### 共有インスタンスの作成とスケーリング

W&Bの強力な協力機能を活用するには、中央サーバー上に共有インスタンスが必要です。[AWS、GCP、Azure、Kubernetes、またはDocker上で設定](/guides/hosting/hosting-options)することができます。

:::warning

**試験モードとプロダクションのセットアップ**

W&Bローカルの試用モードでは、Dockerコンテナを単一のマシンで実行しています。このセットアップは簡単で痛みがなく、製品のテストに適していますが、スケーラブルではありません。

テストプロジェクトから本格的なプロダクション作業に移行する準備ができたら、データ損失を回避するためにスケーラブルなファイルシステムを設定することが重要です。余分なスペースを事前に割り当て、データをログに記録するにつれてファイルシステムを積極的にリサイズし、外部のメタデータおよびオブジェクトストアをバックアップ用に構成します。ディスク容量が不足すると、インスタンスが停止し、追加のデータが失われます。

:::

[**セールスにお問い合わせ -**](https://wandb.ai/site/contact)**>** W&Bサーバーのエンタープライズオプションについて詳しく知る。