---
description: 自分のマシンで Docker を使って Weights and Biases を実行
displayed_sidebar: default
---


# Getting started

この "Hello, world!" の例に従って、Dedicated Cloudおよび自己管理ホスティングオプション用にW&B Serverをインストールする一般的なワークフローを学びましょう。このデモの終わりには、Trial ModeのW&Bライセンスを使用してローカルマシンでW&B Serverをホストする方法を理解できます。

デモンストレーションのために、このデモではポート`8080`（`localhost:8080`）でローカル開発サーバーを使用します。

:::tip
**Trial ModeとProduction設定**

Trial Modeでは、Dockerコンテナを単一のマシンで実行します。この設定は製品のテストに理想的ですが、スケーラブルではありません。

プロダクション作業には、データ損失を避けるためにスケーラブルなファイルシステムを設定してください。W&Bは以下を強く推奨します：
* 事前に余分なスペースを確保する
* より多くのデータをログするにつれて、ファイルシステムを積極的にリサイズする
* バックアップのために外部メタデータおよびオブジェクトストアを設定する
:::

## Prerequisites
始める前に、ローカルマシンが以下の要件を満たしていることを確認してください：

1. [Python](https://www.python.org) をインストール
2. [Docker](https://www.docker.com) をインストールし、実行中であることを確認
3. 最新バージョンのW&Bをインストールまたはアップグレード：
   ```bash
   pip install --upgrade wandb
   ```
##  1. W&B Dockerイメージをプル

ターミナルで以下を実行します：

```bash
wandb server start
```

このコマンドは最新のW&B Dockerイメージ [`wandb/local`](https://hub.docker.com/r/wandb/local) をプルします。

## 2. W&Bアカウントを作成
`http://localhost:8080/signup` に移動し、初期ユーザーアカウントを作成します。名前、メールアドレス、ユーザー名、およびパスワードを入力します：

![](/images/hosting/signup_localhost.png)

**Sign Up** ボタンをクリックして、W&Bアカウントを作成します。

:::note
このデモのために、既存のW&Bアカウントを持っている場合でも新しいW&Bアカウントを作成してください。
:::

### APIキーをコピー
アカウントを作成した後、`http://localhost:8080/authorize` に移動します。

画面に表示されるW&B APIキーをコピーします。このキーは後のステップでログイン資格を確認するために使用します。

![](/images/hosting/copy_api_key.png)

## 3. ライセンスを生成
W&B Deploy Managerにアクセスしてhttps://deploy.wandb.ai/deploy でTrial Mode W&Bライセンスを生成します。

1. プロバイダーとしてDockerを選択
![](/images/hosting/deploy_manager_platform.png)
2. **Next** をクリック
3. **Owner of license** ドロップダウンからライセンスのオーナーを選択
![](/images/hosting/deploy_manager_info.png)
4. **Next** をクリック
5. **Name of Instance** フィールドにライセンスの名前を入力
6. (オプション) **Description** フィールドにライセンスについての説明を入力
7. **Generate License Key** ボタンをクリック
![](/images/hosting/deploy_manager_generate.png)

**Generate License Key** をクリックすると、W&Bはデプロイメントライセンスページにリダイレクトします。デプロイメントライセンスページでは、デプロイメントIDやライセンスが所属する組織などのライセンスインスタンスに関する情報を閲覧できます。

:::tip
特定のライセンスインスタンスを表示するには、次の2通りの方法があります：
1. Deploy Manager UIに移動し、ライセンスインスタンスの名前をクリック
2. `https://deploy.wandb.ai/DeploymentID` に直接アクセス。この`DeploymentID`はライセンスインスタンスに割り当てられた一意のIDです。
:::

## 4. トライアルライセンスをローカルホストに追加
1. ライセンスインスタンスのデプロイメントライセンスページで、**Copy License** ボタンをクリック
![](/images/hosting/deploy_manager_get_license.png)
2. `http://localhost:8080/system-admin/` に移動
3. ライセンスを**License field** に貼り付ける
![](/images/hosting/License.gif)
4. **Update settings** ボタンをクリック

## 5. W&BアプリUIがブラウザで実行されていることを確認
ローカルマシンでW&Bが実行されていることを確認します。`http://localhost:8080/home` に移動します。ブラウザにW&BアプリUIが表示されるはずです。

![](/images/hosting/check_local_host.png)

## 6. ローカルW&Bインスタンスにプログラムアクセスを追加

1. APIキーを取得するために `http://localhost:8080/authorize` に移動
2. ターミナルで以下を実行：
   ```bash
   wandb login --host=http://localhost:8080/
   ```
   既に異なるアカウントでW&Bにログインしている場合は、`relogin` フラグを追加：
   ```bash
   wandb login --relogin --host=http://localhost:8080
   ```
3. プロンプトが表示されたらAPIキーを貼り付け

W&Bは`localhost`のプロファイルとAPIキーを`.netrc`プロファイルに追加します。このファイルは`/Users/username/.netrc`にあります。
 
## データを保持するためのボリュームを追加

ログするすべてのメタデータとファイルは一時的に `https://deploy.wandb.ai/vol` ディレクトリに保存されます。

ボリュームまたは外部ストレージをDockerコンテナにマウントして、ローカルW&Bインスタンスに保存されたファイルやメタデータを保持します。W&Bは、メタデータを外部のMySQLデータベースに、ファイルをAmazon S3などの外部ストレージバケットに保存することを推奨します。

:::info
ローカルのW&Bインスタンス（Trial W&Bライセンスを使用して作成された）は、Dockerを使用してローカルブラウザでW&Bを実行します。デフォルトでは、Dockerコンテナが存在しなくなるとデータは保持されません。`https://deploy.wandb.ai/vol` にボリュームをマウントしない限り、Dockerプロセスが終了するとデータは失われます。
:::

ボリュームのマウント方法やDockerがデータを管理する方法の詳細については、Dockerドキュメントの[Manage data in Docker](https://docs.docker.com/storage/) ページを参照してください。

### ボリュームに関する考慮事項
基盤となるファイルストアはリサイズ可能である必要があります。
W&Bは、最小ストレージしきい値に達する前に通知するアラートを設定し、ファイルシステムをリサイズできるようにすることを推奨します。

:::info
企業トライアルでは、画像/ビデオ/オーディオを多用しないワークロードに対して少なくとも100GBの空きスペースをボリュームに確保することをW&Bは推奨します。
:::
