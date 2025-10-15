---
title: エアギャップインスタンス用の Kubernetes オペレーター
description: Kubernetes Operator を使用して W&B プラットフォーム をデプロイする (Airgapped)
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-kubernetes-operator-operator-airgapped
    parent: kubernetes-operator
---

## イントロダクション

このガイドは、顧客管理のエアギャップ環境で W&B プラットフォームをデプロイするためのステップバイステップの手順を提供します。

Helm チャートとコンテナイメージをホストするために内部のリポジトリーまたはレジストリーを使用します。Kubernetes クラスターへの適切なアクセスを備えたシェルコンソールで、すべてのコマンドを実行してください。

Kubernetes アプリケーションをデプロイするために使用している任意の継続的デリバリーツールで、同様のコマンドを利用できます。

## ステップ 1: 前提条件

開始する前に、環境が次の要件を満たしていることを確認してください:

- Kubernetes バージョン >= 1.28
- Helm バージョン >= 3
- 必要な W&B イメージを備えた内部コンテナレジストリーへのアクセス
- W&B Helm チャートのための内部 Helm リポジトリーへのアクセス

## ステップ 2: 内部コンテナレジストリーの準備

デプロイメントを進める前に、以下のコンテナイメージが内部コンテナレジストリーに利用可能であることを確認する必要があります:
* [`docker.io/wandb/controller`](https://hub.docker.com/r/wandb/controller)
* [`docker.io/wandb/local`](https://hub.docker.com/r/wandb/local)
* [`docker.io/wandb/console`](https://hub.docker.com/r/wandb/console)
* [`docker.io/bitnami/redis`](https://hub.docker.com/r/bitnami/redis)
* [`docker.io/otel/opentelemetry-collector-contrib`](https://hub.docker.com/r/otel/opentelemetry-collector-contrib)
* [`quay.io/prometheus/prometheus`](https://quay.io/repository/prometheus/prometheus)
* [`quay.io/prometheus-operator/prometheus-config-reloader`](https://quay.io/repository/prometheus-operator/prometheus-config-reloader)

これらのイメージは、W&B コンポーネントの正常なデプロイメントに不可欠です。W&B はコンテナレジストリーを準備するために WSM を使用することをお勧めします。

もし組織がすでに内部コンテナレジストリーを使用している場合、イメージを追加することができます。そうでない場合は、次のセクションに進み、WSM と呼ばれるものを使用してコンテナリポジトリーを準備してください。

オペレーターの要件を追跡し、イメージのアップグレードを確認してダウンロードすることは、[WSM を使用して]({{< relref path="#list-images-and-their-versions" lang="ja" >}}) または組織独自のプロセスを使用して行う責任があります。

### WSM のインストール

WSM を次のいずれかのメソッドでインストールします。

{{% alert %}}
WSM は、動作する Docker インストールを必要とします。
{{% /alert %}}

#### Bash
Bash スクリプトを GitHub から直接実行します:

```bash
curl -sSL https://raw.githubusercontent.com/wandb/wsm/main/install.sh | bash
```
スクリプトは、スクリプトを実行したフォルダーにバイナリをダウンロードします。別のフォルダーに移動するには、次を実行します:

```bash
sudo mv wsm /usr/local/bin
```

#### GitHub
W&B が管理する `wandb/wsm` GitHub リポジトリーから WSM をダウンロードまたはクローンします。最新リリースについては、`wandb/wsm` [リリースノート](https://github.com/wandb/wsm/releases)を参照してください。

### イメージとそのバージョンの一覧表示

`wsm list` を使用して最新のイメージバージョンのリストを取得します。

```bash
wsm list
```

出力は次のようになります:

```text
:package: デプロイメントに必要なすべてのイメージを一覧表示するプロセスを開始しています...
オペレーターイメージ:
  wandb/controller:1.16.1
W&B イメージ:
  wandb/local:0.62.2
  docker.io/bitnami/redis:7.2.4-debian-12-r9
  quay.io/prometheus-operator/prometheus-config-reloader:v0.67.0
  quay.io/prometheus/prometheus:v2.47.0
  otel/opentelemetry-collector-contrib:0.97.0
  wandb/console:2.13.1
ここに W&B をデプロイするために必要なイメージがあります。これらのイメージが内部コンテナレジストリーで利用可能であることを確認し、`values.yaml` を適切に更新してください。
```

### イメージのダウンロード

最新バージョンのイメージをすべて `wsm download` を使用してダウンロードします。

```bash
wsm download
```

出力は次のようになります:

```text
オペレーター Helm chart のダウンロード
wandb Helm chart のダウンロード
✓ wandb/controller:1.16.1
✓ docker.io/bitnami/redis:7.2.4-debian-12-r9
✓ otel/opentelemetry-collector-contrib:0.97.0
✓ quay.io/prometheus-operator/prometheus-config-reloader:v0.67.0
✓ wandb/console:2.13.1
✓ quay.io/prometheus/prometheus:v2.47.0

  完了! 7 パッケージがインストールされました。
```

WSM は各イメージの `.tgz` アーカイブを `bundle` ディレクトリーにダウンロードします。

## ステップ 3: 内部 Helm チャートリポジトリーの準備

コンテナイメージとともに、以下の Helm チャートが内部 Helm チャートリポジトリーに利用可能であることも確認する必要があります。前のステップで導入した WSM ツールは Helm チャートをダウンロードすることもできます。別の方法として、こちらでダウンロードしてください:

- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B Platform](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)

`operator` チャートは W&B Oyserator 、つまりコントローラーマネージャーをデプロイするために使用されます。`platform` チャートは、カスタムリソース定義 (CRD) に設定された値を使用して W&B プラットフォームをデプロイするために使用されます。

## ステップ 4: Helm リポジトリーの設定

次に、W&B Helm チャートを内部リポジトリーからプルするために Helm リポジトリーを設定します。以下のコマンドを実行して、Helm リポジトリーを追加および更新します:

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## ステップ 5: Kubernetes オペレーターのインストール

W&B Kubernetes オペレーター、別名コントローラーマネージャーは、W&B プラットフォームのコンポーネントを管理する役割を果たします。エアギャップ環境でインストールするには、内部コンテナレジストリーを使用するように設定する必要があります。

そのためには、内部コンテナレジストリーを使用するためにデフォルトのイメージ設定をオーバーライドし、期待されるデプロイメントタイプを示すためにキー `airgapped: true` を設定する必要があります。以下のように `values.yaml` ファイルを更新します:

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

タグを内部レジストリーで利用可能なバージョンに置き換えてください。

オペレーターと CRD をインストールします:
```bash
helm upgrade --install operator wandb/operator -n wandb --create-namespace -f values.yaml
```

サポートされている値の詳細については、[Kubernetes オペレーター GitHub リポジトリー](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)を参照してください。

## ステップ 6: W&B カスタムリソースの設定

W&B Kubernetes オペレーターをインストールした後、内部 Helm リポジトリーおよびコンテナレジストリーを指すようにカスタムリソース (CR) を設定する必要があります。

この設定により、Kubernetes オペレーターが W&B プラットフォームに必要なコンポーネントをデプロイする際に、内部レジストリーとリポジトリーを使用することが保証されます。

この例の CR をコピーし、`wandb.yaml` という新しいファイルに名前を付けます。

```yaml
apiVersion: apps.wandb.com/v1
kind: WeightsAndBiases
metadata:
  labels:
    app.kubernetes.io/instance: wandb
    app.kubernetes.io/name: weightsandbiases
  name: wandb
  namespace: default

spec:
  chart:
    url: http://charts.yourdomain.com
    name: operator-wandb
    version: 0.18.0

  values:
    global:
      host: https://wandb.yourdomain.com
      license: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      bucket:
        accessKey: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        secretKey: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        name: s3.yourdomain.com:port #Ex.: s3.yourdomain.com:9000
        path: bucket_name
        provider: s3
        region: us-east-1
      mysql:
        database: wandb
        host: mysql.home.lab
        password: password
        port: 3306
        user: wandb
      extraEnv:
        ENABLE_REGISTRY_UI: 'true'
    
    # インストール: true の場合、Helm はデプロイメントが使用するための MySQL データベースをインストールします。独自の外部 MySQL デプロイメントを使用するには `false` に設定してください。
    mysql:
      install: false

    app:
      image:
        repository: registry.yourdomain.com/local
        tag: 0.59.2

    console:
      image:
        repository: registry.yourdomain.com/console
        tag: 2.12.2

    ingress:
      annotations:
        nginx.ingress.kubernetes.io/proxy-body-size: 64m
      class: nginx

    
```

Kubernetes オペレーターは、CR の値を使用して内部リポジトリーから `operator-wandb` Helm チャートを設定し、W&B プラットフォームをデプロイします。

すべてのタグ/バージョンを内部レジストリーで利用可能なバージョンに置き換えてください。

前述の設定ファイルの作成に関する詳細情報は[こちら]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#configuration-reference-for-wb-server" lang="ja" >}})にあります。

## ステップ 7: W&B プラットフォームのデプロイ

Kubernetes オペレーターと CR が設定されたので、`wandb.yaml` 設定を適用して W&B プラットフォームをデプロイします:

```bash
kubectl apply -f wandb.yaml
```

## FAQ

以下のよくある質問 (FAQs) およびデプロイメントプロセス中のトラブルシューティングのヒントを参照してください:

### 別のイングレスクラスがあります。それを使用できますか？
はい、`values.yaml` のイングレス設定を変更して、イングレスクラスを設定できます。

### 証明書バンドルに複数の証明書があります。それは機能しますか？
証明書を `values.yaml` の `customCACerts` セクションに複数のエントリに分割する必要があります。

### Kubernetes オペレーターが無人更新を適用するのを防ぐ方法はありますか？それは可能ですか？
W&B コンソールから自動更新をオフにできます。サポートされているバージョンについて質問がある場合は、W&B チームにお問い合わせください。また、W&B は過去 6 か月以内にリリースされたプラットフォームバージョンをサポートしていることを確認してください。W&B は定期的なアップグレードを推奨しています。

### 環境がパブリックリポジトリーに接続されていない場合、デプロイメントは機能しますか？
設定が `airgapped` を `true` に設定している場合、Kubernetes オペレーターは内部リソースのみを使用し、パブリックリポジトリーに接続しようとしません。