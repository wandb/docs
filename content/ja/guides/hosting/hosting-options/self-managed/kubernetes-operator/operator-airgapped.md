---
title: Kubernetes operator for air-gapped instances
description: Kubernetes Operator を使用して W&B プラットフォーム をデプロイする (エアギャップ)
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-kubernetes-operator-operator-airgapped
    parent: kubernetes-operator
---

## イントロダクション

このガイドは、エアギャップされたカスタマー管理の環境に W&B プラットフォームをデプロイするためのステップバイステップの手順を提供します。

Helm チャートやコンテナイメージをホストするために内部リポジトリーまたはレジストリーを使用します。Kubernetes クラスターへの適切なアクセス権を持つシェルコンソールで、すべてのコマンドを実行します。

Kubernetes アプリケーションをデプロイするために使用する継続的デリバリーのツール内でも、同様のコマンドを利用できます。

## ステップ 1: 前提条件

開始する前に、次の要件を環境が満たしていることを確認してください。

- Kubernetes バージョン >= 1.28
- Helm バージョン >= 3
- 必要な W&B イメージを含む内部コンテナレジストリーへのアクセス
- W&B Helm チャートのための内部 Helm リポジトリーへのアクセス

## ステップ 2: 内部コンテナレジストリーの準備

デプロイメントを進める前に、次のコンテナイメージが内部コンテナレジストリーに存在することを確認しなければなりません:
* [`docker.io/wandb/controller`](https://hub.docker.com/r/wandb/controller)
* [`docker.io/wandb/local`](https://hub.docker.com/r/wandb/local)
* [`docker.io/wandb/console`](https://hub.docker.com/r/wandb/console)
* [`docker.io/bitnami/redis`](https://hub.docker.com/r/bitnami/redis)
* [`docker.io/otel/opentelemetry-collector-contrib`](https://hub.docker.com/r/otel/opentelemetry-collector-contrib)
* [`quay.io/prometheus/prometheus`](https://quay.io/repository/prometheus/prometheus)
* [`quay.io/prometheus-operator/prometheus-config-reloader`](https://quay.io/repository/prometheus-operator/prometheus-config-reloader)

これらのイメージは、W&B コンポーネントを正常にデプロイするために重要です。W&B は、コンテナレジストリーを準備するために WSM を使用することを推奨します。

もし組織がすでに内部コンテナレジストリーを使用しているならば、イメージを追加することができます。それ以外の場合は、次のセクションに従って、WSM というものを利用してコンテナリポジトリーを準備してください。

あなたは、オペレーターの要求を追跡し、イメージのアップグレードをチェックしダウンロードする責任を持っています。それは [WSM を使用して]({{< relref path="#list-images-and-their-versions" lang="ja" >}})や、組織独自のプロセスを用いることで行うことができます。

### WSM のインストール

次のいずれかの方法で WSM をインストールします。

{{% alert %}}
WSM は、機能している Docker インストールを必要とします。
{{% /alert %}}

#### Bash
GitHub から Bash スクリプトを直接実行します。

```bash
curl -sSL https://raw.githubusercontent.com/wandb/wsm/main/install.sh | bash
```
このスクリプトは、スクリプトを実行したフォルダーにバイナリをダウンロードします。別のフォルダーに移動する場合は、以下のコマンドを実行します。

```bash
sudo mv wsm /usr/local/bin
```

#### GitHub
GitHub リポジトリー `wandb/wsm` から WSM をダウンロードまたはクローンします。最新のリリースについては、`wandb/wsm` [リリースノート](https://github.com/wandb/wsm/releases)をご覧ください。

### イメージとそのバージョンのリスト

最新のイメージバージョンのリストを `wsm list` を使用して取得します。

```bash
wsm list
```

出力は以下のようになります。

```text
:package: デプロイメントに必要なすべてのイメージをリストするプロセスを開始中...
Operator イメージ:
  wandb/controller:1.16.1
W&B イメージ:
  wandb/local:0.62.2
  docker.io/bitnami/redis:7.2.4-debian-12-r9
  quay.io/prometheus-operator/prometheus-config-reloader:v0.67.0
  quay.io/prometheus/prometheus:v2.47.0
  otel/opentelemetry-collector-contrib:0.97.0
  wandb/console:2.13.1
ここに W&B をデプロイするために必要なイメージがあります。これらのイメージが内部コンテナレジストリーに存在することを確認し、`values.yaml` を適宜更新してください。
```

### イメージのダウンロード

最新のバージョンのすべてのイメージを `wsm download` でダウンロードします。

```bash
wsm download
```

出力は以下のようになります。

```text
オペレーターヘルムチャートのダウンロード
wandb ヘルムチャートのダウンロード
✓ wandb/controller:1.16.1
✓ docker.io/bitnami/redis:7.2.4-debian-12-r9
✓ otel/opentelemetry-collector-contrib:0.97.0
✓ quay.io/prometheus-operator/prometheus-config-reloader:v0.67.0
✓ wandb/console:2.13.1
✓ quay.io/prometheus/prometheus:v2.47.0

  完了！ 7 つのパッケージがインストールされました。
```

WSM は、各イメージの `.tgz` アーカイブを `bundle` ディレクトリーにダウンロードします。

## ステップ 3: 内部 Helm チャートリポジトリーの準備

コンテナイメージと共に、次の Helm チャートも内部 Helm チャートリポジトリーに存在することを確認する必要があります。前のステップで紹介した WSM ツールは、Helm チャートのダウンロードもできます。代わりに、以下からダウンロードしてください。

- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B Platform](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)

`operator` チャートは、W&B オペレーター (コントローラーマネージャとしても呼ばれます) をデプロイするために使用されます。`platform` チャートは、カスタムリソース定義 (CRD) に設定された値を使用して W&B プラットフォームをデプロイするために使用されます。

## ステップ 4: Helm リポジトリーの設定

現在、Helm リポジトリーを設定して、内部リポジトリーから W&B Helm チャートを取得できるようにします。次のコマンドを実行して Helm リポジトリーを追加し、更新します。

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## ステップ 5: Kubernetes オペレーターのインストール

W&B Kubernetes オペレーター (コントローラーマネージャとも呼ばれる) は、W&B プラットフォームコンポーネントの管理を担当します。エアギャップされた環境にインストールするには、内部コンテナレジストリーを使用するように構成する必要があります。

そのためには、既定のイメージ設定を上書きして内部コンテナレジストリーを使用し、デプロイメントタイプが期待されることを示すために `airgapped: true` キーを設定する必要があります。`values.yaml` ファイルを以下のように更新します。

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

内部レジストリーに存在するバージョンでタグを置き換えます。

オペレーターと CRD をインストールします。
```bash
helm upgrade --install operator wandb/operator -n wandb --create-namespace -f values.yaml
```

サポートされる値の詳細については、[Kubernetes オペレーター GitHub リポジトリー](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml) を参照してください。

## ステップ 6: W&B カスタムリソースの設定

W&B Kubernetes オペレーターをインストールした後、カスタムリソース (CR) を設定して内部 Helm リポジトリーおよびコンテナレジストリーを指すようにしなければなりません。

この設定により、Kubernetes オペレーターは内部レジストリーとリポジトリーを使用して、W&B プラットフォームの必要なコンポーネントをデプロイします。

この例の CR を `wandb.yaml` という新しいファイルにコピーします。

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
        name: s3.yourdomain.com:port #例: s3.yourdomain.com:9000
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
    
    # install: true の場合、Helm はデプロイメントが使用する MySQL データベースをインストールします。自身の外部 MySQL デプロイメントを使用するには `false` に設定してください。
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

W&B プラットフォームをデプロイするには、Kubernetes オペレーターが CR の値を使用して内部リポジトリーの `operator-wandb` Helm チャートを設定します。

すべてのタグ/バージョンを内部レジストリーで使用可能なバージョンに置き換えます。

前述の設定ファイルを作成する詳細については、[こちら]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#configuration-reference-for-wb-server" lang="ja" >}})をご覧ください。

## ステップ 7: W&B プラットフォームのデプロイ

現在 Kubernets オペレーターと CR が構成されているので、`wandb.yaml` の設定を適用して W&B プラットフォームをデプロイしましょう。

```bash
kubectl apply -f wandb.yaml
```

## FAQ

デプロイメントプロセス中に、以下のよくある質問 (FAQ) およびトラブルシューティングのヒントを参照してください。

### 別のイングレスクラスがあります。それを使えますか？
はい、`values.yaml` のイングレス設定を変更して自身のイングレスクラスを設定できます。

### 証明書バンドルに複数の証明書があります。それでも動作しますか？
`values.yaml` の `customCACerts` セクションで証明書を複数のエントリに分割する必要があります。

### Kubernetes オペレーターが未監ダウンロードテッド更新を適用するのを防ぐ方法はありますか？
W&B コンソールから自動更新をオフにできます。サポートされているバージョンについて質問がある場合は、W&B チームにお問い合わせください。また、W&B は過去 6 か月間にリリースされたプラットフォームバージョンをサポートします。W&B は定期的なアップグレードを行うことを推奨します。

### 環境がパブリックリポジトリーに接続していない場合、デプロイメントは動作しますか？
あなたの設定で `airgapped` を true に設定すると、Kubernetes オペレーターは内部リソースのみを使用し、パブリックリポジトリーに接続しようとはしません。