---
title: エアギャップ環境向けの Kubernetes Operator
description: W&B プラットフォームを Kubernetes Operator でデプロイ (エアギャップ環境)
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-kubernetes-operator-operator-airgapped
    parent: kubernetes-operator
---

## イントロダクション
このガイドでは、エアギャップ環境の顧客管理環境に W&B Platform をデプロイするための詳細な手順を説明します。
Helm chart とコンテナイメージをホストするには、内部リポジトリまたはレジストリを使用します。 Kubernetes クラスターへの適切な アクセス 権を持つシェルコンソールで、すべての コマンド を実行します。
Kubernetes アプリケーションのデプロイに使用する継続的デリバリーツールで、同様の コマンド を利用できます。

## ステップ 1: 前提条件
開始する前に、お使いの環境が以下の要件を満たしていることを確認してください。
- Kubernetes バージョン >= 1.28
- Helm バージョン >= 3
- 必要な W&B イメージを含む内部コンテナレジストリへの アクセス
- W&B Helm chart 用の内部 Helm リポジトリへの アクセス

## ステップ 2: 内部コンテナレジストリの準備
デプロイを進める前に、以下のコンテナイメージが内部コンテナレジストリで利用可能であることを確認する必要があります。これらのイメージは、W&B コンポーネントのデプロイを成功させるために不可欠です。 W&B では、組織独自のコンテナレジストリ管理 プロセス に従うか、または [WSM](#install-wsm) を使用して準備することをお勧めします。
W&B Operator の要件を追跡し、更新されたイメージを定期的にチェックして適用する責任は、お客様にあります。

### コア W&B コンポーネントコンテナ
* [`docker.io/wandb/controller`](https://hub.docker.com/r/wandb/controller)
* [`docker.io/wandb/local`](https://hub.docker.com/r/wandb/local)
* [`docker.io/wandb/console`](https://hub.docker.com/r/wandb/console)

### 依存関係
* [`docker.io/bitnamilegacy/redis`](https://hub.docker.com/r/bitnamilegacy/redis): W&B は、W&B のコンポーネントが使用するジョブのキューイングとデータキャッシュを処理するために、単一ノードの Redis 7.x デプロイメントに依存しています。概念実証のテストおよび開発中の便宜のため、W&B Self-Managed は、プロダクションデプロイメントには適さないローカルの Redis デプロイメントをデプロイします。ローカルの Redis デプロイメントを使用するには、このイメージがコンテナレジストリで利用可能であることを確認してください。
* [`docker.io/otel/opentelemetry-collector-contrib`](https://hub.docker.com/r/otel/opentelemetry-collector-contrib): W&B は、W&B で表示するために Kubernetes レイヤーのリソースから メトリクス とログを収集するために OpenTelemetry エージェント に依存しています。
* [`quay.io/prometheus/prometheus`](https://quay.io/repository/prometheus/prometheus): W&B は、W&B で表示するために様々なコンポーネントから メトリクス をキャプチャするために Prometheus に依存しています。
* [`quay.io/prometheus-operator/prometheus-config-reloader`](https://quay.io/repository/prometheus-operator/prometheus-config-reloader): Prometheus の必須の依存関係です。

### WSM のインストール
いずれかの メソッド を使用して WSM をインストールします。
{{% alert %}}
WSM には、動作する Docker インストールが必要です。
{{% /alert %}}

#### Bash
GitHub から Bash スクリプトを直接実行します。
```bash
curl -sSL https://raw.githubusercontent.com/wandb/wsm/main/install.sh | bash
```
このスクリプトは、スクリプトを実行したフォルダーにバイナリをダウンロードします。別のフォルダーに移動するには、以下を実行します。
```bash
sudo mv wsm /usr/local/bin
```

#### GitHub
`https://github.com/wandb/wsm` にある W&B マネージドの `wandb/wsm` GitHub リポジトリから WSM をダウンロードまたはクローンします。最新のリリースについては、`wandb/wsm` の [リリースノート](https://github.com/wandb/wsm/releases) を参照してください。

### イメージとそのバージョンのリスト表示
`wsm list` を使用して、イメージの最新バージョンのリストを取得します。
```bash
wsm list
```
出力は以下のようになります。
```text
:package: Starting the process to list all images required for deployment...
Operator Images:
  wandb/controller:1.16.1
W&B Images:
  wandb/local:0.62.2
  docker.io/bitnamilegacy/redis:7.2.4-debian-12-r9
  quay.io/prometheus-operator/prometheus-config-reloader:v0.67.0
  quay.io/prometheus/prometheus:v2.47.0
  otel/opentelemetry-collector-contrib:0.97.0
  wandb/console:2.13.1
```

### イメージのダウンロード
`wsm download` を使用して、すべてのイメージを最新バージョンでダウンロードします。
```bash
wsm download
```
出力は以下のようになります。
```text
Downloading operator helm chart
Downloading wandb helm chart
✓ wandb/controller:1.16.1
✓ docker.io/bitnamilegacy/redis:7.2.4-debian-12-r9
✓ otel/opentelemetry-collector-contrib:0.97.0
✓ quay.io/prometheus-operator/prometheus-config-reloader:v0.67.0
✓ wandb/console:2.13.1
✓ quay.io/prometheus/prometheus:v2.47.0

  Done! Installed 7 packages.
```
WSM は、各イメージの `.tgz` アーカイブを `bundle` ディレクトリーにダウンロードします。

## ステップ 3: 内部 Helm chart リポジトリの準備
コンテナイメージに加えて、以下の Helm chart が内部 Helm chart リポジトリで利用可能であることを確認する必要があります。 WSM ツールは Helm chart をダウンロードできます。または、以下から手動でダウンロードすることもできます。
- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B Platform](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)
`operator` chart は、Controller Manager とも呼ばれる W&B Operator をデプロイするために使用されます。 `platform` chart は、カスタムリソース定義 (CRD) で設定された 値 を使用して W&B Platform をデプロイするために使用されます。

## ステップ 4: Helm リポジトリの設定
次に、内部リポジトリから W&B Helm chart をプルするように Helm リポジトリを設定します。次の コマンド を実行して Helm リポジトリを追加および更新します。
```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## ステップ 5: Kubernetes Operator のインストール
Controller Manager としても知られる W&B Kubernetes Operator は、W&B Platform コンポーネントの管理を担当します。エアギャップ環境にインストールするには、内部コンテナレジストリを使用するように設定する必要があります。
そのためには、デフォルトのイメージ設定を上書きして内部コンテナレジストリを使用し、予期される デプロイメント タイプを示すために キー `airgapped: true` を設定する必要があります。以下に示すように `values.yaml` ファイルを更新します。
```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```
タグを内部レジストリで利用可能な バージョン に置き換えてください。
Operator と CRD をインストールします。
```bash
helm upgrade --install operator wandb/operator -n wandb --create-namespace -f values.yaml
```
サポートされている 値 の完全な詳細については、[Kubernetes Operator GitHub リポジトリ](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml) を参照してください。

## ステップ 6: W&B Custom Resource の設定
W&B Kubernetes Operator をインストールした後、Custom Resource (CR) を設定して内部 Helm リポジトリとコンテナレジストリを指すようにする必要があります。
この設定により、Kubernetes Operator は W&B Platform の必要なコンポーネントをデプロイするときに、内部レジストリとリポジトリが使用されることを保証します。
この CR の例を `wandb.yaml` という名前の新しいファイルにコピーします。
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
    
    # install: true の場合、Helm はデプロイメントが使用する MySQL データベースをインストールします。独自の外部 MySQL デプロイメントを使用するには、`false` に設定します。
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
W&B Platform をデプロイするために、Kubernetes Operator は CR の 値 を使用して内部リポジトリから `operator-wandb` Helm chart を設定します。
すべてのタグ/バージョンを内部レジストリで利用可能な バージョン に置き換えてください。

## ステップ 7: W&B Platform のデプロイ
Kubernetes Operator と CR が設定されたので、`wandb.yaml` 設定を適用して W&B Platform をデプロイします。
```bash
kubectl apply -f wandb.yaml
```

## FAQ
デプロイメント プロセス 中のよくある質問 (FAQ) とトラブルシューティングのヒントを以下に示します。

### 別の ingress class があります。その class を使用できますか？
はい、`values.yaml` の ingress 設定を変更することで、ingress class を設定できます。

### 証明書バンドルに複数の証明書があります。それで動作しますか？
証明書を `values.yaml` の `customCACerts` セクションで複数のエントリーに分割する必要があります。

### Kubernetes Operator が自動更新を適用しないようにするにはどうすればよいですか？可能ですか？
W&B console から自動更新をオフにできます。サポートされている バージョン に関する質問は、W&B チームにお問い合わせください。 W&B は、メジャー W&B Server リリースを最初のリリース日から 12 ヶ月間サポートします。**Self-managed** インスタンスをご利用のお客様は、サポートを維持するために期限内にアップグレードする責任があります。サポートされていない バージョン を使い続けないでください。 [リリースポリシーとプロセス]({{< relref path="/ref/release-notes/release-policies.md" lang="ja" >}}) を参照してください。
{{% alert %}}
W&B は、**Self-managed** インスタンスをご利用のお客様に対し、サポートを維持し、最新の機能、パフォーマンスの改善、修正を受け取るために、少なくとも四半期に一度、最新リリースでデプロイメントを更新することを強くお勧めします。
{{% /alert %}}

### 環境がパブリックリポジトリに接続できない場合でもデプロイメントは機能しますか？
設定で `airgapped` を `true` に設定した場合、Kubernetes Operator は内部リソースのみを使用し、パブリックリポジトリへの接続を試みません。