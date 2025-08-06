---
title: エアギャップ環境向け Kubernetes オペレーター
description: Kubernetes Operator（エアギャップ環境）で W&B プラットフォームをデプロイする
menu:
  default:
    identifier: operator-airgapped
    parent: kubernetes-operator
---

## イントロダクション

このガイドは、エアギャップされたお客様管理環境への W&B Platform のデプロイ方法をステップごとに解説します。

Helm チャートやコンテナイメージのホスティングには、社内リポジトリやレジストリを利用してください。Kubernetes クラスターへの適切なアクセス権を持ったシェルコンソールで、すべてのコマンドを実行してください。

Kubernetes アプリケーションをデプロイする際に利用している継続的デリバリーツールでも、同様のコマンドを活用できます。

## ステップ 1: 前提条件

開始する前に、環境が以下の条件を満たしていることを確認してください。

- Kubernetes バージョン >= 1.28
- Helm バージョン >= 3
- 必要な W&B イメージが保存された社内コンテナレジストリへのアクセス
- W&B 用 Helm チャートが保存された社内 Helm リポジトリへのアクセス

## ステップ 2: 社内コンテナレジストリの準備

デプロイを進める前に、以下のコンテナイメージが社内コンテナレジストリに存在することを確認してください。
* [`docker.io/wandb/controller`](https://hub.docker.com/r/wandb/controller)
* [`docker.io/wandb/local`](https://hub.docker.com/r/wandb/local)
* [`docker.io/wandb/console`](https://hub.docker.com/r/wandb/console)
* [`docker.io/bitnami/redis`](https://hub.docker.com/r/bitnami/redis)
* [`docker.io/otel/opentelemetry-collector-contrib`](https://hub.docker.com/r/otel/opentelemetry-collector-contrib)
* [`quay.io/prometheus/prometheus`](https://quay.io/repository/prometheus/prometheus)
* [`quay.io/prometheus-operator/prometheus-config-reloader`](https://quay.io/repository/prometheus-operator/prometheus-config-reloader)

これらのイメージは W&B コンポーネントのデプロイ成功に不可欠です。コンテナレジストリの準備には WSM の使用を推奨します。

すでに社内コンテナレジストリを利用している場合は、そこにイメージを追加できます。まだの場合は、以下のセクションの手順に沿って WSM を使い、コンテナリポジトリを準備してください。

Operator の要件状況を把握し、イメージのアップグレードを [WSM を使用して]({{< relref "#list-images-and-their-versions" >}}) もしくはご自身のプロセスで確認・ダウンロードするのはお客様の責任となります。

### WSM のインストール

WSM のインストール方法は以下の通りです。

{{% alert %}}
WSM には、動作する Docker のインストールが必要です。
{{% /alert %}}

#### Bash
GitHub から直接 Bash スクリプトを実行します:

```bash
curl -sSL https://raw.githubusercontent.com/wandb/wsm/main/install.sh | bash
```
このスクリプトは実行ディレクトリーにバイナリをダウンロードします。別のフォルダに移動するには、以下のコマンドを実行してください。

```bash
sudo mv wsm /usr/local/bin
```

#### GitHub
W&B 管理の `wandb/wsm` GitHub リポジトリ（`https://github.com/wandb/wsm`）から WSM をダウンロードまたはクローンしてください。最新リリースは `wandb/wsm` の [リリースノート](https://github.com/wandb/wsm/releases) を参照してください。

### イメージ一覧とバージョンの取得

`wsm list` を使って、最新のイメージバージョン一覧を取得できます。

```bash
wsm list
```

出力例は次のようになります。

```text
:package: Starting the process to list all images required for deployment...
Operator Images:
  wandb/controller:1.16.1
W&B Images:
  wandb/local:0.62.2
  docker.io/bitnami/redis:7.2.4-debian-12-r9
  quay.io/prometheus-operator/prometheus-config-reloader:v0.67.0
  quay.io/prometheus/prometheus:v2.47.0
  otel/opentelemetry-collector-contrib:0.97.0
  wandb/console:2.13.1
Here are the images required to deploy W&B. Ensure these images are available in your internal container registry and update the values.yaml accordingly.
```

### イメージのダウンロード

`wsm download` を使用して、最新バージョンの全イメージをダウンロードします。

```bash
wsm download
```

出力例は次のようになります。

```text
Downloading operator helm chart
Downloading wandb helm chart
✓ wandb/controller:1.16.1
✓ docker.io/bitnami/redis:7.2.4-debian-12-r9
✓ otel/opentelemetry-collector-contrib:0.97.0
✓ quay.io/prometheus-operator/prometheus-config-reloader:v0.67.0
✓ wandb/console:2.13.1
✓ quay.io/prometheus/prometheus:v2.47.0

  Done! Installed 7 packages.
```

WSM は各イメージの `.tgz` アーカイブを `bundle` ディレクトリーにダウンロードします。

## ステップ 3: 社内 Helm チャートリポジトリの準備

コンテナイメージと合わせて、下記 Helm チャートも社内 Helm チャートリポジトリにあることを確認してください。前ステップで導入した WSM ツールでも Helm チャートのダウンロードが可能です。あるいはこちらからもダウンロードできます。

- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B Platform](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)

`operator` チャートは W&B Operator（コントローラーマネージャーとも呼ばれます）のデプロイに使用し、`platform` チャートはカスタムリソース定義（CRD）で設定された値を利用して W&B Platform をデプロイします。

## ステップ 4: Helm リポジトリの設定

次に、社内リポジトリから W&B Helm チャートを取得できるように Helm リポジトリを設定します。以下のコマンドを実行してください。

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## ステップ 5: Kubernetes オペレーターのインストール

W&B Kubernetes オペレーター（コントローラーマネージャーとも呼ばれます）は、W&B プラットフォームの各コンポーネントを管理します。エアギャップ環境へインストールする場合は、社内コンテナレジストリを利用するよう設定が必要です。

社内コンテナレジストリを利用するため、デフォルトのイメージ設定を上書きし、デプロイタイプを示すため `airgapped: true` キーを設定する必要があります。`values.yaml` ファイルを下記のように更新してください。

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

`tag` には、社内レジストリで利用可能なバージョンを指定してください。

Operator と CRD のインストール:
```bash
helm upgrade --install operator wandb/operator -n wandb --create-namespace -f values.yaml
```

サポートされている各種値の詳細については、[Kubernetes operator GitHub リポジトリ](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml) を参照してください。

## ステップ 6: W&B カスタムリソースの設定

W&B Kubernetes オペレーターのインストール後、Custom Resource（CR）を設定し、社内 Helm リポジトリやコンテナレジストリを参照するようにします。

この設定により、Kubernetes オペレーターは W&B プラットフォームの必要な各コンポーネントのデプロイ時に社内リポジトリ・レジストリを利用します。

以下の CR の例をコピーして、新しいファイル `wandb.yaml` として保存してください。

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
    
    # install: true の場合、ヘルムがこのデプロイ用の MySQL データベースをインストールします。外部の MySQL デプロイメントを使用する場合は `false` に設定してください。
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

Kubernetes オペレーターは、CR で指定された値を使い、あなたの社内リポジトリから `operator-wandb` Helm チャートを設定します。

すべてのタグやバージョンは、社内レジストリで利用可能なものに置き換えてください。

この設定ファイルの作成について詳しくは [こちら]({{< relref "/guides/hosting/hosting-options/self-managed/kubernetes-operator/#configuration-reference-for-wb-server" >}}) をご確認ください。

## ステップ 7: W&B プラットフォームのデプロイ

Kubernetes オペレーターと CR の設定が完了したら、`wandb.yaml` を適用して W&B プラットフォームをデプロイします。

```bash
kubectl apply -f wandb.yaml
```

## FAQ

デプロイ作業中によくある質問（FAQ）とトラブルシュートのヒントを以下にまとめます。

### ほかの Ingress クラスを使いたい場合は？
`values.yaml` の ingress 設定を修正することで、任意の ingress クラスを指定できます。

### 証明書バンドルに複数の証明書が含まれていますが動作しますか？
`values.yaml` の `customCACerts` セクションで、証明書ごとに複数エントリーに分割してください。

### Kubernetes オペレーターによる自動アップデートを防ぐ方法はありますか？
W&B console から自動アップデートをオフにできます。サポートされるバージョンについては、W&B の担当者にご相談ください。**Self-managed** インスタンスの場合はリリース日より 12 ヶ月間サポートされます。お客様自身で期間内にアップグレードを実施してください。サポート切れバージョンの利用は避けましょう。[リリースポリシーとプロセス]({{< relref "/ref/release-notes/release-policies.md" >}})を参照してください。

{{% alert %}}
**Self-managed** インスタンスをご利用のお客様は、サポートを維持するため最低でも四半期ごとに最新リリースへアップデートし、
最新の機能・パフォーマンス改善・修正を受け取ることを強く推奨します。
{{% /alert %}}

### 環境がパブリックリポジトリと接続できない場合でもデプロイできますか？
構成で `airgapped` を `true` に設定すれば、Kubernetes オペレーターは社内のリソースのみを使用し、パブリックリポジトリに接続を試みません。