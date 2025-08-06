---
title: エアギャップ環境向け Kubernetes オペレーター
description: Kubernetes Operator（エアギャップ環境）で W&B プラットフォームをデプロイする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-kubernetes-operator-operator-airgapped
    parent: kubernetes-operator
---

## イントロダクション

このガイドでは、W&B プラットフォームをエアギャップされた顧客管理環境にデプロイするためのステップバイステップ手順を説明します。

Helm チャートやコンテナイメージをホストするために、社内リポジトリやレジストリを使用してください。全てのコマンドは、Kubernetes クラスターへ適切なアクセス権を持つシェルコンソールで実行してください。

Kubernetes アプリケーションをデプロイするために使用している継続的デリバリーツールでも、同様のコマンドを活用できます。

## ステップ 1: 前提条件

開始する前に、環境が以下の要件を満たしていることを確認してください。

- Kubernetes バージョン >= 1.28
- Helm バージョン >= 3
- 必要な W&B イメージが格納されている社内コンテナレジストリへのアクセス
- W&B Helm チャート用の社内 Helm リポジトリへのアクセス

## ステップ 2: 社内コンテナレジストリの準備

デプロイメントを進める前に、以下のコンテナイメージが社内コンテナレジストリで利用可能であることを確認してください。
* [`docker.io/wandb/controller`](https://hub.docker.com/r/wandb/controller)
* [`docker.io/wandb/local`](https://hub.docker.com/r/wandb/local)
* [`docker.io/wandb/console`](https://hub.docker.com/r/wandb/console)
* [`docker.io/bitnami/redis`](https://hub.docker.com/r/bitnami/redis)
* [`docker.io/otel/opentelemetry-collector-contrib`](https://hub.docker.com/r/otel/opentelemetry-collector-contrib)
* [`quay.io/prometheus/prometheus`](https://quay.io/repository/prometheus/prometheus)
* [`quay.io/prometheus-operator/prometheus-config-reloader`](https://quay.io/repository/prometheus-operator/prometheus-config-reloader)

これらのイメージは、W&B コンポーネントを正しくデプロイするために不可欠です。W&B では、WSM を使用してコンテナレジストリを準備することを推奨しています。

すでに社内コンテナレジストリを運用している場合は、イメージをそこに追加してください。そうでない場合は、次のセクションに従って WSM を利用し、コンテナリポジトリを準備してください。

Operator の要件管理やイメージのアップグレード確認・ダウンロードは、[WSM を使用する]({{< relref path="#list-images-and-their-versions" lang="ja" >}})か、組織独自のプロセスで必ず実施してください。

### WSM のインストール

以下のいずれかの方法で WSM をインストールします。

{{% alert %}}
WSM の利用には、Docker が正常にインストールされている必要があります。
{{% /alert %}}

#### Bash
Bash スクリプトを GitHub から直接実行します:

```bash
curl -sSL https://raw.githubusercontent.com/wandb/wsm/main/install.sh | bash
```
スクリプトは、実行したフォルダーにバイナリをダウンロードします。別のフォルダーに移動させるには、以下のコマンドを実行してください。

```bash
sudo mv wsm /usr/local/bin
```

#### GitHub
W&B 管理の `wandb/wsm` GitHub リポジトリ（`https://github.com/wandb/wsm`）からダウンロードまたはクローンします。最新リリースについては `wandb/wsm` の [リリースノート](https://github.com/wandb/wsm/releases) をご確認ください。

### イメージとバージョンの一覧取得

`wsm list` を使って、最新のイメージバージョン一覧を取得できます。

```bash
wsm list
```

出力例は以下のようになります。

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

`wsm download` を使って、全てのイメージの最新バージョンをダウンロードします。

```bash
wsm download
```

出力例は以下のようになります。

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

WSM は、それぞれのイメージについて `.tgz` 形式のアーカイブを `bundle` ディレクトリーにダウンロードします。

## ステップ 3: 社内 Helm チャートリポジトリの準備

コンテナイメージと同様に、以下の Helm チャートが社内の Helm チャートリポジトリで利用可能である必要があります。前述の WSM ツールでも Helm チャートをダウンロードできます。もしくは、以下から直接ダウンロードしてください。

- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B Platform](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)

`operator` チャートは、W&B Operator（Controller Manager とも呼ばれる）をデプロイするために使用します。`platform` チャートは、カスタムリソース定義（CRD）に設定された値を使って W&B Platform をデプロイするために使用されます。

## ステップ 4: Helm リポジトリの設定

社内リポジトリから W&B Helm チャートを取得できるよう、Helm リポジトリを設定します。以下のコマンドを実行して、Helm リポジトリを追加・更新してください。

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## ステップ 5: Kubernetes Operator のインストール

W&B Kubernetes Operator（Controller Manager とも呼ばれます）は、W&B プラットフォームコンポーネントの管理を担当します。エアギャップ環境でインストールするには、社内コンテナレジストリを使用するよう設定する必要があります。

そのためには、デフォルトのイメージ設定を上書きし、社内コンテナレジストリを使用するように設定し、また `airgapped: true` キーを指定して期待されるデプロイメントタイプであることを示します。`values.yaml` ファイルを以下のように更新してください。

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

tag の部分は、社内レジストリで利用可能なバージョンに置き換えてください。

Operator と CRD をインストールします。

```bash
helm upgrade --install operator wandb/operator -n wandb --create-namespace -f values.yaml
```

サポートされている値の詳細については、[Kubernetes Operator GitHub リポジトリ](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)をご覧ください。

## ステップ 6: W&B カスタムリソースの設定

W&B Kubernetes Operator をインストールした後、カスタムリソース（CR）を設定し、社内 Helm リポジトリとコンテナレジストリを参照するようにします。

この設定により、Kubernetes Operator は必要な W&B プラットフォームコンポーネントをデプロイする際に社内リソースのみを利用します。

以下の例を新規ファイル `wandb.yaml` として保存してください。

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
    
    # install: true にすると、Helm はデプロイ用の MySQL データベースをインストールします。独自の外部 MySQL を使う場合は `false` に設定してください。
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

Kubernetes Operator は、上記 CR の値を参照して社内リポジトリの `operator-wandb` Helm チャートを設定し、W&B プラットフォームをデプロイします。

全てのタグやバージョンについては、社内レジストリで利用可能なものに必ず置き換えてください。

上記設定ファイルの作成方法の詳細は[こちら]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#configuration-reference-for-wb-server" lang="ja" >}})をご参照ください。

## ステップ 7: W&B プラットフォームのデプロイ

Kubernetes Operator と CR の設定が完了したら、`wandb.yaml` を適用して W&B プラットフォームをデプロイします。

```bash
kubectl apply -f wandb.yaml
```

## よくある質問（FAQ）

デプロイ中によくある質問やトラブルシューティングのヒントは以下をご覧ください。

### 他の ingress クラスがあります。そのクラスは使用できますか？
はい、`values.yaml` 内の ingress 設定を変更することで、ご自身の ingress クラスを指定できます。

### 証明書バンドル内に複数の証明書が含まれています。これは動作しますか？
`values.yaml` の `customCACerts` セクションに、それぞれの証明書を個別のエントリーとして分割して記載する必要があります。

### Kubernetes Operator が自動アップデートを適用しないようにするには？
W&B Console から自動アップデートを無効にできます。サポート対象バージョンについてご質問がある場合は、W&B チームまでお問い合わせください。W&B では **Self-managed** インスタンスについて、初回リリース日から12ヶ月間 W&B Server のメジャーリリースをサポートしています。お客様ご自身で適切なタイミングでのアップグレードを実施し、サポートを維持してください。サポート対象外バージョンの継続利用は避けてください。[リリースポリシーと運用]({{< relref path="/ref/release-notes/release-policies.md" lang="ja" >}})もご参照ください。

{{% alert %}}
**Self-managed** インスタンスをご利用のお客様は、サポート維持・機能拡張・パフォーマンス向上、バグ修正のため、最低でも四半期ごとに最新リリースへアップデートすることを強く推奨します。
{{% /alert %}}

### 環境がパブリックリポジトリと接続できない場合でもデプロイは可能ですか？
設定で `airgapped` を `true` にしていれば、Kubernetes Operator は社内リソースのみを利用し、パブリックリポジトリへの接続は行いません。