---
title: Kubernetes operator for air-gapped instances
description: Kubernetes Operator で W&B プラットフォーム をデプロイする (エアギャップ)
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-kubernetes-operator-operator-airgapped
    parent: kubernetes-operator
---

## イントロダクション

このガイドでは、エアギャップされた顧客管理環境に W&B Platform をデプロイするためのステップごとの手順を説明します。

内部リポジトリまたはレジストリを使用して、Helm chartとコンテナイメージをホストします。 Kubernetes クラスターへの適切なアクセス権を持つシェルコンソールですべてのコマンドを実行します。

Kubernetes アプリケーションのデプロイに使用する継続的デリバリー ツールで、同様のコマンドを利用できます。

## ステップ 1: 前提条件

開始する前に、ご使用の環境が次の要件を満たしていることを確認してください。

- Kubernetes バージョン >= 1.28
- Helm バージョン >= 3
- 必要な W&B イメージを持つ内部コンテナレジストリへのアクセス
- W&B Helm chart 用の内部 Helm リポジトリへのアクセス

## ステップ 2: 内部コンテナレジストリの準備

デプロイメントに進む前に、次のコンテナイメージが内部コンテナレジストリで利用可能であることを確認する必要があります。
* [`docker.io/wandb/controller`](https://hub.docker.com/r/wandb/controller)
* [`docker.io/wandb/local`](https://hub.docker.com/r/wandb/local)
* [`docker.io/wandb/console`](https://hub.docker.com/r/wandb/console)
* [`docker.io/bitnami/redis`](https://hub.docker.com/r/bitnami/redis)
* [`docker.io/otel/opentelemetry-collector-contrib`](https://hub.docker.com/r/otel/opentelemetry-collector-contrib)
* [`quay.io/prometheus/prometheus`](https://quay.io/repository/prometheus/prometheus)
* [`quay.io/prometheus-operator/prometheus-config-reloader`](https://quay.io/repository/prometheus-operator/prometheus-config-reloader)

これらのイメージは、W&B コンポーネントを正常にデプロイするために重要です。 W&B は、WSM を使用してコンテナレジストリを準備することを推奨します。

組織がすでに内部コンテナレジストリを使用している場合は、イメージをそこに追加できます。 それ以外の場合は、次のセクションに従って、WSM と呼ばれるものを使用してコンテナリポジトリを準備します。

Operator の要件の追跡、およびイメージのアップグレードの確認とダウンロードは、[WSM を使用]({{< relref path="#list-images-and-their-versions" lang="ja" >}})するか、組織独自のプロセスを使用することによって、ユーザーが行う必要があります。

### WSM のインストール

次のいずれかの方法を使用して WSM をインストールします。

{{% alert %}}
WSM には、機能する Docker インストールが必要です。
{{% /alert %}}

#### Bash
GitHub から Bash スクリプトを直接実行します。

```bash
curl -sSL https://raw.githubusercontent.com/wandb/wsm/main/install.sh | bash
```
スクリプトは、スクリプトを実行したフォルダーにバイナリをダウンロードします。 別のフォルダーに移動するには、次を実行します。

```bash
sudo mv wsm /usr/local/bin
```

#### GitHub
W&B が管理する GitHub リポジトリ `wandb/wsm`（`https://github.com/wandb/wsm`）から WSM をダウンロードまたはクローンします。 最新リリースについては、`wandb/wsm` の [リリースノート](https://github.com/wandb/wsm/releases)を参照してください。

### イメージとそのバージョンのリスト

`wsm list` を使用して、イメージバージョンの最新リストを取得します。

```bash
wsm list
```

出力は次のようになります。

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

`wsm download` を使用して、最新バージョンのすべてのイメージをダウンロードします。

```bash
wsm download
```

出力は次のようになります。

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

WSM は、各イメージの `.tgz` アーカイブを `bundle` ディレクトリーにダウンロードします。

## ステップ 3: 内部 Helm chart リポジトリの準備

コンテナイメージに加えて、次の Helm chartが内部 Helm Chart リポジトリで利用可能であることも確認する必要があります。 前のステップで紹介した WSM ツールは、Helm chartもダウンロードできます。 または、ここでダウンロードしてください。

- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B Platform](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)

`operator` chartは、Controller Manager とも呼ばれる W&B Operator をデプロイするために使用されます。 `platform` chartは、カスタムリソース定義 (CRD) で構成された値を使用して W&B Platform をデプロイするために使用されます。

## ステップ 4: Helm リポジトリの設定

次に、内部リポジトリから W&B Helm chartをプルするように Helm リポジトリを設定します。 次のコマンドを実行して、Helm リポジトリを追加および更新します。

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## ステップ 5: Kubernetes operator のインストール

コントローラマネージャとも呼ばれる W&B Kubernetes operator は、W&B platform コンポーネントの管理を担当します。 エアギャップ環境にインストールするには、内部コンテナレジストリを使用するように構成する必要があります。

これを行うには、内部コンテナレジストリを使用するようにデフォルトのイメージ設定をオーバーライドし、キー `airgapped: true` を設定して、予想されるデプロイメントタイプを示す必要があります。 次に示すように、`values.yaml` ファイルを更新します。

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

タグを、内部レジストリで使用可能なバージョンに置き換えます。

operator と CRD をインストールします。
```bash
helm upgrade --install operator wandb/operator -n wandb --create-namespace -f values.yaml
```

サポートされている値の詳細については、[Kubernetes operator GitHub リポジトリ](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)を参照してください。

## ステップ 6: W&B カスタムリソースの設定

W&B Kubernetes operator をインストールしたら、内部 Helm リポジトリとコンテナレジストリを指すようにカスタムリソース (CR) を構成する必要があります。

この設定により、Kubernetes operator は、W&B platform の必要なコンポーネントをデプロイするときに、内部レジストリとリポジトリを使用することが保証されます。

この CR の例を `wandb.yaml` という新しいファイルにコピーします。

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
    
    # If install: true, Helm installs a MySQL database for the deployment to use. Set to `false` to use your own external MySQL deployment.
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

W&B platform をデプロイするために、Kubernetes Operator は CR の値を使用して、内部リポジトリから `operator-wandb` Helm chart を構成します。

すべてのタグ/バージョンを、内部レジストリで使用可能なバージョンに置き換えます。

上記の構成ファイルの作成の詳細については、[こちら]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#configuration-reference-for-wb-server" lang="ja" >}})をご覧ください。

## ステップ 7: W&B platform のデプロイ

Kubernetes operator と CR が構成されたので、`wandb.yaml` 構成を適用して W&B platform をデプロイします。

```bash
kubectl apply -f wandb.yaml
```

## FAQ

デプロイメントプロセス中に、以下のよくある質問 (FAQ) とトラブルシューティングのヒントを参照してください。

### 別の ingress クラスがあります。 そのクラスを使用できますか？
はい、`values.yaml` で ingress 設定を変更することで、ingress クラスを構成できます。

### 証明書バンドルに複数の証明書があります。 それは機能しますか？
証明書を `values.yaml` の `customCACerts` セクションの複数のエントリに分割する必要があります。

### Kubernetes operator が無人アップデートを適用するのを防ぐにはどうすればよいですか。 それは可能ですか？
W&B console から自動アップデートをオフにすることができます。 サポートされているバージョンに関する質問については、W&B チームにお問い合わせください。 また、W&B は過去 6 か月以内にリリースされた platform バージョンをサポートしていることに注意してください。 W&B は定期的なアップグレードを実行することをお勧めします。

### 環境がパブリックリポジトリに接続されていない場合、デプロイメントは機能しますか？
構成で `airgapped` を `true` に設定すると、Kubernetes operator は内部リソースのみを使用し、パブリックリポジトリへの接続を試みません。
