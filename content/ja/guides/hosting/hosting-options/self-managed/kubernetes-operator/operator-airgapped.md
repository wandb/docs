---
title: Kubernetes operator for air-gapped instances
description: Kubernetes Operator を使用して W&B Platform をデプロイする (エアギャップ)
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-kubernetes-operator-operator-airgapped
    parent: kubernetes-operator
---

## イントロダクション

この ガイド では、エアギャップされた顧客管理 環境 に W&B Platform を デプロイ するためのステップごとの 手順 を説明します。

内部 リポジトリ または レジストリ を使用して、Helm チャート と コンテナ イメージ をホストします。Kubernetes クラスター への適切な アクセス 権を持つ シェル コンソール で全ての コマンド を 実行 します。

Kubernetes アプリケーション の デプロイ に使用する継続的な デリバリー ツール で、同様の コマンド を利用できます。

## ステップ 1：前提条件

開始する前に、ご使用の 環境 が以下の要件を満たしていることを確認してください。

- Kubernetes バージョン >= 1.28
- Helm バージョン >= 3
- 必要な W&B イメージ を持つ内部 コンテナ レジストリ への アクセス
- W&B Helm チャート 用の内部 Helm リポジトリ への アクセス

## ステップ 2：内部 コンテナ レジストリ の準備

デプロイメント を進める前に、以下の コンテナ イメージ が内部 コンテナ レジストリ で利用可能であることを確認する必要があります。
* [`docker.io/wandb/controller`](https://hub.docker.com/r/wandb/controller)
* [`docker.io/wandb/local`](https://hub.docker.com/r/wandb/local)
* [`docker.io/wandb/console`](https://hub.docker.com/r/wandb/console)
* [`docker.io/bitnami/redis`](https://hub.docker.com/r/bitnami/redis)
* [`docker.io/otel/opentelemetry-collector-contrib`](https://hub.docker.com/r/otel/opentelemetry-collector-contrib)
* [`quay.io/prometheus/prometheus`](https://quay.io/repository/prometheus/prometheus)
* [`quay.io/prometheus-operator/prometheus-config-reloader`](https://quay.io/repository/prometheus-operator/prometheus-config-reloader)

これらの イメージ は、W&B コンポーネント の デプロイ を成功させるために重要です。W&B は、WSM を使用して コンテナ レジストリ を準備することを推奨します。

組織がすでに内部 コンテナ レジストリ を使用している場合は、イメージ をそれに追加できます。そうでない場合は、次の セクション に従って、WSM という名前を使用して コンテナ リポジトリ を準備します。

[WSM を使用]({{< relref path="#list-images-and-their-versions" lang="ja" >}})するか、組織独自の プロセス を使用して、Operator の要件を追跡し、イメージ のアップグレードをチェックしてダウンロードする責任があります。

### WSM の インストール

次のいずれかの 方法 で WSM を インストール します。

{{% alert %}}
WSM には、機能する Docker の インストール が必要です。
{{% /alert %}}

#### Bash
GitHub から Bash スクリプト を直接 実行 します。

```bash
curl -sSL https://raw.githubusercontent.com/wandb/wsm/main/install.sh | bash
```
スクリプト は、スクリプト を 実行 した フォルダー に バイナリ をダウンロードします。別の フォルダー に移動するには、次を 実行 します。

```bash
sudo mv wsm /usr/local/bin
```

#### GitHub
W&B が管理する `wandb/wsm` GitHub リポジトリ （`https://github.com/wandb/wsm`）から WSM をダウンロードまたは クローン します。最新 リリース については、`wandb/wsm` の[リリース ノート](https://github.com/wandb/wsm/releases)を参照してください。

### イメージ とその バージョン の一覧表示

`wsm list`を使用して、イメージ バージョン の最新の リスト を取得します。

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

### イメージ のダウンロード

`wsm download`を使用して、最新 バージョン の全ての イメージ をダウンロードします。

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

WSM は、各 イメージ の `.tgz` アーカイブ を `bundle` ディレクトリー にダウンロードします。

## ステップ 3：内部 Helm チャート リポジトリ の準備

コンテナ イメージ と共に、次の Helm チャート が内部 Helm Chart リポジトリ で利用可能であることを確認する必要もあります。前の ステップ で紹介した WSM ツール は、Helm チャート をダウンロードすることもできます。または、こちらからダウンロードしてください。

- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B Platform](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)

`operator` チャート は、Controller Manager とも呼ばれる W&B Operator の デプロイ に使用されます。`platform` チャート は、カスタム リソース 定義 （CRD）で 設定 された 値 を使用して W&B Platform を デプロイ するために使用されます。

## ステップ 4：Helm リポジトリ の 設定

次に、Helm リポジトリ を 設定 して、内部 リポジトリ から W&B Helm チャート をプルします。次の コマンド を 実行 して、Helm リポジトリ を追加および更新します。

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## ステップ 5：Kubernetes operator の インストール

コントローラ マネージャー としても知られる W&B Kubernetes operator は、W&B platform コンポーネント の管理を担当します。エアギャップされた 環境 に インストール するには、
内部 コンテナ レジストリ を使用するように 設定 する必要があります。

これを行うには、内部 コンテナ レジストリ を使用するようにデフォルトの イメージ 設定 をオーバーライドし、予期される デプロイ タイプ を示すために キー `airgapped: true` を 設定 する必要があります。以下に示すように、`values.yaml` ファイル を更新します。

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

タグ を、内部 レジストリ で利用可能な バージョン に置き換えます。

operator と CRD を インストール します。
```bash
helm upgrade --install operator wandb/operator -n wandb --create-namespace -f values.yaml
```

サポートされている 値 の詳細については、[Kubernetes operator GitHub リポジトリ](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)を参照してください。

## ステップ 6：W&B カスタム リソース の 設定

W&B Kubernetes operator を インストール した後、内部 Helm リポジトリ と コンテナ レジストリ を指すようにカスタム リソース （CR）を 設定 する必要があります。

この 設定 により、Kubernetes operator が W&B platform の必要な コンポーネント を デプロイ する際に、内部 レジストリ と リポジトリ が確実に使用されるようになります。

この例の CR を `wandb.yaml` という名前の新しい ファイル にコピーします。

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

W&B platform を デプロイ するために、Kubernetes Operator は CR からの 値 を使用して、内部 リポジトリ から `operator-wandb` Helm チャート を 設定 します。

全ての タグ / バージョン を、内部 レジストリ で利用可能な バージョン に置き換えます。

上記の 設定 ファイル の 作成 に関する詳細については、[こちら]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#configuration-reference-for-wb-server" lang="ja" >}})をご覧ください。

## ステップ 7：W&B platform の デプロイ

Kubernetes operator と CR が 設定 されたので、`wandb.yaml` 設定 を 適用 して W&B platform を デプロイ します。

```bash
kubectl apply -f wandb.yaml
```

## FAQ

デプロイ プロセス 中に、以下のよくある質問（FAQ）とトラブルシューティングの ヒント を参照してください。

### 別の ingress クラス があります。その クラス を使用できますか？
はい、`values.yaml` の ingress 設定 を変更することで、ingress クラス を 設定 できます。

### 証明書 バンドル に複数の 証明書 が含まれています。それは機能しますか？
`values.yaml` の `customCACerts` セクション で、 証明書 を複数の エントリー に分割する必要があります。

### Kubernetes operator が無人 アップデート を 適用 するのを防ぐにはどうすればよいですか。それは可能ですか？
W&B console から 自動 更新 をオフにすることができます。サポートされている バージョン についての質問は、W&B チーム にお問い合わせください。また、W&B は過去 6 か月以内に リリース された platform バージョン をサポートしていることに注意してください。W&B は 定期的なアップグレード を 推奨 しています。

### 環境 が 公開 リポジトリ に 接続 されていない場合、 デプロイ は機能しますか？
ご使用の 設定 で `airgapped` が `true` に 設定 されている場合、Kubernetes operator は内部 リソース のみを使用し、 公開 リポジトリ への 接続 を試みません。
