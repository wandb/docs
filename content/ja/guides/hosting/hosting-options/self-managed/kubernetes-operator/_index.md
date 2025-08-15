---
title: Kubernetes 上で W&B サーバーを実行する
description: Kubernetes Operator で W&B プラットフォームをデプロイする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-kubernetes-operator-_index
    parent: self-managed
url: guides/hosting/operator
weight: 2
---

## W&B Kubernetes Operator

W&B Kubernetes Operator を使うことで、Kubernetes 上での W&B Server デプロイメントの展開・管理・トラブルシュートやスケーリング作業を効率化できます。Operator は W&B インスタンスの賢いアシスタントのような存在とイメージしてください。

W&B Server のアーキテクチャーや設計は AI 開発者向けツールの強化、高性能・拡張性・管理性の向上に合わせて進化し続けています。この進化は、計算リソースサービス・関連ストレージ・それらの接続性すべてにおいて当てはまります。さまざまなデプロイメントタイプに対する継続的なアップデートと改善を進めるため、W&B では Kubernetes operator を採用しています。

{{% alert %}}
W&B は Operator を利用して AWS・GCP・Azure のパブリッククラウド上に専用クラウドインスタンスをデプロイ・管理します。
{{% /alert %}}

Kubernetes operator についての詳細は Kubernetes ドキュメント内の [Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) をご覧ください。

### アーキテクチャー変更の理由

従来、W&B アプリケーションは Kubernetes クラスター内の単一のデプロイメント・Pod、または 1 つの Docker コンテナとして展開されていました。W&B は一貫して、Database や Object Store の外部化を推奨しています。これによりアプリケーションの状態管理を分離します。

アプリケーションが成長するにつれ、モノリシックなコンテナから分散システム（マイクロサービス）への進化が不可欠に。これによりバックエンドの処理や Kubernetes ネイティブなインフラ機能の活用が容易になりました。また追加のサービスを容易にデプロイでき、W&B が必要とする機能拡張への対応もスムーズになります。

2024 年以前は、Kubernetes 関連の変更ごとに [terraform-kubernetes-wandb](https://github.com/wandb/terraform-kubernetes-wandb) Terraform モジュールを手動で更新する必要がありました。Terraform モジュールの更新はクラウドごとの互換性確保、必要な変数の設定、各種変更ごとの Terraform apply 実行などが必要です。

この作業はスケーラブルではなく、W&B サポートが顧客ごとにアップグレード手順を支援しなければなりませんでした。

この課題を解決するために、Operator を実装し、中央の [deploy.wandb.ai](https://deploy.wandb.ai) サーバーと接続して最新の仕様変更を取得・反映する方式に変更しました。ライセンスが有効である限りアップデートを自動で受信します。[Helm](https://helm.sh/) は W&B operator のデプロイ手段であり、W&B Kubernetes スタックの設定テンプレートも処理します（Helm-ception）。

### 仕組み

Operator は helm またはソースからインストールできます。詳しい手順は [charts/operator](https://github.com/wandb/helm-charts/tree/main/charts/operator) を参照してください。

インストールプロセスでは `controller-manager` というデプロイメントを作成し、`weightsandbiases.apps.wandb.com`（短縮名: `wandb`）という[カスタムリソース](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)定義を利用します。`spec` を 1 つ受け取りクラスターへ適用します:

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: weightsandbiases.apps.wandb.com
```

`controller-manager` はカスタムリソース・リリースチャンネル・ユーザー定義の config に基づいて [charts/operator-wandb](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb) をインストールします。設定仕様の階層設計により、ユーザー側で最大限の柔軟性を持った設定が可能となり、W&B からのイメージ・構成・機能や Helm のアップデートも自動反映されます。

設定仕様の階層については [configuration specification hierarchy]({{< relref path="#configuration-specification-hierarchy" lang="ja" >}})、設定内容の詳細は [configuration reference]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) を参照してください。

デプロイメントは複数の Pod から構成され、各 Pod 名は `wandb-` で始まります。

### 設定仕様の階層

設定仕様は階層モデルを採用しており、高位の設定が下位の設定を上書きします。仕組みは以下のとおりです：

- **リリースチャンネル値**: デプロイごとに W&B 側で設定されるリリースチャンネルに基づくデフォルト値。
- **ユーザー入力値**: System Console からユーザーがリリースチャンネルのデフォルト値を上書き可能です。
- **カスタムリソース値**: ユーザーが指定する最上位の設定。ここで指定した値は、それより下位（ユーザー入力・リリースチャンネル）を上書きします。詳細は [Configuration Reference]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) をご覧ください。

この階層性で柔軟かつ状況に合わせた設定を維持しつつ、運用・アップグレードも体系的に行えます。

### W&B Kubernetes Operator 利用要件

W&B Kubernetes operator で W&B をデプロイするには、以下の要件を満たしてください。

[リファレンスアーキテクチャー]({{< relref path="../ref-arch.md#infrastructure-requirements" lang="ja" >}}) も参照してください。また、[有効な W&B Server ライセンスの取得]({{< relref path="../#obtain-your-wb-server-license" lang="ja" >}})が必要です。

詳細なセルフマネージドインストール手順については [bare-metal installation guide]({{< relref path="../bare-metal.md" lang="ja" >}}) をご覧ください。

インストール方法によって、以下の要件が必要な場合があります。
* Kubectl がインストール済みで正しい Kubernetes クラスターコンテキストに設定されていること
* Helm がインストール済みであること

### エアギャップ（インターネット遮断環境）でのインストール

エアギャップ環境で W&B Kubernetes Operator をインストールする手順は [Deploy W&B in airgapped environment with Kubernetes]({{< relref path="operator-airgapped.md" lang="ja" >}}) チュートリアルをご覧ください。

## W&B Server アプリケーションのデプロイ

このセクションでは W&B Kubernetes operator を使ったさまざまなデプロイ手順を説明します。

{{% alert %}}
W&B Operator はデフォルトかつ推奨される W&B Server のインストール方法です。
{{% /alert %}}

### Helm CLI で W&B のデプロイ

W&B は Kubernetes クラスターへ Operator をデプロイできる Helm Chart を提供しています。この手順を用いると Helm CLI だけでなく ArgoCD などの継続的デリバリー ツールでもデプロイできます。上記要件が満たされていることを確認してください。

Helm CLI を使った W&B Kubernetes Operator インストール手順：

1. W&B Helm リポジトリを追加します。W&B Helm チャートは次のリポジトリにあります。
    ```shell
    helm repo add wandb https://charts.wandb.ai
    helm repo update
    ```
2. Kubernetes クラスターへ Operator をインストールします。
    ```shell
    helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
    ```
3. W&B Operator カスタムリソースを設定し、W&B Server のインストールをトリガーします。Helm の `values.yaml` ファイルでデフォルト設定を上書きするか、CRD（カスタムリソース定義）を直接フルカスタマイズします。

    - **`values.yaml` で上書き（推奨）** : 新しく `values.yaml` というファイルを作成し、[values.yaml の全仕様](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml) から必要なキーのみ含めます。例えば MySQL を設定する場合：

      {{< prism file="/operator/values_mysql.yaml" title="values.yaml">}}{{< /prism >}}
    - **CRD をフル設定** : [設定例](https://github.com/wandb/helm-charts/blob/main/charts/operator/crds/wandb.yaml) を `operator.yaml` という新規ファイルへコピーし、必要な変更を加えてください。詳細は [Configuration Reference]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) を参照。

      {{< prism file="/operator/wandb.yaml" title="operator.yaml">}}{{< /prism >}}

4. カスタマイズした設定で Operator を起動し、W&B Server アプリケーションをインストール・設定・管理します。

    - `values.yaml` で起動する場合:

        ```shell
        kubectl apply -f values.yaml
        ```
    - 完全カスタマイズ CRD で起動する場合:
      ```shell
      kubectl apply -f operator.yaml
      ```

    デプロイメントが完了するまで数分かかります。

5. Web UI でインストールを検証するには最初の管理者ユーザーを作成し、 [検証手順]({{< relref path="#verify-the-installation" lang="ja" >}}) を参照してください。

### Helm Terraform モジュールで W&B のデプロイ

この方法では Terraform の Infrastructure as Code 特性を通じて、固有要件に合わせたカスタムデプロイを構築できます。公式の W&B Helm ベース Terraform モジュールは [こちら](https://registry.terraform.io/modules/wandb/wandb/helm/latest) です。

次のコードはプロダクション向け基本構成例です。

```hcl
module "wandb" {
  source  = "wandb/wandb/helm"

  spec = {
    values = {
      global = {
        host    = "https://<HOST_URI>"
        license = "eyJhbGnUzaH...j9ZieKQ2x5GGfw"

        bucket = {
          <details depend on the provider>
        }

        mysql = {
          <redacted>
        }
      }

      ingress = {
        annotations = {
          "a" = "b"
          "x" = "y"
        }
      }
    }
  }
}
```

設定項目は [Configuration Reference]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}})で解説しているものと同一ですが、書式は HashiCorp Configuration Language (HCL) に従います。Terraform モジュールが W&B のカスタムリソース定義（CRD）を作成します。

W&B&Biases 自体が Helm Terraform モジュールを使って “専用クラウド” インストールを行う例は以下を参照ください：
- [AWS](https://github.com/wandb/terraform-aws-wandb/blob/45e1d746f53e78e73e68f911a1f8cad5408e74b6/main.tf#L225)
- [Azure](https://github.com/wandb/terraform-azurerm-wandb/blob/170e03136b6b6fc758102d59dacda99768854045/main.tf#L155)
- [GCP](https://github.com/wandb/terraform-google-wandb/blob/49ddc3383df4cefc04337a2ae784f57ce2a2c699/main.tf#L189)

### W&B Cloud Terraform モジュールでのデプロイ

W&B は AWS・GCP・Azure 向け Terraform モジュールを提供しています。これらのモジュールは Kubernetes クラスターやロードバランサー、MySQL データベース等のインフラも含めて、W&B Server アプリケーションまですべてデプロイします。W&B Kubernetes Operator はこれらの公式クラウド向けモジュールに標準で組み込まれており、対応バージョンは以下の通りです。

| Terraform Registry                                                  | Source Code                                      | Version |
| ------------------------------------------------------------------- | ------------------------------------------------ | ------- |
| [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest) | https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| [Azure](https://github.com/wandb/terraform-azurerm-wandb)           | https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| [GCP](https://github.com/wandb/terraform-google-wandb)              | https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

この統合によりインスタンス用 Operator の準備がシンプルに行え、クラウド環境でもスムーズな W&B Server のデプロイ・管理が可能です。

モジュールの使い方詳細は [self-managed installations section]({{< relref path="../#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}}) を参照ください。

### インストールの検証手順

インストールの検証には [W&B CLI]({{< relref path="/ref/cli/" lang="ja" >}}) を使うことを推奨します。`verify` コマンドは複数の項目をテストし、全ての構成・コンポーネントの検証を実行します。

{{% alert %}}
この手順では最初の管理者ユーザーがブラウザで作成されていることを前提とします。
{{% /alert %}}

検証手順は以下の通りです。

1. W&B CLI をインストール
    ```shell
    pip install wandb
    ```
2. W&B へログイン
    ```shell
    wandb login --host=https://YOUR_DNS_DOMAIN
    ```

    例:
    ```shell
    wandb login --host=https://wandb.company-name.com
    ```

3. インストールを検証
    ```shell
    wandb verify
    ```

すべて正常に動作している場合、次のような出力となります。

```console
Default host selected:  https://wandb.company-name.com
Find detailed logs for this test at: /var/folders/pn/b3g3gnc11_sbsykqkm3tx5rh0000gp/T/tmpdtdjbxua/wandb
Checking if logged in...................................................✅
Checking signed URL upload..............................................✅
Checking ability to send large payloads through proxy...................✅
Checking requests to base url...........................................✅
Checking requests made over signed URLs.................................✅
Checking CORs configuration of the bucket...............................✅
Checking wandb package version is up to date............................✅
Checking logged metrics, saving and downloading a file..................✅
Checking artifact save and download workflows...........................✅
``` 

## W&B 管理コンソールにアクセスする

W&B Kubernetes operator には管理コンソールが付属しています。`${HOST_URI}/console` でアクセス可能（例 `https://wandb.company-name.com/console`）。

管理コンソールのログイン方法は2通りあります。

{{< tabpane text=true >}}
{{% tab header="オプション 1（推奨）" value="option1" %}}
1. ブラウザで W&B アプリケーションにアクセスし、ログインします。ログイン先は `${HOST_URI}/` です（例 `https://wandb.company-name.com/`）。
2. コンソールにアクセスします。右上のアイコンをクリックし、**System console** を選択します。管理者権限ユーザーのみ **System console** エントリーが表示されます。

    {{< img src="/images/hosting/access_system_console_via_main_app.png" alt="System console access" >}}
{{% /tab %}}

{{% tab header="オプション 2" value="option2"%}}
{{% alert %}}
オプション 1 でアクセスできない場合のみ、以下の手順でのコンソールアクセスを推奨します。
{{% /alert %}}

1. ブラウザでコンソールアプリケーションに直接アクセスします。説明した URL を開くとログイン画面に遷移します。
    {{< img src="/images/hosting/access_system_console_directly.png" alt="Direct system console access" >}}
2. インストール時に生成される Kubernetes シークレットからパスワードを取得します。
    ```shell
    kubectl get secret wandb-password -o jsonpath='{.data.password}' | base64 -d
    ```
    パスワードをコピーしてください。
3. コンソールにログインします。コピーしたパスワードを貼り付けて **Login** をクリックします。
{{% /tab %}}
{{< /tabpane >}}

## W&B Kubernetes operator のアップデート

このセクションでは W&B Kubernetes operator のアップデート方法を説明します。

{{% alert %}}
* W&B Kubernetes operator のアップデートは W&B server アプリケーション自体のアップデートにはなりません。
* もし以前に Operator を使っていない Helm チャートを使っていた場合は、この手順の前に [こちら]({{< relref path="#migrate-self-managed-instances-to-wb-operator" lang="ja" >}}) を参照してください。
{{% /alert %}}

下記のコードスニペットをターミナルで実行してください。

1. まずリポジトリを [`helm repo update`](https://helm.sh/docs/helm/helm_repo_update/) で最新化します。
    ```shell
    helm repo update
    ```

2. 次に [`helm upgrade`](https://helm.sh/docs/helm/helm_upgrade/) で Helm チャートをアップデートします。
    ```shell
    helm upgrade operator wandb/operator -n wandb-cr --reuse-values
    ```

## W&B Server アプリケーションのアップデート

W&B Kubernetes Operator を使っている場合、W&B Server アプリケーションは自動的にアップデートされるため手動更新は不要です。

Operator が新バージョンのリリース時に W&B Server を自動更新します。

## 自己管理インスタンスから W&B Operator への移行

このセクションでは自己管理型の W&B Server インストールから Operator を使った運用へ移行する方法を説明します。移行方法は、W&B Server をどのようにインストールしていたかによります：

{{% alert %}}
W&B Operator は W&B Server のデフォルトかつ推奨のインストール方法です。ご不明な点は [カスタマーサポート](mailto:support@wandb.com) または W&B チームにお問い合わせください。
{{% /alert %}}

- 公式 W&B Cloud Terraform Module を使用していた場合は各ドキュメントの該当手順に従ってください：
  - [AWS]({{< relref path="#migrate-to-operator-based-aws-terraform-modules" lang="ja" >}})
  - [GCP]({{< relref path="#migrate-to-operator-based-gcp-terraform-modules" lang="ja" >}})
  - [Azure]({{< relref path="#migrate-to-operator-based-azure-terraform-modules" lang="ja" >}})
- [W&B Non-Operator Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/wandb) を使っていた場合は [こちら]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ja" >}}) に進んでください。
- [W&B Non-Operator Helm chart with Terraform](https://registry.terraform.io/modules/wandb/wandb/kubernetes/latest) を使っていた場合は [こちら]({{< relref path="#migrate-to-operator-based-terraform-helm-chart" lang="ja" >}}) に進んでください。
- マニフェストで Kubernetes 資源を作成していた場合は [こちら]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ja" >}}) へ。

### Operator ベース AWS Terraform Module への移行

移行手順の詳細は [こちら]({{< relref path="../install-on-public-cloud/aws-tf.md#migrate-to-operator-based-aws-terraform-modules" lang="ja" >}}) をご覧ください。

### Operator ベース GCP Terraform Module への移行

ご質問やご相談は [カスタマーサポート](mailto:support@wandb.com) または W&B チームまで。

### Operator ベース Azure Terraform Module への移行

ご質問やご相談は [カスタマーサポート](mailto:support@wandb.com) または W&B チームまで。

### Operator ベース Helm chart への移行

下記手順で Operator ベース Helm チャートへ移行してください。

1. 現行 W&B の設定を取得します。Non-Operator ベース Helm チャートでデプロイしていた場合は次のコマンドで値をエクスポートします。
    ```shell
    helm get values wandb
    ```
    Kubernetes マニフェストで展開していた場合は以下のコマンドです。
    ```shell
    kubectl get deployment wandb -o yaml
    ```
    これで必要な全設定値が揃いました。

2. `operator.yaml` というファイルを作成します。[Configuration Reference]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) の形式に従い、ステップ1の値を使ってください。

3. 現在のデプロイメントを Pod 数 0 にスケールします。これで現在のデプロイメントを停止します。
    ```shell
    kubectl scale --replicas=0 deployment wandb
    ```
4. Helm チャートリポジトリをアップデートします。
    ```shell
    helm repo update
    ```
5. 新しい Helm チャートをインストールします。
    ```shell
    helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
    ```
6. 新しい Helm チャートを設定し、W&B アプリケーションのデプロイメントをトリガーします。
    ```shell
    kubectl apply -f operator.yaml
    ```
    デプロイメント完了まで数分かかります。

7. インストールを検証します。[Verify the installation]({{< relref path="#verify-the-installation" lang="ja" >}}) の手順で動作確認をしてください。

8. 古いインストールを削除します。古い Helm チャートをアンインストールするか、マニフェストで作成されたリソースを削除します。

### Operator ベース Terraform Helm chart への移行

下記手順で Operator ベース Helm チャートへの移行を行ってください。

1. Terraform 設定の準備：古いデプロイメントの Terraform コードを [こちら]({{< relref path="#deploy-wb-with-helm-terraform-module" lang="ja" >}}) の記法に置き換え、以前と同様の変数を設定します。.tfvars ファイルは変更せずそのままにします。
2. Terraform 実行：`terraform init`、`terraform plan`、`terraform apply` を順に実行してください。
3. インストールの検証：[Verify the installation]({{< relref path="#verify-the-installation" lang="ja" >}}) の手順で動作確認します。
4. 古いインストールを削除します。古い Helm チャートのアンインストールまたはマニフェスト作成リソースの削除を行います。

## W&B Server の設定リファレンス

このセクションでは W&B Server アプリケーションの設定項目について説明します。アプリケーションは [WeightsAndBiases]({{< relref path="#how-it-works" lang="ja" >}}) というカスタムリソース定義で設定内容を受け取ります。設定の一部は下記設定ファイルで露出し、特定項目は環境変数としてセットする必要があります。

環境変数には [基本]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}) & [高度な設定]({{< relref path="/guides/hosting/iam/advanced_env_vars/" lang="ja" >}}) の2つがあります。設定項目が Helm Chart 経由で指定できない場合のみ、環境変数をご利用ください。

プロダクション用の W&B Server アプリケーション設定ファイルは以下の内容が必須です。この YAML ファイルは W&B デプロイメントの望ましい状態（バージョン・環境変数・外部リソース・DB 等）を定義します。

```yaml
apiVersion: apps.wandb.com/v1
kind: WeightsAndBiases
metadata:
  labels:
    app.kubernetes.io/name: weightsandbiases
    app.kubernetes.io/instance: wandb
  name: wandb
  namespace: default
spec:
  values:
    global:
      host: https://<HOST_URI>
      license: eyJhbGnUzaH...j9ZieKQ2x5GGfw
      bucket:
        <details depend on the provider>
      mysql:
        <redacted>
    ingress:
      annotations:
        <redacted>
```

すべての値の詳細仕様は [W&B Helm repository](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml) を参照し、上書きが必要な項目だけ修正してください。

### 完全な例

GCP Kubernetes × GCP Ingress × GCS（GCPオブジェクトストレージ）の例：

```yaml
apiVersion: apps.wandb.com/v1
kind: WeightsAndBiases
metadata:
  labels:
    app.kubernetes.io/name: weightsandbiases
    app.kubernetes.io/instance: wandb
  name: wandb
  namespace: default
spec:
  values:
    global:
      host: https://abc-wandb.sandbox-gcp.wandb.ml
      bucket:
        name: abc-wandb-moving-pipefish
        provider: gcs
      mysql:
        database: wandb_local
        host: 10.218.0.2
        name: wandb_local
        password: 8wtX6cJHizAZvYScjDzZcUarK4zZGjpV
        port: 3306
        user: wandb
      license: eyJhbGnUzaHgyQjQyQWhEU3...ZieKQ2x5GGfw
    ingress:
      annotations:
        ingress.gcp.kubernetes.io/pre-shared-cert: abc-wandb-cert-creative-puma
        kubernetes.io/ingress.class: gce
        kubernetes.io/ingress.global-static-ip-name: abc-wandb-operator-address
```

### ホスト指定

```yaml
 # FQDN（プロトコル込み）を指定してください
global:
  # 例：ご自身のホスト名に置き換えてください
  host: https://wandb.example.com
```

### オブジェクトストレージ（バケット）

**AWS**
```yaml
global:
  bucket:
    provider: "s3"
    name: ""
    kmsKey: ""
    region: ""
```

**GCP**
```yaml
global:
  bucket:
    provider: "gcs"
    name: ""
```

**Azure**
```yaml
global:
  bucket:
    provider: "az"
    name: ""
    secretKey: ""
```

**その他プロバイダー（Minio, Ceph など）**

S3互換ストレージの場合、以下のように設定します。
```yaml
global:
  bucket:
    # 例：各自で置き換え
    provider: s3
    name: storage.example.com
    kmsKey: null
    path: wandb
    region: default
    accessKey: 5WOA500...P5DK7I
    secretKey: HDKYe4Q...JAp1YyjysnX
```

AWS 以外の S3 互換ストレージでは `kmsKey` を `null` にしてください。

`accessKey` と `secretKey` をシークレットから参照する場合：
```yaml
global:
  bucket:
    # 例：各自で置き換え
    provider: s3
    name: storage.example.com
    kmsKey: null
    path: wandb
    region: default
    secret:
      secretName: bucket-secret
      accessKeyName: ACCESS_KEY
      secretKeyName: SECRET_KEY
```

### MySQL 設定

```yaml
global:
   mysql:
     # 例：各自で置き換え
     host: db.example.com
     port: 3306
     database: wandb_local
     user: wandb
     password: 8wtX6cJH...ZcUarK4zZGjpV 
```

`password` をシークレットから参照するには：
```yaml
global:
   mysql:
     # 例
     host: db.example.com
     port: 3306
     database: wandb_local
     user: wandb
     passwordSecret:
       name: database-secret
       passwordKey: MYSQL_WANDB_PASSWORD
```

### ライセンス

```yaml
global:
  # 例：自身のライセンスに置き換え
  license: eyJhbGnUzaHgyQjQy...VFnPS_KETXg1hi
```

シークレットから `license` を参照する場合：
```yaml
global:
  licenseSecret:
    name: license-secret
    key: CUSTOMER_WANDB_LICENSE
```

### Ingress

Ingress クラスの特定方法はこの[FAQ]({{< relref path="#how-to-identify-the-kubernetes-ingress-class" lang="ja" >}})を参照ください。

**TLS なし**

```yaml
global:
# 注意：ingress は global と同じ階層です
ingress:
  class: ""
```

**TLS あり**

証明書を格納するシークレットを作成します。

```console
kubectl create secret tls wandb-ingress-tls --key wandb-ingress-tls.key --cert wandb-ingress-tls.crt
```

このシークレットを設定内で参照します。
```yaml
global:
# 注意：ingress は global と同じ階層です
ingress:
  class: ""
  annotations:
    {}
    # kubernetes.io/ingress.class: nginx
    # kubernetes.io/tls-acme: "true"
  tls: 
    - secretName: wandb-ingress-tls
      hosts:
        - <HOST_URI>
```

Nginx 利用時は次のアノテーション追加が必要な場合があります。

```
ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: 64m
```

### カスタム Kubernetes ServiceAccount の指定

各 W&B Pod に独自の Kubernetes ServiceAccount を指定できます。

次の例はデプロイと同時に指定名の ServiceAccount を作成します。

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: true

parquet:
  serviceAccount:
    name: custom-service-account
    create: true

global:
  ...
```
"app" と "parquet" サブシステムが指定したサービスアカウントで実行されます。他サブシステムはデフォルトアカウントで動作します。

すでにクラスターに ServiceAccount が存在する場合は `create: false` を指定してください。

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: false

parquet:
  serviceAccount:
    name: custom-service-account
    create: false
    
global:
  ...
```

app/parquet/console などサブシステムごとに指定可能です。

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: true

console:
  serviceAccount:
    name: custom-service-account
    create: true

global:
  ...
```

サブシステムごとに異なるアカウントも指定可能です。

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: false

console:
  serviceAccount:
    name: another-custom-service-account
    create: true

global:
  ...
```

### 外部 Redis の利用

```yaml
redis:
  install: false

global:
  redis:
    host: ""
    port: 6379
    password: ""
    parameters: {}
    caCert: ""
```

`password` をシークレットで管理したい場合：

```console
kubectl create secret generic redis-secret --from-literal=redis-password=supersecret
```

設定では以下のように参照します：
```yaml
redis:
  install: false

global:
  redis:
    host: redis.example
    port: 9001
    auth:
      enabled: true
      secret: redis-secret
      key: redis-password
```

### LDAP

**TLS なし**
```yaml
global:
  ldap:
    enabled: true
    # LDAP サーバーアドレス (ldap:// / ldaps://含む)
    host:
    # ユーザー検索用 BaseDN
    baseDN:
    # バインドユーザー 
    bindDN:
    # バインドパスワード用シークレット名・キー（匿名バインドでなければ）
    bindPW:
    # email/groupID 属性名のカンマ区切り
    attributes:
    # グループ許可リスト
    groupAllowList:
    # LDAP TLS有効化
    tls: false
```

**TLS あり**

LDAP TLS証明書の config map を事前に用意します。

config map の作成例：

```console
kubectl create configmap ldap-tls-cert --from-file=certificate.crt
```

YAML では下記のように参照します：

```yaml
global:
  ldap:
    enabled: true
    # LDAP サーバーアドレス (ldap:// / ldaps://含む)
    host:
    # ユーザー検索用 BaseDN
    baseDN:
    # バインドユーザー 
    bindDN:
    # バインドパスワード用シークレット名・キー（匿名バインドでなければ）
    bindPW:
    # email/groupID 属性名のカンマ区切り
    attributes:
    # グループ許可リスト
    groupAllowList:
    # LDAP TLS有効化
    tls: true
    # CA証明書用 ConfigMap名とキー
    tlsCert:
      configMap:
        name: "ldap-tls-cert"
        key: "certificate.crt"
```

### OIDC SSO

```yaml
global: 
  auth:
    sessionLengthHours: 720
    oidc:
      clientId: ""
      secret: ""
      # IdP によっては必須
      authMethod: ""
      issuer: ""
```

`authMethod` は任意です。

### SMTP

```yaml
global:
  email:
    smtp:
      host: ""
      port: 587
      user: ""
      password: ""
```

### 環境変数

```yaml
global:
  extraEnv:
    GLOBAL_ENV: "example"
```

### カスタム認証局証明書

`customCACerts` は複数の証明書をリストとして指定できます。この設定内容は W&B Server アプリケーションにのみ適用されます。

```yaml
global:
  customCACerts:
  - |
    -----BEGIN CERTIFICATE-----
    MIIBnDCCAUKgAwIBAg.....................fucMwCgYIKoZIzj0EAwIwLDEQ
    MA4GA1UEChMHSG9tZU.....................tZUxhYiBSb290IENBMB4XDTI0
    MDQwMTA4MjgzMFoXDT.....................oNWYggsMo8O+0mWLYMAoGCCqG
    SM49BAMCA0gAMEUCIQ.....................hwuJgyQRaqMI149div72V2QIg
    P5GD+5I+02yEp58Cwxd5Bj2CvyQwTjTO4hiVl1Xd0M0=
    -----END CERTIFICATE-----
  - |
    -----BEGIN CERTIFICATE-----
    MIIBxTCCAWugAwIB.......................qaJcwCgYIKoZIzj0EAwIwLDEQ
    MA4GA1UEChMHSG9t.......................tZUxhYiBSb290IENBMB4XDTI0
    MDQwMTA4MjgzMVoX.......................UK+moK4nZYvpNpqfvz/7m5wKU
    SAAwRQIhAIzXZMW4.......................E8UFqsCcILdXjAiA7iTluM0IU
    aIgJYVqKxXt25blH/VyBRzvNhViesfkNUQ==
    -----END CERTIFICATE-----
```

ConfigMap で管理する場合は：

```yaml
global:
  caCertsConfigMap: custom-ca-certs
```

ConfigMap の例：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: custom-ca-certs
data:
  ca-cert1.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
  ca-cert2.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
```

{{% alert %}}
ConfigMap 利用時、各キーは `.crt` で終わる必要があります（例：`my-cert.crt`、`ca-cert1.crt`）。この命名則が `update-ca-certificates` による処理に必須です。
{{% /alert %}}

### カスタムセキュリティコンテキスト

各 W&B コンポーネントで独自のセキュリティコンテキスト設定が可能です。

```yaml
pod:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1001
    runAsGroup: 0
    fsGroup: 1001
    fsGroupChangePolicy: Always
    seccompProfile:
      type: RuntimeDefault
container:
  securityContext:
    capabilities:
      drop:
        - ALL
    readOnlyRootFilesystem: false
    allowPrivilegeEscalation: false 
```

{{% alert %}}
`runAsGroup:` の有効値は `0` のみ。他はエラーになります。
{{% /alert %}}

例えばアプリケーション Pod の場合、設定へ `app` セクションを追加してください：

```yaml
global:
  ...
app:
  pod:
    securityContext:
      runAsNonRoot: true
      runAsUser: 1001
      runAsGroup: 0
      fsGroup: 1001
      fsGroupChangePolicy: Always
      seccompProfile:
        type: RuntimeDefault
  container:
    securityContext:
      capabilities:
        drop:
          - ALL
      readOnlyRootFilesystem: false
      allowPrivilegeEscalation: false 
```

同様に `console`、`weave`、`weave-trace`、`parquet` にも適用できます。

## W&B Operator の設定リファレンス

このセクションでは W&B Kubernetes operator（`wandb-controller-manager`）の設定項目について解説します。YAML ファイルで設定内容を渡します。

通常は W&B Kubernetes operator に設定ファイルは不要です。必要な場合のみ用意してください（例：独自認証局証明書の指定やエアギャップ環境デプロイなど）。

設定内容の全リストは [Helm repository](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml) をご参照ください。

### カスタム認証局（Custom CA）

`customCACerts` では複数の証明書をリストで指定できます。ここで指定した証明書は W&B Kubernetes operator（`wandb-controller-manager`）のみに適用されます。

```yaml
customCACerts:
- |
  -----BEGIN CERTIFICATE-----
  MIIBnDCCAUKgAwIBAg.....................fucMwCgYIKoZIzj0EAwIwLDEQ
  MA4GA1UEChMHSG9tZU.....................tZUxhYiBSb290IENBMB4XDTI0
  MDQwMTA4MjgzMFoXDT.....................oNWYggsMo8O+0mWLYMAoGCCqG
  SM49BAMCA0gAMEUCIQ.....................hwuJgyQRaqMI149div72V2QIg
  P5GD+5I+02yEp58Cwxd5Bj2CvyQwTjTO4hiVl1Xd0M0=
  -----END CERTIFICATE-----
- |
  -----BEGIN CERTIFICATE-----
  MIIBxTCCAWugAwIB.......................qaJcwCgYIKoZIzj0EAwIwLDEQ
  MA4GA1UEChMHSG9t.......................tZUxhYiBSb290IENBMB4XDTI0
  MDQwMTA4MjgzMVoX.......................UK+moK4nZYvpNpqfvz/7m5wKU
  SAAwRQIhAIzXZMW4.......................E8UFqsCcILdXjAiA7iTluM0IU
  aIgJYVqKxXt25blH/VyBRzvNhViesfkNUQ==
  -----END CERTIFICATE-----
```

ConfigMap で管理する場合：

```yaml
caCertsConfigMap: custom-ca-certs
```

ConfigMap の書式例：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: custom-ca-certs
data:
  ca-cert1.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
  ca-cert2.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
```

{{% alert %}}
ConfigMap 利用時、各キーは必ず `.crt` で終わるようにしてください（例：`my-cert.crt` または `ca-cert1.crt`）。このルールが `update-ca-certificates` によるシステム CA への追加に必要です。
{{% /alert %}}

## FAQ

### 各 Pod の役割は？

* **`wandb-app`**: W&B の中核。GraphQL API とフロントエンドアプリを持ち、主な機能を提供します。
* **`wandb-console`**: 管理コンソール用 Pod。`/console` でアクセスします。
* **`wandb-otel`**: OpenTelemetry エージェント。Kubernetes レイヤーの各資源からメトリクス/ログを収集し管理コンソールで表示します。
* **`wandb-prometheus`**: Prometheus サーバ。各コンポーネントのメトリクス取得・管理コンソール表示に利用されます。
* **`wandb-parquet`**: `wandb-app` とは別のバックエンドマイクロサービス。データベースのデータを Parquet 形式でオブジェクトストレージにエクスポートします。
* **`wandb-weave`**: UI でクエリテーブルをロードしたりアプリのコア機能を支援するバックエンドマイクロサービスです。
* **`wandb-weave-trace`**: LLM アプリのトラッキング・実験・評価・デプロイ等に利用されるフレームワーク。`wandb-app` Pod 経由で利用します。

### W&B Operator コンソールパスワードの取得方法

[W&B Kubernetes Operator 管理コンソールへのアクセス方法]({{< relref path="#access-the-wb-management-console" lang="ja" >}}) を参照してください。

### Ingress が使えない場合の W&B Operator コンソールアクセス

Kubernetes クラスターへアクセスできるホスト上で以下のコマンドを実行：

```console
kubectl port-forward svc/wandb-console 8082
```

ブラウザから `https://localhost:8082/` でコンソールへアクセス可能。

パスワード取得法は [管理コンソールへのアクセス方法]({{< relref path="#access-the-wb-management-console" lang="ja" >}})「オプション 2」を参照。

### W&B Server のログを見る方法

アプリケーション Pod の名前は **wandb-app-xxx** です。

```console
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

### Kubernetes Ingress クラスの特定方法

クラスターにインストールされている IngressClass を調べるには以下を実行：

```console
kubectl get ingressclass
```