---
title: Kubernetes 上で W&B サーバー を実行する
description: Kubernetes Operator で W&B プラットフォームをデプロイする
menu:
  default:
    identifier: kubernetes-operator
    parent: self-managed
weight: 2
url: guides/hosting/operator
---

## W&B Kubernetes Operator

W&B Kubernetes Operator を使うことで、Kubernetes 上での W&B Server のデプロイ、管理、トラブルシューティング、スケーリングが簡単になります。このオペレーターは、W&B インスタンス用のスマートアシスタントのような存在です。

W&B Server のアーキテクチャーと設計は、AI 開発者向けツールの機能拡充や、高性能化・スケーラビリティ向上、管理の簡素化のために絶えず進化しています。この進化はコンピュートサービス、関連ストレージ、そしてそれらの接続性にも及びます。継続的なアップデートやデプロイメントタイプ間での改善を容易にするため、W&B では Kubernetes オペレーターを採用しています。

{{% alert %}}
W&B は AWS、GCP、Azure のパブリッククラウド上の専用クラウドインスタンスのデプロイおよび管理にオペレーターを使用しています。
{{% /alert %}}

Kubernetes オペレーターについての詳細は、Kubernetes ドキュメントの [Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) を参照してください。

### アーキテクチャー変更の理由
W&B アプリケーションはこれまで、Kubernetes クラスター内の 1 つのデプロイメント/Pod や、単一の dockerコンテナ としてデプロイされてきました。W&B では引き続き、Database と Object Store を外部化することを推奨しています。Database・Object Store を外部化することでアプリケーションの状態を分離できます。

アプリケーションの成長に伴い、モノリシックなコンテナから分散システム (マイクロサービス) への進化が必然となりました。この変更により、バックエンドの処理ロジックが効率的に処理され Kubernetes のインフラストラクチャー機能を活かせるようになります。分散システムは W&B が必要とする新機能サービスの追加デプロイにも不可欠です。

2024 年以前は、Kubernetes 関連の変更を [terraform-kubernetes-wandb](https://github.com/wandb/terraform-kubernetes-wandb) Terraform モジュールを手動で更新する必要がありました。モジュールを最新化しクラウド各社での互換を保ち、必要な変数を調整・適用しなければなりませんでした。

この運用は非スケーラブルで、Terraform モジュールのアップグレード時は毎回 W&B サポートがユーザーごとに支援していました。

解決策として、中央の [deploy.wandb.ai](https://deploy.wandb.ai) サーバーと連携し、リリースチャンネルごとの最新スペック変更を取得・適用するオペレーターを実装しました。ライセンスが有効な限り更新を受信できます。デプロイメントには [Helm](https://helm.sh/) を活用し、W&B オペレーターや Kubernetes スタックの設定テンプレートも Helm で管理しています（ヘルム-セプション）。

### 仕組み
オペレーターは helm もしくはソースからインストールできます。詳しくは [charts/operator](https://github.com/wandb/helm-charts/tree/main/charts/operator) をご覧ください。

インストールは `controller-manager` というデプロイメントを作成し、`weightsandbiases.apps.wandb.com`（短縮名: `wandb`）という [カスタムリソース](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)定義を利用します。ここに1つの `spec` を指定し、クラスターへ適用します:

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: weightsandbiases.apps.wandb.com
```

`controller-manager` はカスタムリソースの spec・リリースチャンネル・ユーザー定義の config を基に、[charts/operator-wandb](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb) をインストールします。この設定仕様の階層構造により、ユーザーの高い柔軟性が確保され、W&B 側は新しいイメージや設定・機能・Helm のアップデート配信を自動化できます。

設定仕様の階層構造や詳細な参照は、[configuration specification hierarchy]({{< relref "#configuration-specification-hierarchy" >}}) と [configuration reference]({{< relref "#configuration-reference-for-wb-operator" >}}) をご覧ください。

デプロイでは各サービスごとに複数の Pod で構成され、各 Pod 名は `wandb-` で始まります。

### 設定仕様の階層構造
設定仕様は上位レベルの仕様が下位レベルの仕様を上書きする階層モデルです。構造は以下の通りです：

- **Release Channel Values**: ベース設定。W&B がリリースチャンネルで指定したデフォルト値が利用されます。
- **User Input Values**: System Console からユーザーが Release Channel Spec のデフォルト設定を上書きできます。
- **Custom Resource Values**: ユーザーがカスタムリソースで指定する設定です。ここに書かれた値は User Input および Release Channel 設定の両方を上書きします。詳しいオプションは [Configuration Reference]({{< relref "#configuration-reference-for-wb-operator" >}}) を参照してください。

この階層モデルにより、用途に応じて柔軟かつ管理しやすい運用・アップグレードが可能です。

### W&B Kubernetes Operator 利用要件
W&B Kubernetes operator で W&B をデプロイするには、以下の要件を満たしてください：

[リファレンスアーキテクチャー]({{< relref "../ref-arch.md#infrastructure-requirements" >}}) を参照の上、[有効な W&B Server ライセンス入手]({{< relref "../#obtain-your-wb-server-license" >}}) を行ってください。

セルフマネージドインストールの詳細は [bare-metal installation guide]({{< relref "../bare-metal.md" >}}) をご覧ください。

インストール方法に応じ、以下が必要な場合があります：
* kubectl がインストールされ、適切な Kubernetes クラスターコンテキストで設定されていること
* Helm のインストール

### エアギャップ環境でのインストール
エアギャップ環境で W&B Kubernetes Operator をインストールする手順は、[Deploy W&B in airgapped environment with Kubernetes]({{< relref "operator-airgapped.md" >}}) チュートリアルを参照してください。

## W&B Server アプリケーションのデプロイ
このセクションでは W&B Kubernetes operator の様々なデプロイ手順を説明します。
{{% alert %}}
W&B Operator は W&B Server のデフォルト推奨インストール方式です。
{{% /alert %}}

### Helm CLI で W&B をデプロイ
W&B は Kubernetes クラスターへの W&B Kubernetes operator デプロイ用 Helm Chart を提供しています。この方法なら Helm CLI や ArgoCD のような継続的デリバリーツールで W&B Server を導入できます。上記要件が揃っていることを確認してください。

手順は以下の通りです：

1. W&B Helm リポジトリを追加します。Helm チャートは W&B Helm リポジトリにあります:
    ```shell
    helm repo add wandb https://charts.wandb.ai
    helm repo update
    ```
2. Kubernetes クラスターへ Operator をインストールします:
    ```shell
    helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
    ```
3. W&B operator のカスタムリソースで W&B Server インストールを開始します。Helm の `values.yaml` ファイルを使ってデフォルト設定を上書き、または CRD を直接カスタマイズします。

    - **`values.yaml` オーバーライド**（推奨）：[`values.yaml` の仕様](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml) から上書きしたいキーだけ記載した `values.yaml` ファイルを作成します（例: MySQL の設定）:

      {{< prism file="/operator/values_mysql.yaml" title="values.yaml">}}{{< /prism >}}
    - **CRD 直接編集**：この [設定例](https://github.com/wandb/helm-charts/blob/main/charts/operator/crds/wandb.yaml) を `operator.yaml` として保存し、必要事項を修正。詳細は [Configuration Reference]({{< relref "#configuration-reference-for-wb-operator" >}}) を参照。

      {{< prism file="/operator/wandb.yaml" title="operator.yaml">}}{{< /prism >}}

4. カスタム設定で Operator を起動し、W&B Server アプリケーションのインストール・設定・管理を行います。

    - `values.yaml` オーバーライドで Operator を開始するには:

        ```shell
        kubectl apply -f values.yaml
        ```
    - CRD を直接記述した場合:
      ```shell
      kubectl apply -f operator.yaml
      ```

    デプロイ完了まで数分待ちます。

5. Web UI でインストールを確認します。まず管理者ユーザーを作成し、[Verify the installation]({{< relref "#verify-the-installation" >}}) の手順に従って検証してください。

### Helm Terraform Module で W&B をデプロイ

この方法なら、特定の要件に合わせたカスタムデプロイが可能です。Terraform の IaC（インフラストラクチャー as Code）による一貫性と再現性も得られます。公式の W&B Helm-based Terraform Module は [こちら](https://registry.terraform.io/modules/wandb/wandb/helm/latest) です。

次のコード例は本番環境レベルのデプロイに必要な主な設定を網羅しています。

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

構成オプションは [Configuration Reference]({{< relref "#configuration-reference-for-wb-operator" >}}) と同じ内容ですが、構文は HashiCorp Configuration Language (HCL) に従う必要があります。Terraform モジュールは W&B のカスタムリソース定義（CRD）も作成します。

W&B が顧客の “専用クラウド” インストールに Helm Terraform module をどう使っているかは、下記リンクをご覧ください：
- [AWS](https://github.com/wandb/terraform-aws-wandb/blob/45e1d746f53e78e73e68f911a1f8cad5408e74b6/main.tf#L225)
- [Azure](https://github.com/wandb/terraform-azurerm-wandb/blob/170e03136b6b6fc758102d59dacda99768854045/main.tf#L155)
- [GCP](https://github.com/wandb/terraform-google-wandb/blob/49ddc3383df4cefc04337a2ae784f57ce2a2c699/main.tf#L189)

### W&B Cloud Terraform modules で W&B をデプロイ

W&B は AWS、GCP、Azure 向けの Terraform Modules を提供しています。これらのモジュールは Kubernetes クラスター、ロードバランサー、MySQL データベースなど、クラウド用インフラ丸ごと一式と同時に W&B Server アプリケーションも構築します。W&B Kubernetes Operator は、この公式クラウド向け Terraform Modules にすでに組み込まれています。バージョンは以下の通りです。

| Terraform Registry                                                  | Source Code                                      | Version |
| ------------------------------------------------------------------- | ------------------------------------------------ | ------- |
| [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest) | https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| [Azure](https://github.com/wandb/terraform-azurerm-wandb)           | https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| [GCP](https://github.com/wandb/terraform-google-wandb)              | https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

この統合により、最小のセットアップ作業で W&B Kubernetes Operator の利用準備が整い、クラウド環境へのデプロイ・運用がスムーズに行えます。

具体的な利用方法は、ドキュメント内の [セルフマネージドインストールの章]({{< relref "../#deploy-wb-server-within-self-managed-cloud-accounts" >}}) を参照してください。

### インストールの確認

インストール確認には [W&B CLI]({{< relref "/ref/cli/" >}}) の利用を推奨します。verify コマンドで各コンポーネントや設定の自動検証を行います。

{{% alert %}}
この手順では、管理者ユーザーアカウントが事前にブラウザで作成されている必要があります。
{{% /alert %}}

手順は以下です：

1. W&B CLI をインストール:
    ```shell
    pip install wandb
    ```
2. W&B にログイン:
    ```shell
    wandb login --host=https://YOUR_DNS_DOMAIN
    ```

    例:
    ```shell
    wandb login --host=https://wandb.company-name.com
    ```

3. インストール確認:
    ```shell
    wandb verify
    ```

インストールが成功し、W&B の全機能が正常動作している場合、以下のような出力になります：

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

## W&B 管理コンソールへのアクセス
W&B Kubernetes operator には管理コンソールが付属しています。URL は `${HOST_URI}/console` です。例: `https://wandb.company-name.com/console`。

管理コンソールへのログイン方法は2通りあります。

{{< tabpane text=true >}}
{{% tab header="オプション 1（推奨）" value="option1" %}}
1. ブラウザで W&B アプリケーションにアクセスしログイン。`${HOST_URI}/`（例: `https://wandb.company-name.com/`）にアクセスします。
2. 画面右上のアイコンをクリックし、**System console** をクリック。管理者権限を持つユーザーのみ **System console** が表示されます。

    {{< img src="/images/hosting/access_system_console_via_main_app.png" alt="System console access" >}}
{{% /tab %}}

{{% tab header="オプション 2" value="option2"%}}
{{% alert %}}
オプション1でアクセスできない場合のみ、以下の手順でコンソールにアクセスしてください。
{{% /alert %}}

1. ブラウザでコンソールの URL を直接開いてください。ログイン画面へリダイレクトされます：
    {{< img src="/images/hosting/access_system_console_directly.png" alt="Direct system console access" >}}
2. インストール時に生成された Kubernetes シークレットからパスワードを取得します:
    ```shell
    kubectl get secret wandb-password -o jsonpath='{.data.password}' | base64 -d
    ```
    パスワードをコピーしてください。
3. コンソールにログイン。コピーしたパスワードを貼り付け **Login** をクリックします。
{{% /tab %}}
{{< /tabpane >}}

## W&B Kubernetes operator のアップデート
このセクションでは、W&B Kubernetes operator のアップデート方法を説明します。

{{% alert %}}
* W&B Kubernetes operator のアップデートは W&B Server アプリケーション自体のアップデートではありません。
* これまで operator を使わず Helm チャートで運用していた場合、[こちら]({{< relref "#migrate-self-managed-instances-to-wb-operator" >}}) の手順を先にご確認ください。
{{% /alert %}}

下記のコマンドをターミナルで実行してください。

1. まずリポジトリを [`helm repo update`](https://helm.sh/docs/helm/helm_repo_update/) で更新します:
    ```shell
    helm repo update
    ```

2. 次に [`helm upgrade`](https://helm.sh/docs/helm/helm_upgrade/) で Helm チャートをアップデート:
    ```shell
    helm upgrade operator wandb/operator -n wandb-cr --reuse-values
    ```

## W&B Server アプリケーションのアップデート
W&B Kubernetes operator をお使いの場合、W&B Server アプリケーションを個別にアップデートする必要はありません。

新バージョンがリリースされると、オペレーターが自動で W&B Server アプリケーションをアップデートします。

## セルフマネージド環境から W&B Operator への移行
この章では、従来のセルフマネージドな W&B Server インストールから、W&B Operator を使う方式へ移行する方法を説明します。移行手順は、これまでのインストール方法によって異なります。

{{% alert %}}
W&B Operator は W&B Server のデフォルト推奨インストール方式です。ご不明点は [Customer Support](mailto:support@wandb.com) または W&B チームまでご連絡ください。
{{% /alert %}}

- 公式 W&B Cloud Terraform Modules を利用していた場合は、該当するドキュメントに従ってください。
  - [AWS]({{< relref "#migrate-to-operator-based-aws-terraform-modules" >}})
  - [GCP]({{< relref "#migrate-to-operator-based-gcp-terraform-modules" >}})
  - [Azure]({{< relref "#migrate-to-operator-based-azure-terraform-modules" >}})
- [W&B Non-Operator Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/wandb) を利用していた場合、[こちら]({{< relref "#migrate-to-operator-based-helm-chart" >}}) をご覧ください。
- [W&B Non-Operator Helm chart with Terraform](https://registry.terraform.io/modules/wandb/wandb/kubernetes/latest) の場合は [こちら]({{< relref "#migrate-to-operator-based-terraform-helm-chart" >}}) へ。
- Kubernetes マニフェストでリソースを作成した場合も、[こちら]({{< relref "#migrate-to-operator-based-helm-chart" >}}) を参照。

### Operator ベース AWS Terraform Modules への移行

移行手順の詳細は [こちら]({{< relref "../install-on-public-cloud/aws-tf.md#migrate-to-operator-based-aws-terraform-modules" >}}) へ。

### Operator ベース GCP Terraform Modules への移行

[Customer Support](mailto:support@wandb.com) または W&B チームにお問い合わせください。

### Operator ベース Azure Terraform Modules への移行

[Customer Support](mailto:support@wandb.com) または W&B チームにお問い合わせください。

### Operator ベース Helm チャートへの移行

Operator ベースへの移行手順は次の通りです：

1. 現行の W&B 構成を取得。オペレータ未使用の Helm チャートでデプロイされている場合は下記でエクスポート。
    ```shell
    helm get values wandb
    ```
    Kubernetes マニフェストの場合は:
    ```shell
    kubectl get deployment wandb -o yaml
    ```
    このステップで必要な構成値が集まります。

2. `operator.yaml` というファイルを作成し、[Configuration Reference]({{< relref "#configuration-reference-for-wb-operator" >}}) のフォーマットに従い、1 の値を反映。

3. 現行デプロイメントの Pod を 0 にスケールダウン（停止）。
    ```shell
    kubectl scale --replicas=0 deployment wandb
    ```
4. Helm チャートリポジトリのアップデート:
    ```shell
    helm repo update
    ```
5. 新しい Helm チャートのインストール:
    ```shell
    helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
    ```
6. 新しい設定を適用して W&B アプリケーションのデプロイを開始:
    ```shell
    kubectl apply -f operator.yaml
    ```
    デプロイ完了まで数分かかります。

7. インストールを検証。[Verify the installation]({{< relref "#verify-the-installation" >}}) の手順に従い、全機能動作を確認。

8. 旧インストールの削除。不要な helm チャートやリソースをアンインストールまたは削除。

### Operator ベース Terraform Helm チャートへの移行

Operator ベースへの移行手順は次の通りです：

1. Terraform 構成変更。古い構成を [こちら]({{< relref "#deploy-wb-with-helm-terraform-module" >}}) のコードに置き換え、変数を維持します (.tfvars ファイルは変更不要)。
2. Terraform の実行。terraform init, plan, apply を連続実行します。
3. インストールを検証。[Verify the installation]({{< relref "#verify-the-installation" >}}) の手順に従い、動作確認します。
4. 旧インストールの削除。不要リソースや helm チャートを削除。

## W&B Server の設定リファレンス

このセクションでは W&B Server アプリケーションの設定オプションを説明します。アプリケーションは [WeightsAndBiases]({{< relref "#how-it-works" >}}) という名前のカスタムリソース定義として設定を受け取ります。一部は設定ファイルで指定し、その他は環境変数としてセットします。

環境変数の一覧は [基本]({{< relref "/guides/hosting/env-vars/" >}})・[高度]({{< relref "/guides/hosting/iam/advanced_env_vars/" >}}) の2種類あります。Helm チャートで指定できないもののみ環境変数を利用してください。

本番環境デプロイで最低限必要な W&B Server アプリケーション設定ファイルは以下の通りです。YAML で W&B デプロイの desired state（バージョン、環境変数、外部DBなど）を定義します。

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

全設定値は [W&B Helm repository](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml) でご確認ください。必要な項目だけ変更しましょう。

### 完全な例
これは GCP Kubernetes, GCP Ingress, GCS (GCP Object storage) を用いたサンプル設定例です：

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

### Host
```yaml
 # FQDN（プロトコル付き）を指定してください
global:
  # 以下は例です。ご自身のホスト名に置き換えてください
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

**その他プロバイダ（Minio、Ceph等）**

S3互換ストレージの場合、バケット設定例：
```yaml
global:
  bucket:
    # 自分の値を設定
    provider: s3
    name: storage.example.com
    kmsKey: null
    path: wandb
    region: default
    accessKey: 5WOA500...P5DK7I
    secretKey: HDKYe4Q...JAp1YyjysnX
```

AWS 以外の S3 互換ストレージでは、`kmsKey` を `null` にしてください。

`accessKey` と `secretKey` をシークレットから参照する場合：
```yaml
global:
  bucket:
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

### MySQL

```yaml
global:
   mysql:
     # 「例」です。ご自身の値に差し替えてください
     host: db.example.com
     port: 3306
     database: wandb_local
     user: wandb
     password: 8wtX6cJH...ZcUarK4zZGjpV 
```

`password` をシークレットから参照する場合：
```yaml
global:
   mysql:
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
  # 実際のライセンスキーに差し替えてください
  license: eyJhbGnUzaHgyQjQy...VFnPS_KETXg1hi
```

`license` をシークレット参照
```yaml
global:
  licenseSecret:
    name: license-secret
    key: CUSTOMER_WANDB_LICENSE
```

### Ingress

ingress class の確認方法は [FAQエントリ]({{< relref "#how-to-identify-the-kubernetes-ingress-class" >}}) 参照。

**TLSなし**

```yaml
global:
# ※「ingress」は「global」と同じインデント階層です
ingress:
  class: ""
```

**TLSあり**

証明書を含むシークレットを作成

```console
kubectl create secret tls wandb-ingress-tls --key wandb-ingress-tls.key --cert wandb-ingress-tls.crt
```

ingress 設定でシークレットを参照
```yaml
global:
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

Nginx を使う場合は次の annotation を追加します。

```
ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: 64m
```

### カスタム Kubernetes ServiceAccount

W&B の Pod を実行するサービスアカウントを指定できます。

以下の例では、デプロイメント時に指定名のサービスアカウントを作成：

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
「app」「parquet」サブシステムが指定サービスアカウントで動作。他はデフォルトサービスアカウントを利用。

既存サービスアカウントがある場合は `create: false`：

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

app, parquet, console など複数のサブシステムごとに指定可能：

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

違うサービスアカウントをサブシステムごとに指定可能：

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

### 外部 Redis

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

`password` をシークレットから参照する場合：

```console
kubectl create secret generic redis-secret --from-literal=redis-password=supersecret
```

参照例:
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
**TLSなし**
```yaml
global:
  ldap:
    enabled: true
    # LDAP サーバーアドレス（"ldap://"または"ldaps://"付き）
    host:
    # ユーザー検索用 baseDN
    baseDN:
    # バインド用ユーザー（匿名バインドしない場合）
    bindDN:
    # バインドパスワード用シークレット名・キー（匿名バインドしない場合）
    bindPW:
    # メールアドレスやグループID属性名（カンマ区切り）
    attributes:
    # グループ許可リスト
    groupAllowList:
    # LDAP TLS 利用
    tls: false
```

**TLSあり**

LDAP TLS証明書用 ConfigMap を事前作成します。

```console
kubectl create configmap ldap-tls-cert --from-file=certificate.crt
```

ConfigMap 利用例：

```yaml
global:
  ldap:
    enabled: true
    # LDAP サーバーアドレス（"ldap://" または "ldaps://"）
    host:
    # ユーザー検索用 baseDN
    baseDN:
    # バインドユーザー
    bindDN:
    # バインドパスワード用シークレット
    bindPW:
    # メールやグループID属性名（カンマ区切り）
    attributes:
    # グループ許可リスト
    groupAllowList:
    # LDAP TLS を有効化
    tls: true
    # LDAP サーバー用 CA証明書を含むConfigMap名とキー
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
      # IdP により必要な場合のみ指定
      authMethod: ""
      issuer: ""
```

`authMethod` は任意指定です。

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

### カスタム証明書認証局

`customCACerts` はリストで複数証明書を指定可。ここで指定したものは W&B Server アプリケーションのみに適用されます。

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

CA 証明書を ConfigMap で渡すこともできます：
```yaml
global:
  caCertsConfigMap: custom-ca-certs
```

ConfigMap 例：
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
ConfigMap の各キーは `.crt` で終わる必要があります（例: `my-cert.crt`、`ca-cert1.crt` など）。この命名規則がないと `update-ca-certificates` がシステム CA ストアに追加できません。
{{% /alert %}}

### カスタム セキュリティコンテキスト

各 W&B コンポーネントは以下の形でカスタムセキュリティコンテキストをサポートしています：

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
`runAsGroup:` は `0` のみが有効値です。それ以外を指定するとエラーになります。
{{% /alert %}}

例えばアプリケーション Pod の構成は下記のように定義します：

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

同じ方式で `console`, `weave`, `weave-trace`, `parquet` も設定できます。

## W&B Operator の設定リファレンス

このセクションでは W&B Kubernetes operator (`wandb-controller-manager`) の設定オプションを説明します。オペレーターは YAML で設定を受け取ります。

基本的には W&B Kubernetes operator は設定ファイル不要です。必要な場合にのみ作成してください（例: カスタム CA, エアギャップ環境対応など）。

全 spec のカスタマイズは [Helm repository](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml) を参照してください。

### カスタム CA
`customCACerts` はリストで複数証明書を指定可。ここで指定したものは W&B Kubernetes オペレーター（`wandb-controller-manager`）のみに適用されます。

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

CA 証明書を ConfigMap で渡すことも可能です：
```yaml
caCertsConfigMap: custom-ca-certs
```

ConfigMap 例：
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
ConfigMap の各キー名は `.crt` で終わる必要があります（例: `my-cert.crt`、`ca-cert1.crt`）。この命名規則がないと `update-ca-certificates` がシステム CA ストアに追加できません。
{{% /alert %}}

## FAQ

### 各 Pod の役割・用途は？

* **`wandb-app`**: W&B のコア。GraphQL API とフロントエンドを含み、プラットフォーム機能の大半を担います。
* **`wandb-console`**: 管理用コンソール。`/console` でアクセス。
* **`wandb-otel`**: OpenTelemetry エージェント。Kubernetes レイヤーのリソースからメトリクス・ログを収集し管理コンソールに表示。
* **`wandb-prometheus`**: Prometheus サーバー。各コンポーネントのメトリクスを管理コンソールに送ります。
* **`wandb-parquet`**: `wandb-app` とは別のバックエンドマイクロサービス。DBデータを Parquet 形式でオブジェクトストレージにエクスポートします。
* **`wandb-weave`**: UI でテーブルを読み込み、各種コア機能を提供するバックエンドサービス。
* **`wandb-weave-trace`**: LLM アプリ開発の追跡・実験・評価・デプロイ・改善用フレームワーク。`wandb-app` Pod からアクセスされます。

### W&B Operator コンソールのパスワード取得方法

[W&B Kubernetes Operator 管理コンソールへのアクセス方法]({{< relref "#access-the-wb-management-console" >}}) をご確認ください。

### Ingress が動かない場合の W&B Operator Console へのアクセス

Kubernetes クラスターに接続できる端末で下記コマンドを実行：

```console
kubectl port-forward svc/wandb-console 8082
```

ブラウザで `https://localhost:8082/` にアクセスできます。

パスワード取得方法は [W&B Kubernetes Operator 管理コンソールへのアクセス]({{< relref "#access-the-wb-management-console" >}}) の「オプション2」を参照。

### W&B Server のログ確認方法

アプリケーションポッド名は **wandb-app-xxx** です。

```console
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

### Kubernetes ingress class の確認方法

クラスターにインストール済みの ingress class を確認するには：

```console
kubectl get ingressclass
```