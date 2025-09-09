---
title: Kubernetes 上で W&B サーバーを実行する
description: Kubernetes Operator で W&B プラットフォーム をデプロイする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-kubernetes-operator-_index
    parent: self-managed
url: guides/hosting/operator
weight: 2
---

## W&B Kubernetes Operator

W&B Kubernetes Operator を使うと、Kubernetes 上での W&B Server のデプロイ、運用、トラブルシュート、スケールをシンプルにできます。W&B インスタンス用のスマートアシスタント、と考えてください。

W&B Server のアーキテクチャーは、高性能・スケーラビリティ・運用容易性を高めるための適切なプリミティブを提供しつつ、AI 開発者向けツール群の機能拡張に合わせて継続的に進化しています。この進化は、コンピュートサービス、関連するストレージ、およびそれらの接続性に及びます。デプロイメント形態をまたいだ継続的なアップデートと改善を促進するため、W&B は Kubernetes オペレーターを採用しています。

{{% alert %}}
W&B は、このオペレーターを使って AWS、GCP、Azure のパブリッククラウド上に 専用クラウド インスタンスをデプロイ・管理します。
{{% /alert %}}

Kubernetes オペレーターの詳細は、Kubernetes ドキュメントの [Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) を参照してください。

### アーキテクチャー変更の理由
歴史的に、W&B アプリケーションは Kubernetes クラスター内の単一の Deployment/Pod または単一の Docker コンテナとしてデプロイされていました。W&B はこれまでも、そして今後も、Database と Object Store の外部化を推奨しています。Database と Object Store を外部化することで、アプリケーションの状態を分離できます。

アプリケーションの拡大に伴い、モノリシックなコンテナから分散システム（マイクロサービス）へ進化する必要性が明確になりました。この変更により、バックエンドロジックの処理が容易になり、Kubernetes の組み込み機能をシームレスに取り込めます。分散システムにすることで、W&B の新機能に不可欠な新しいサービスのデプロイも可能になります。

2024 年以前は、Kubernetes に関連する変更はすべて [terraform-kubernetes-wandb](https://github.com/wandb/terraform-kubernetes-wandb) Terraform モジュールを手動で更新する必要がありました。Terraform モジュールの更新は、クラウドプロバイダー間の互換性を確保し、必要な Terraform 変数を設定し、バックエンドや Kubernetes レベルの変更ごとに Terraform apply を実行することを意味します。

このプロセスはスケールしませんでした。毎回、各顧客の Terraform モジュールのアップグレードを W&B Support が支援する必要があったためです。

その解決策として、オペレーターを実装し、中央の [deploy.wandb.ai](https://deploy.wandb.ai) サーバーに接続して、指定したリリースチャンネルの最新仕様変更を取得して適用するようにしました。ライセンスが有効である限り、更新を受け取れます。[Helm](https://helm.sh/) は、W&B オペレーターのデプロイ手段であると同時に、W&B Kubernetes スタック全体の設定テンプレートをオペレーターが扱うための仕組みとしても使われています（Helm の二重活用）。

### 仕組み
オペレーターは Helm もしくはソースからインストールできます。詳しい手順は [charts/operator](https://github.com/wandb/helm-charts/tree/main/charts/operator) を参照してください。

インストール時に `controller-manager` という名前の Deployment が作成され、`weightsandbiases.apps.wandb.com`（shortName: `wandb`）という [custom resource](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) 定義を使って、単一の `spec` をクラスターに適用します:

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: weightsandbiases.apps.wandb.com
```

`controller-manager` は、カスタムリソースの spec、リリースチャンネル、ユーザー定義の config に基づいて [charts/operator-wandb](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb) をインストールします。設定仕様の階層により、ユーザー側で最大限の柔軟性を確保しつつ、W&B 側は新しいイメージ、設定、機能、Helm の更新を自動でリリースできます。

設定オプションについては、[設定仕様の階層]({{< relref path="#configuration-specification-hierarchy" lang="ja" >}}) および [設定リファレンス]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) を参照してください。

デプロイメントは複数の Pod（サービスごとに 1 つ）で構成されます。各 Pod の名前には `wandb-` というプレフィックスが付きます。

### 設定仕様の階層
設定仕様は階層モデルに従い、上位レベルの指定が下位レベルの指定を上書きします。仕組みは次のとおりです。

- リリースチャンネルの値: このベースレベルの設定は、W&B がデプロイメントに対して設定したリリースチャンネルに基づくデフォルト値と設定を定義します。
- ユーザー入力の値: ユーザーは System Console から、リリースチャンネルの設定で提供されるデフォルトを上書きできます。
- カスタムリソースの値: 最上位の指定で、ユーザーが提供します。ここで指定した値は、ユーザー入力とリリースチャンネルの両方を上書きします。設定オプションの詳細は [設定リファレンス]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) を参照してください。

この階層モデルにより、アップグレードや変更を体系的に管理しながら、さまざまなニーズに合わせて柔軟にカスタマイズできます。

### W&B Kubernetes Operator の利用要件
W&B を W&B Kubernetes オペレーターでデプロイするには、次の要件を満たしてください。

[リファレンスアーキテクチャー]({{< relref path="../ref-arch.md#infrastructure-requirements" lang="ja" >}}) を参照し、加えて [有効な W&B Server ライセンスを取得]({{< relref path="../#obtain-your-wb-server-license" lang="ja" >}}) してください。

セルフマネージドのインストール手順は、[ベアメタルインストールガイド]({{< relref path="../bare-metal.md" lang="ja" >}}) を参照してください。

インストール方法によっては、次を満たす必要があります。
* 正しい Kubernetes クラスターコンテキストで設定済みの Kubectl がインストールされていること。
* Helm がインストールされていること。

### エアギャップ環境でのインストール
エアギャップ環境で W&B Kubernetes Operator をインストールする方法は、[Deploy W&B in airgapped environment with Kubernetes]({{< relref path="operator-airgapped.md" lang="ja" >}}) チュートリアルを参照してください。

## W&B Server アプリケーションのデプロイ
このセクションでは、W&B Kubernetes オペレーターを使ったさまざまなデプロイ方法を説明します。
{{% alert %}}
W&B Operator は W&B Server のデフォルトかつ推奨のインストール方法です。
{{% /alert %}}

### Helm CLI で W&B をデプロイ
W&B は、Kubernetes クラスターに W&B Kubernetes オペレーターをデプロイするための Helm Chart を提供しています。この方法により、Helm CLI や ArgoCD のような継続的デリバリーツールで W&B Server をデプロイできます。上記の要件を満たしていることを確認してください。

Helm CLI で W&B Kubernetes Operator をインストールする手順:

1. W&B Helm リポジトリを追加します。W&B Helm チャートは W&B Helm リポジトリで提供されています:
    ```shell
    helm repo add wandb https://charts.wandb.ai
    helm repo update
    ```
2. Kubernetes クラスターに Operator をインストールします:
    ```shell
    helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
    ```
3. W&B Server のインストールをトリガーするため、W&B オペレーターのカスタムリソースを設定します。W&B デプロイの設定を記した `operator.yaml` を作成してください。利用可能なオプションは [設定リファレンス]({{< relref path="#configuration-reference-for-wb-server" lang="ja" >}}) を参照してください。

    最小構成の例は次のとおりです:

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

4. カスタム設定で Operator を起動し、W&B Server アプリケーションのインストール・設定・管理を行わせます:

    ```shell
    kubectl apply -f operator.yaml
    ```

    デプロイ完了まで数分待ちます。

5. Web UI でインストールを検証します。最初の管理者ユーザーアカウントを作成し、[インストールの検証]({{< relref path="#verify-the-installation" lang="ja" >}}) の手順に従ってください。


### Helm Terraform Module で W&B をデプロイ

この方法は、Terraform の IaC アプローチを活用して、要件に合わせてカスタマイズされたデプロイを一貫性と再現性をもって実現します。公式の W&B Helm ベース Terraform Module は [こちら](https://registry.terraform.io/modules/wandb/wandb/helm/latest) にあります。

以下のコードは出発点として使用でき、プロダクション品質のデプロイに必要な設定オプションをすべて含みます。

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

設定オプションは [設定リファレンス]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) で説明している内容と同じですが、記法は HashiCorp Configuration Language (HCL) に従う必要があります。Terraform モジュールは W&B のカスタムリソース定義（CRD）を作成します。

Weights & Biases 自身が顧客向けの「専用クラウド」インストールをデプロイする際に Helm Terraform モジュールをどのように使っているかは、次のリンクを参照してください:
- [AWS](https://github.com/wandb/terraform-aws-wandb/blob/45e1d746f53e78e73e68f911a1f8cad5408e74b6/main.tf#L225)
- [Azure](https://github.com/wandb/terraform-azurerm-wandb/blob/170e03136b6b6fc758102d59dacda99768854045/main.tf#L155)
- [GCP](https://github.com/wandb/terraform-google-wandb/blob/49ddc3383df4cefc04337a2ae784f57ce2a2c699/main.tf#L189)

### W&B Cloud Terraform modules で W&B をデプロイ

W&B は AWS、GCP、Azure 向けの Terraform Modules を提供しています。これらのモジュールは、Kubernetes クラスター、ロードバランサー、MySQL データベースなどのインフラ一式と、W&B Server アプリケーションをデプロイします。公式の W&B クラウド別 Terraform Modules には、すでに W&B Kubernetes Operator が以下のバージョンで組み込まれています:

| Terraform Registry                                                  | Source Code                                      | バージョン |
| ------------------------------------------------------------------- | ------------------------------------------------ | ---------- |
| [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest) | https://github.com/wandb/terraform-aws-wandb     | v4.0.0+    |
| [Azure](https://github.com/wandb/terraform-azurerm-wandb)           | https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+    |
| [GCP](https://github.com/wandb/terraform-google-wandb)              | https://github.com/wandb/terraform-google-wandb  | v2.0.0+    |

この統合により、最小限のセットアップで W&B Kubernetes Operator をすぐに使える状態にし、クラウド環境での W&B Server のデプロイと管理をスムーズに開始できます。

これらのモジュールの詳しい使い方は、ドキュメントの [セルフマネージドインストールのセクション]({{< relref path="../#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}}) を参照してください。

### インストールの検証

インストールの検証には [W&B CLI]({{< relref path="/ref/cli/" lang="ja" >}}) の使用を推奨します。verify コマンドは、すべてのコンポーネントや設定を検証するテストを実行します。

{{% alert %}}
この手順では、最初の管理者ユーザーアカウントがブラウザで作成済みであることを前提としています。
{{% /alert %}}

次の手順に従ってインストールを検証します。

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

3. インストールを検証:
    ```shell
    wandb verify
    ```

インストールが成功し、W&B のデプロイが正常に動作している場合、次のような出力になります:

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
W&B Kubernetes オペレーターには管理コンソールが付属します。場所は `${HOST_URI}/console`（例: `https://wandb.company-name.com/console`）です。

管理コンソールへのログイン方法は 2 通りあります。

{{< tabpane text=true >}}
{{% tab header="Option 1 (Recommended)" value="option1" %}}
1. ブラウザで W&B アプリケーションを開き、ログインします。`${HOST_URI}/`（例: `https://wandb.company-name.com/`）にアクセスしてログインします。
2. コンソールにアクセスします。右上のアイコンをクリックし、**System console** をクリックします。管理者権限のあるユーザーだけが **System console** エントリを表示できます。

    {{< img src="/images/hosting/access_system_console_via_main_app.png" alt="System console へのアクセス" >}}
{{% /tab %}}

{{% tab header="Option 2" value="option2"%}}
{{% alert %}}
Option 1 が使えない場合のみ、以下の手順でコンソールにアクセスすることを推奨します。
{{% /alert %}}

1. ブラウザでコンソールアプリケーションを開きます。上記の URL を開くとログイン画面にリダイレクトされます:
    {{< img src="/images/hosting/access_system_console_directly.png" alt="System console に直接アクセス" >}}
2. インストール時に生成された Kubernetes Secret からパスワードを取得します:
    ```shell
    kubectl get secret wandb-password -o jsonpath='{.data.password}' | base64 -d
    ```
    パスワードをコピーします。
3. コンソールにログインします。コピーしたパスワードを貼り付け、**Login** をクリックします。
{{% /tab %}}
{{< /tabpane >}}

## W&B Kubernetes オペレーターの更新
このセクションでは、W&B Kubernetes オペレーターの更新方法を説明します。

{{% alert %}}
* W&B Kubernetes オペレーターを更新しても、W&B Server アプリケーションは更新されません。
* 先に W&B Kubernetes オペレーターを使わない Helm チャートを使用している場合の移行手順は [こちら]({{< relref path="#migrate-self-managed-instances-to-wb-operator" lang="ja" >}}) を参照してください。その後で以下の手順に従って W&B オペレーターを更新してください。
{{% /alert %}}

以下のコードスニペットを ターミナル にコピー＆ペーストしてください。

1. まず、[`helm repo update`](https://helm.sh/docs/helm/helm_repo_update/) でリポジトリを更新します:
    ```shell
    helm repo update
    ```

2. 次に、[`helm upgrade`](https://helm.sh/docs/helm/helm_upgrade/) で Helm チャートを更新します:
    ```shell
    helm upgrade operator wandb/operator -n wandb-cr --reuse-values
    ```

## W&B Server アプリケーションの更新
W&B Kubernetes オペレーターを使用している場合、W&B Server アプリケーションを手動で更新する必要はありません。

新しい W&B のバージョンがリリースされると、オペレーターが自動的に W&B Server アプリケーションを更新します。


## セルフマネージドインスタンスを W&B Operator に移行
以下では、セルフマネージドの W&B Server インストールから、W&B Operator による管理へ移行する方法を説明します。移行プロセスは、W&B Server のインストール方法によって異なります。

{{% alert %}}
W&B Operator は W&B Server のデフォルトかつ推奨のインストール方法です。ご不明点があれば [Customer Support](mailto:support@wandb.com) または担当の W&B チームにお問い合わせください。
{{% /alert %}}

- 公式の W&B Cloud Terraform Modules を使用している場合は、該当ドキュメントに移動して手順に従ってください:
  - [AWS]({{< relref path="#migrate-to-operator-based-aws-terraform-modules" lang="ja" >}})
  - [GCP]({{< relref path="#migrate-to-operator-based-gcp-terraform-modules" lang="ja" >}})
  - [Azure]({{< relref path="#migrate-to-operator-based-azure-terraform-modules" lang="ja" >}})
- [W&B Non-Operator Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/wandb) を使っている場合は、[こちら]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ja" >}}) に進んでください。
- [W&B Non-Operator Helm chart with Terraform](https://registry.terraform.io/modules/wandb/wandb/kubernetes/latest) を使っている場合は、[こちら]({{< relref path="#migrate-to-operator-based-terraform-helm-chart" lang="ja" >}}) に進んでください。
- Kubernetes マニフェストでリソースを作成した場合は、[こちら]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ja" >}}) に進んでください。


### Operator ベースの AWS Terraform Modules に移行

移行プロセスの詳細は、[こちら]({{< relref path="../install-on-public-cloud/aws-tf.md#migrate-to-operator-based-aws-terraform-modules" lang="ja" >}}) を参照してください。

### Operator ベースの GCP Terraform Modules に移行

ご不明点や支援が必要な場合は [Customer Support](mailto:support@wandb.com) または担当の W&B チームにお問い合わせください。


### Operator ベースの Azure Terraform Modules に移行

ご不明点や支援が必要な場合は [Customer Support](mailto:support@wandb.com) または担当の W&B チームにお問い合わせください。

### Operator ベースの Helm チャートに移行

Operator ベースの Helm チャートに移行するには、次の手順に従ってください。

1. 現在の W&B の設定を取得します。W&B をオペレーター非対応の Helm チャートでデプロイしている場合、次のように値をエクスポートします:
    ```shell
    helm get values wandb
    ```
    W&B を Kubernetes マニフェストでデプロイしている場合、次のように値をエクスポートします:
    ```shell
    kubectl get deployment wandb -o yaml
    ```
    これで次のステップに必要な設定値は揃いました。 

2. `operator.yaml` というファイルを作成します。[設定リファレンス]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) の形式に従い、ステップ 1 の値を用います。

3. 現行のデプロイメントを 0 Pod にスケールダウンします。これにより、現在のデプロイメントが停止します。
    ```shell
    kubectl scale --replicas=0 deployment wandb
    ```
4. Helm チャートのリポジトリを更新します:
    ```shell
    helm repo update
    ```
5. 新しい Helm チャートをインストールします:
    ```shell
    helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
    ```
6. 新しい Helm チャートを設定し、W&B アプリケーションのデプロイをトリガーします。新しい設定を適用します。
    ```shell
    kubectl apply -f operator.yaml
    ```
    デプロイ完了まで数分かかります。

7. インストールを検証します。[インストールの検証]({{< relref path="#verify-the-installation" lang="ja" >}}) の手順に従って、すべてが動作していることを確認します。

8. 旧インストールを削除します。古い Helm チャートをアンインストールするか、マニフェストで作成したリソースを削除します。

### Operator ベースの Terraform Helm チャートに移行

Operator ベースの Helm チャートに移行するには、次の手順に従ってください。


1. Terraform の設定を準備します。Terraform 構成内で、旧デプロイの Terraform コードを [こちら]({{< relref path="#deploy-wb-with-helm-terraform-module" lang="ja" >}}) に記載のコードに置き換えます。以前と同じ変数を設定してください。.tfvars ファイルを使用している場合は変更しないでください。
2. Terraform を実行します。terraform init、plan、apply を実行します。
3. インストールを検証します。[インストールの検証]({{< relref path="#verify-the-installation" lang="ja" >}}) の手順に従って、すべてが動作していることを確認します。
4. 旧インストールを削除します。古い Helm チャートをアンインストールするか、マニフェストで作成したリソースを削除します。



## W&B Server の設定リファレンス

このセクションでは、W&B Server アプリケーションの設定オプションを説明します。アプリケーションは [WeightsAndBiases]({{< relref path="#how-it-works" lang="ja" >}}) という名前のカスタムリソース定義で設定を受け取ります。いくつかの設定オプションは以下の設定で公開され、その他は環境変数で指定する必要があります。

環境変数はドキュメント内の [basic]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}) と [advanced]({{< relref path="/guides/hosting/iam/advanced_env_vars/" lang="ja" >}}) の 2 つのリストに分かれています。必要な設定オプションが Helm Chart で公開されていない場合にのみ、環境変数を使用してください。

### 基本例
この例は W&B に必要な最小限の値を定義します。より現実的なプロダクション例は [完全な例]({{< relref path="#complete-example" lang="ja" >}}) を参照してください。

この YAML ファイルは、バージョン、環境変数、データベースなどの外部リソース、その他の必要な設定を含む、W&B デプロイの望ましい状態を定義します。

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

すべての値の一覧は [W&B Helm リポジトリ](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml) にあります。**上書きが必要な値のみ変更してください。**

### 完全な例
この例は、GCP Anthos に Google Cloud Storage を使って W&B をデプロイします:

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
 # プロトコル付きの FQDN を指定
global:
  # ホスト名の例。自身の値に置き換えてください
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

**その他のプロバイダー（Minio、Ceph など）**

他の S3 互換プロバイダーの場合、バケットの設定は次のとおりです:
```yaml
global:
  bucket:
    # 値は例です。自身の値に置き換えてください
    provider: s3
    name: storage.example.com
    kmsKey: null
    path: wandb
    region: default
    accessKey: 5WOA500...P5DK7I
    secretKey: HDKYe4Q...JAp1YyjysnX
```

AWS の外部にホストされた S3 互換ストレージでは、`kmsKey` は `null` にする必要があります。

Secret にある `accessKey` と `secretKey` を参照するには:
```yaml
global:
  bucket:
    # 値は例です。自身の値に置き換えてください
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
     # 値は例です。自身の値に置き換えてください
     host: db.example.com
     port: 3306
     database: wandb_local
     user: wandb
     password: 8wtX6cJH...ZcUarK4zZGjpV 
```

`password` を Secret から参照するには:
```yaml
global:
   mysql:
     # 値は例です。自身の値に置き換えてください
     host: db.example.com
     port: 3306
     database: wandb_local
     user: wandb
     passwordSecret:
       name: database-secret
       passwordKey: MYSQL_WANDB_PASSWORD
```

### License

```yaml
global:
  # ライセンスの例。自身のものに置き換えてください
  license: eyJhbGnUzaHgyQjQy...VFnPS_KETXg1hi
```

`license` を Secret から参照するには:
```yaml
global:
  licenseSecret:
    name: license-secret
    key: CUSTOMER_WANDB_LICENSE
```

### Ingress

Ingress クラスの特定方法は、この FAQ の [項目]({{< relref path="#how-to-identify-the-kubernetes-ingress-class" lang="ja" >}}) を参照してください。

**TLS なし**

```yaml
global:
# 重要: Ingress は YAML 上で ‘global’ と同じレベル（子ではない）
ingress:
  class: ""
```

**TLS あり**

証明書を含む Secret を作成します

```console
kubectl create secret tls wandb-ingress-tls --key wandb-ingress-tls.key --cert wandb-ingress-tls.crt
```

Ingress 設定で Secret を参照します
```yaml
global:
# 重要: Ingress は YAML 上で ‘global’ と同じレベル（子ではない）
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

Nginx の場合、次のアノテーションが必要になることがあります:

```
ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: 64m
```

### カスタム Kubernetes ServiceAccount

W&B の Pod を実行するカスタム Kubernetes サービスアカウントを指定できます。

次のスニペットは、指定した名前でデプロイ時にサービスアカウントを作成します:

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
サブシステム「app」と「parquet」は指定のサービスアカウントで動作します。その他のサブシステムはデフォルトのサービスアカウントで動作します。

サービスアカウントがすでにクラスターに存在する場合は、`create: false` を設定します:

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

app、parquet、console など、サブシステムごとにサービスアカウントを指定できます:

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

サブシステムごとに異なるサービスアカウントを指定することもできます:

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

`password` を Secret から参照するには:

```console
kubectl create secret generic redis-secret --from-literal=redis-password=supersecret
```

以下の設定で参照します:
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
    # "ldap://" または "ldaps://" を含む LDAP サーバーアドレス
    host:
    # ユーザー検索に使用する LDAP の search base
    baseDN:
    # 匿名バインドを使わない場合のバインドユーザー
    bindDN:
    # 匿名バインドを使わない場合のバインドパスワードを格納した Secret 名とキー
    bindPW:
    # メールとグループ ID に使う LDAP 属性名（カンマ区切りの文字列）
    attributes:
    # 許可する LDAP グループのリスト
    groupAllowList:
    # LDAP の TLS を有効化
    tls: false
```

**TLS あり**

LDAP の TLS 証明書設定には、証明書の内容を格納した ConfigMap を事前に作成しておく必要があります。

ConfigMap は次のコマンドで作成できます:

```console
kubectl create configmap ldap-tls-cert --from-file=certificate.crt
```

作成した ConfigMap を、以下の例のように YAML で使用します

```yaml
global:
  ldap:
    enabled: true
    # "ldap://" または "ldaps://" を含む LDAP サーバーアドレス
    host:
    # ユーザー検索に使用する LDAP の search base
    baseDN:
    # 匿名バインドを使わない場合のバインドユーザー
    bindDN:
    # 匿名バインドを使わない場合のバインドパスワードを格納した Secret 名とキー
    bindPW:
    # メールとグループ ID に使う LDAP 属性名（カンマ区切りの文字列）
    attributes:
    # 許可する LDAP グループのリスト
    groupAllowList:
    # LDAP の TLS を有効化
    tls: true
    # LDAP サーバー用 CA 証明書を格納した ConfigMap の名前とキー
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
      # IdP が必要とする場合のみ指定
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

### カスタム認証局（CA）
`customCACerts` はリストで、複数の証明書を指定できます。`customCACerts` に指定した認証局は W&B Server アプリケーションにのみ適用されます。

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

CA 証明書は ConfigMap に保存することもできます:
```yaml
global:
  caCertsConfigMap: custom-ca-certs
```

ConfigMap は次のようになります:
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
ConfigMap を使用する場合、ConfigMap 内の各キーは `.crt`（例: `my-cert.crt` や `ca-cert1.crt`）で終わっている必要があります。この命名規則は、`update-ca-certificates` が各証明書を解析してシステムの CA ストアに追加するために必要です。
{{% /alert %}}

### カスタムセキュリティコンテキスト

各 W&B コンポーネントは、以下の形式でカスタムセキュリティコンテキストを設定できます。

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
`runAsGroup:` に設定できる有効な値は `0` のみです。その他の値はエラーになります。
{{% /alert %}}


例えば、アプリケーションの Pod を設定するには、設定に `app` セクションを追加します:

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

同様の考え方は、`console`、`weave`、`weave-trace`、`parquet` にも適用できます。

## W&B Operator の設定リファレンス

このセクションでは、W&B Kubernetes オペレーター（`wandb-controller-manager`）の設定オプションを説明します。オペレーターは YAML ファイルの形式で設定を受け取ります。

デフォルトでは、W&B Kubernetes オペレーターに設定ファイルは不要です。必要に応じて作成してください。例えば、カスタム認証局の指定やエアギャップ環境でのデプロイなどが該当します。

spec のカスタマイズの全一覧は [Helm リポジトリ](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml) を参照してください。

### カスタム CA
カスタム認証局（`customCACerts`）はリストで、複数の証明書を指定できます。ここで追加した認証局は、W&B Kubernetes オペレーター（`wandb-controller-manager`）にのみ適用されます。

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

CA 証明書は ConfigMap に保存することもできます:
```yaml
caCertsConfigMap: custom-ca-certs
```

ConfigMap は次のようになります:
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
ConfigMap を使用する場合、ConfigMap 内の各キーは `.crt`（例: `my-cert.crt` や `ca-cert1.crt`）で終わっている必要があります。この命名規則は、`update-ca-certificates` が各証明書を解析してシステムの CA ストアに追加するために必要です。
{{% /alert %}}

## FAQ

### 各 Pod の役割は？
* **`wandb-app`**: W&B の中核で、GraphQL API とフロントエンドアプリケーションを含みます。プラットフォーム機能の大部分を提供します。
* **`wandb-console`**: 管理コンソール（`/console` からアクセス）。
* **`wandb-otel`**: OpenTelemetry エージェント。Kubernetes レイヤーのリソースからメトリクスとログを収集し、管理コンソールに表示します。
* **`wandb-prometheus`**: Prometheus サーバー。各コンポーネントからメトリクスを取得し、管理コンソールに表示します。
* **`wandb-parquet`**: `wandb-app` Pod とは別のバックエンドマイクロサービスで、データベースのデータを Parquet 形式でオブジェクトストレージにエクスポートします。
* **`wandb-weave`**: もう一つのバックエンドマイクロサービスで、UI でクエリテーブルをロードし、さまざまなコア機能を支えます。
* **`wandb-weave-trace`**: LLM ベースのアプリケーションのトラッキング、実験、評価、デプロイ、改善のためのフレームワーク。`wandb-app` Pod を介してアクセスします。

### W&B Operator Console のパスワードの取得方法
[W&B Kubernetes オペレーター管理コンソールへのアクセス]({{< relref path="#access-the-wb-management-console" lang="ja" >}}) を参照してください。


### Ingress が機能しない場合の W&B Operator Console へのアクセス方法

Kubernetes クラスターへ到達可能なホストで次のコマンドを実行します:

```console
kubectl port-forward svc/wandb-console 8082
```

ブラウザで `https://localhost:8082/` にアクセスしてコンソールを開きます。

パスワードの取得方法（Option 2）は [W&B Kubernetes オペレーター管理コンソールへのアクセス]({{< relref path="#access-the-wb-management-console" lang="ja" >}}) を参照してください。

### W&B Server のログ表示方法

アプリケーションの Pod 名は **wandb-app-xxx** です。

```console
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

### Kubernetes の Ingress クラスの特定方法

クラスターにインストールされている Ingress クラスは、次のコマンドで確認できます

```console
kubectl get ingressclass
```