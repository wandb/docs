---
title: Run W&B Server on Kubernetes
description: Kubernetes オペレーターを使用して W&B プラットフォーム をデプロイする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-kubernetes-operator-_index
    parent: self-managed
url: guides/hosting/operator
weight: 2
---

## W&B Kubernetes Operator

W&B Kubernetes Operator を使用して、Kubernetes 上での W&B サーバーデプロイメントのデプロイ、管理、トラブルシューティング、スケーリングを簡素化します。オペレーターは、W&B インスタンスのスマートなアシスタントと考えることができます。

W&B サーバーのアーキテクチャーと設計は、AI 開発者向けのツールの能力を拡張し、高パフォーマンス、より良いスケーラビリティ、より簡単な管理のための適切なプリミティブを提供するために、継続的に進化しています。この進化は、コンピュートサービス、関連するストレージ、およびそれらの間の接続性に適用されます。デプロイメントタイプ全体での継続的な更新と改善を促進するために、W&B ユーザーは Kubernetes オペレーターを使用します。

{{% alert %}}
W&B はオペレーターを使用して、AWS、GCP、Azure のパブリッククラウドで Dedicated cloud インスタンスをデプロイおよび管理します。
{{% /alert %}}

Kubernetes オペレーターの詳細については、Kubernetes ドキュメントの [Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) を参照してください。

### アーキテクチャー変更の理由

歴史的に、W&B アプリケーションは Kubernetes クラスター内または単一の Docker コンテナ内で単一デプロイメントとポッドとしてデプロイされていました。W&B は、データベースとオブジェクトストアを外部化することを推奨し続けています。データベースとオブジェクトストアを外部化することで、アプリケーションの状態が分離されます。

アプリケーションが成長するにつれて、モノリシックなコンテナから分散システム (マイクロサービス) への進化の必要性が明らかになりました。この変更により、バックエンドのロジック処理が可能になり、Kubernetes の組み込みのインフラストラクチャー機能がスムーズに導入されます。分散システムは、W&B が依存している追加機能に不可欠な新しいサービスのデプロイをサポートします。

2024 年以前は、Kubernetes に関連する変更には、[terraform-kubernetes-wandb](https://github.com/wandb/terraform-kubernetes-wandb) Terraform モジュールの手動更新が必要でした。Terraform モジュールを更新することで、クラウドプロバイダー間の互換性を確保し、必要な Terraform 変数を設定し、各バックエンドまたは Kubernetes レベルの変更に対して Terraform を適用します。

このプロセスはスケーラブルではありませんでした。W&B サポートは、Terraform モジュールのアップグレード時に各顧客を支援しなければなりませんでした。

解決策は、中央の [deploy.wandb.ai](https://deploy.wandb.ai) サーバーに接続して、特定のリリースチャンネルの最新の仕様変更を要求し、それを適用するオペレーターを実装することでした。ライセンスが有効な限り、更新を受け取ります。[Helm](https://helm.sh/) は、W&B オペレーターのデプロイメントメカニズムとして、また W&B Kubernetes スタックのすべての設定テンプレートを処理する手段として使用されます。これは Helm-ception と呼ばれます。

### 仕組み

オペレーターを helm またはソースからインストールできます。詳細な手順については、[charts/operator](https://github.com/wandb/helm-charts/tree/main/charts/operator) を参照してください。

インストールプロセスは `controller-manager` と呼ばれるデプロイメントを作成し、クラスタに適用する単一の `spec` を使用する `weightsandbiases.apps.wandb.com` (短縮名: `wandb`) という名前の [custom resource](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) 定義を使用します。

apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: weightsandbiases.apps.wandb.com

`controller-manager` は、カスタムリソース、リリースチャンネル、およびユーザー定義設定の spec に基づいて [charts/operator-wandb](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb) をインストールします。設定仕様の階層は、ユーザー側での最大の設定柔軟性を可能にし、W&B が新しいイメージ、設定、機能、Helm の更新を自動的にリリースできるようにします。

設定オプションについては、[設定仕様の階層構造]({{< relref path="#configuration-specification-hierarchy" lang="ja" >}}) および [W&B オペレーターの設定リファレンス]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) を参照してください。

### 設定仕様の階層構造

設定仕様は、上位の仕様が下位のものを上書きする階層モデルに従います。仕組みは次のとおりです。

- **Release Channel Values**: この基本レベルの設定は、デプロイメントのために W&B によって設定されるリリースチャンネルに基づいてデフォルトの値と設定を設定します。
- **ユーザー入力値**: ユーザーは、Release Channel Spec によって提供されるデフォルト設定をシステムコンソールを通じて上書きすることができます。
- **カスタムリソース値**: ユーザーから提供される仕様の最高レベル。ここで指定された値は、ユーザー入力とリリースチャンネルの両方の仕様を上書きします。設定オプションの詳細については、[Configuration Reference]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) を参照してください。

この階層型モデルは、設定がフレキシブルでカスタマイズ可能であり、さまざまなニーズに対応できる一方で、アップグレードや変更に対して管理しやすく体系的なアプローチを維持します。

### W&B Kubernetes Operator を使用するための要件

W&B Kubernetes Operator を使用して W&B をデプロイするために次の要件を満たしてください:

[リファレンスアーキテクチャー]({{< relref path="../ref-arch.md#infrastructure-requirements" lang="ja" >}}) を参照してください。さらに、[有効な W&B サーバーライセンスを取得]({{< relref path="../#obtain-your-wb-server-license" lang="ja" >}}) してください。

詳細なセットアップおよび自己管理型インストールの設定方法については、[このガイド]({{< relref path="../bare-metal.md" lang="ja" >}}) を参照してください。

インストール方法によっては、次の要件を満たす必要があります:

* 正しい Kubernetes クラスタコンテキストでインストールされ、設定済みの Kubectl。
* Helm がインストールされていること。

### エアギャップインストール

Kubernetes を使用したエアギャップ環境での W&B のデプロイ方法については、[Deploy W&B in airgapped environment with Kubernetes]({{< relref path="./operator-airgapped.md" lang="ja" >}}) のチュートリアルを参照してください。

## W&B サーバーアプリケーションのデプロイ

このセクションでは、W&B Kubernetes Operator をデプロイするさまざまな方法について説明します。

{{% alert %}}
W&B Operator は W&B サーバーのデフォルトおよび推奨されるインストール方法です。
{{% /alert %}}

**次のいずれかを選択してください:**

- 必要な外部サービスすべてをプロビジョニングして W&B を Kubernetes に Helm CLI でデプロイしたい場合は、[こちらを続けてください]({{< relref path="#deploy-wb-with-helm-cli" lang="ja" >}})。
- インフラストラクチャーと W&B サーバーを Terraform で管理することを希望する場合は、[こちらを続けてください]({{< relref path="#deploy-wb-with-helm-terraform-module" lang="ja" >}})。
- W&B Cloud Terraform Modules を利用したい場合は、[こちらを続けてください]({{< relref path="#deploy-wb-with-wb-cloud-terraform-modules" lang="ja" >}})。

### Helm CLI で W&B をデプロイ

W&B は W&B Kubernetes オペレーターを Kubernetes クラスターにデプロイするための Helm Chart を提供します。このアプローチにより、W&B サーバーを Helm CLI または ArgoCD のような継続的デリバリーツールでデプロイできます。上記の要件が満たされていることを確認してください。

Helm CLI で W&B Kubernetes Operator をインストールする手順を以下に従ってください:

1. W&B Helm リポジトリを追加します。W&B Helm chart は W&B Helm リポジトリにあります。次のコマンドを使用してリポジトリを追加します:

helm repo add wandb https://charts.wandb.ai
helm repo update

2. Kubernetes クラスターに Operator をインストールします。次をコピーして貼り付けてください:

helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace

3. W&B オペレーターのカスタムリソースを設定して、W&B サーバーのインストールをトリガーします。この例の設定を `operator.yaml` という名前のファイルにコピーして、W&B デプロイメントをカスタマイズします。[Configuration Reference]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) を参照してください。

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

       # Ensure it's set to use your own MySQL
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

    カスタム設定でオペレーターを起動し、W&B サーバーアプリケーションをインストールおよび設定できます。

    kubectl apply -f operator.yaml

    デプロイメントが完了するまで待ちます。これには数分かかります。

5. Web UI を使用してインストールを確認するには、最初の管理者ユーザーアカウントを作成し、[インストールの確認]({{< relref path="#verify-the-installation" lang="ja" >}}) に記載された確認手順に従います。

### Helm Terraform Module で W&B をデプロイ

この方法は、特定の要件に合わせたカスタマイズされたデプロイメントを可能にし、Terraform のインフラストラクチャーコードとしてのアプローチを活用して一貫性と再現性を提供します。公式の W&B Helm ベースの Terraform Module は [ここ](https://registry.terraform.io/modules/wandb/wandb/helm/latest) にあります。

次のコードは始めるためのものとして使用できます。すべての必要な設定オプションを含むプロダクショングレードのデプロイメントを含んでいます。

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

設定オプションは [Configuration Reference]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) に記載されているものと同じであることに注意してください。ただし、構文は HashiCorp Configuration Language (HCL) に従わなければなりません。Terraform モジュールは W&B カスタムリソース定義 (CRD) を作成します。

W&B&Biases 自体が顧客向けに「Dedicated cloud」インストールをデプロイするために Helm Terraform モジュールを使用する方法を確認するには、以下のリンクに従ってください。

- [AWS](https://github.com/wandb/terraform-aws-wandb/blob/45e1d746f53e78e73e68f911a1f8cad5408e74b6/main.tf#L225)

- [Azure](https://github.com/wandb/terraform-azurerm-wandb/blob/170e03136b6b6fc758102d59dacda99768854045/main.tf#L155)

- [GCP](https://github.com/wandb/terraform-google-wandb/blob/49ddc3383df4cefc04337a2ae784f57ce2a2c699/main.tf#L189)

### W&B Cloud Terraform モジュールを使用した W&B のデプロイ

W&B は AWS、GCP、Azure 用の一連の Terraform Modules を提供しています。これらのモジュールは、Kubernetes クラスター、ロードバランサー、MySQL データベースなどを含むインフラストラクチャー全体をデプロイし、W&B サーバーアプリケーションを含めます。公式の W&B クラウド専用 Terraform Modules には、次のバージョンで W&B Kubernetes Operator がすでに組み込まれており、事前に設定されています。

| Terraform Registry                                                  | Source Code                                      | Version |
| ------------------------------------------------------------------- | ------------------------------------------------ | ------- |
| [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest) | https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| [Azure](https://github.com/wandb/terraform-azurerm-wandb)           | https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| [GCP](https://github.com/wandb/terraform-google-wandb)              | https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

このインテグレーションにより、最小限のセットアップで W&B Kubernetes Operator をインスタンスで使用する準備が整い、クラウド環境で W&B サーバーのデプロイと管理のための合理化されたパスが提供されます。

これらのモジュールの使用方法の詳細については、ドキュメントの自己管理型インストールセクションのこの[セクション]({{< relref path="../#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}}) を参照してください。

### インストールの確認

インストールを確認するには、W&B は [W&B CLI]({{< relref path="/ref/cli/" lang="ja" >}}) を使用することを推奨します。verify コマンドは、すべてのコンポーネントと設定を確認するいくつかのテストを実行します。

{{% alert %}}
この手順では、ブラウザで最初の管理者ユーザーアカウントを作成したことを前提としています。
{{% /alert %}}

インストールを確認するために以下の手順に従ってください:

1. W&B CLI をインストールします。

    pip install wandb

2. W&B にログインします。

    wandb login --host=https://YOUR_DNS_DOMAIN

    たとえば:
    ```shell
    wandb login --host=https://wandb.company-name.com
    ```

3. インストールを確認します:
    ```shell
    wandb verify
    ```

正常なインストールと完全に動作する W&B のデプロイメントは、次の出力を示します:

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

W&B Kubernetes オペレーターには管理コンソールが付属しています。これは `${HOST_URI}/console` にあります。たとえば、`https://wandb.company-name.com/console` です。

管理コンソールにログインする方法は2つあります:

{{< tabpane text=true >}}
{{% tab header="オプション1 (推奨)" value="option1" %}}
1. ブラウザで W&B アプリケーションを開き、ログインします。W&B アプリケーションには `${HOST_URI}/` でログインします。たとえば `https://wandb.company-name.com/` です。
2. コンソールにアクセスします。右上隅のアイコンをクリックし、**System console** をクリックします。管理権限のあるユーザーのみが**System console** エントリーを見ることができます。

    {{< img src="/images/hosting/access_system_console_via_main_app.png" alt="" >}}
{{% /tab %}}

{{% tab header="オプション2" value="option2"%}}
{{% alert %}}
オプション1でうまくいかない場合にのみ、以下の手順でコンソールにアクセスすることを W&B は推奨します。
{{% /alert %}}

1. ブラウザでコンソールアプリケーションを開きます。上記の説明に従った URL を開くと、ログイン画面にリダイレクトされます:
    {{< img src="/images/hosting/access_system_console_directly.png" alt="" >}}
2. インストールで生成された Kubernetes シークレットからパスワードを取得します:
    ```shell
    kubectl get secret wandb-password -o jsonpath='{.data.password}' | base64 -d
    ```
    パスワードをコピーします。
3. コンソールにログインします。コピーしたパスワードを貼り付け、**Login** をクリックします。
{{% /tab %}}
{{< /tabpane >}}

## W&B Kubernetes オペレーターの更新

このセクションでは、W&B Kubernetes オペレーターを更新する方法について説明します。

{{% alert %}}
* W&B Kubernetes オペレーターの更新は、W&B サーバーアプリケーションを更新するわけではありません。
* W&B Kubernetes オペレーターを使用しない Helm チャートを使用している場合、W&B オペレーターを更新するための手順に従う前に、[こちら]({{< relref path="#migrate-self-managed-instances-to-wb-operator" lang="ja" >}}) の説明を参照してください。
{{% /alert %}}

下記のコードスニペットをターミナルにコピーして貼り付けてください。

1. まず、[`helm repo update`](https://helm.sh/docs/helm/helm_repo_update/) でリポジトリを更新します:
    ```shell
    helm repo update
    ```

2. 次に、[`helm upgrade`](https://helm.sh/docs/helm/helm_upgrade/) を使用して Helm チャートを更新します:
    ```shell
    helm upgrade operator wandb/operator -n wandb-cr --reuse-values
    ```

## W&B サーバーアプリケーションの更新

W&B Kubernetes オペレーターを使用している場合、W&B サーバーアプリケーションを更新する必要はありません。

オペレーターは、新しいバージョンのソフトウェアがリリースされると、W&B サーバーアプリケーションを自動的に更新します。

## 独自管理インスタンスを W&B オペレーターに移行する

このセクションでは、独自に W&B サーバーインストールを管理する方法から、W&B オペレーターを使用してそれを行う方法に移行する方法を説明します。移行プロセスは W&B サーバーをインストールした方法によって異なります。

{{% alert %}}
W&B オペレーターは W&B サーバーのデフォルトおよび推奨されるインストール方法です。ご質問がある場合は、[カスタマーサポート](mailto:support@wandb.com) または W&B チームにお問い合わせください。
{{% /alert %}}

- 公式の W&B Cloud Terraform Modules を使用した場合、適切なドキュメントに移動してそこに記載されている手順に従ってください:
  - [AWS]({{< relref path="#migrate-to-operator-based-aws-terraform-modules" lang="ja" >}})
  - [GCP]({{< relref path="#migrate-to-operator-based-gcp-terraform-modules" lang="ja" >}})
  - [Azure]({{< relref path="#migrate-to-operator-based-azure-terraform-modules" lang="ja" >}})
- [W&B Non-Operator Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/wandb) を使用した場合、[こちらを続けてください]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ja" >}})。
- [W&B Non-Operator Helm chart with Terraform](https://registry.terraform.io/modules/wandb/wandb/kubernetes/latest) を使用した場合、[こちらを続けてください]({{< relref path="#migrate-to-operator-based-terraform-helm-chart" lang="ja" >}})。
- マニフェストを使用して Kubernetes リソースを作成した場合、[こちらを続けてください]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ja" >}})。

### オペレーターをベースとした AWS Terraform モジュールへの移行

移行プロセスの詳細な説明については、[こちら]({{< relref path="../install-on-public-cloud/aws-tf.md#migrate-to-operator-based-aws-terraform-modules" lang="ja" >}}) を参照してください。

### オペレーターをベースとした GCP Terraform モジュールへの移行

ご質問や支援が必要な場合は、[カスタマーサポート](mailto:support@wandb.com) または W&B チームにお問い合わせください。

### オペレーターをベースとした Azure Terraform モジュールへの移行

ご質問や支援が必要な場合は、[カスタマーサポート](mailto:support@wandb.com) または W&B チームにお問い合わせください。

### オペレーターをベースとした Helm チャートへの移行

オペレーターをベースとした Helm チャートへの移行手順に従ってください:

1. 現在の W&B 設定を取得します。非オペレーター基盤の Helm チャートで W&B がデプロイされた場合、次のように値をエクスポートします:
    ```shell
    helm get values wandb
    ```
    Kubernetes マニフェストで W&B をデプロイした場合、次のように値をエクスポートします:
    ```shell
    kubectl get deployment wandb -o yaml
    ```
    次のステップに必要なすべての設定値が得られました。

2. `operator.yaml` というファイルを作成します。[Configuration Reference]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) に記載されている形式に従ってください。ステップ1からの値を使用します。

3. 現在のデプロイメントを 0 ポッドにスケールします。このステップは現在のデプロイメントを停止します。
    ```shell
    kubectl scale --replicas=0 deployment wandb
    ```
4. Helm チャートリポジトリを更新します:
    ```shell
    helm repo update
    ```
5. 新しい Helm チャートをインストールします:
    ```shell
    helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
    ```
6. 新しい Helm チャートを設定して W&B アプリケーションのデプロイをトリガーします。新しい設定を適用します。
    ```shell
    kubectl apply -f operator.yaml
    ```
    デプロイメントが完了するまで数分かかります。

7. インストールを確認します。[インストールの確認]({{< relref path="#verify-the-installation" lang="ja" >}}) のステップに従って、すべてが機能することを確認してください。

8. 古いインストールを削除します。古い Helm チャートをアンインストールするか、マニフェストで作成されたリソースを削除します。

### オペレーターをベースとした Terraform Helm チャートへの移行

これらのステップに従ってオペレーターをベースとした Helm チャートに移行します:

1. Terraform の設定を準備します。古いデプロイメントのために Terraform 設定にある Terraform コードを、[ここに記載されている]({{< relref path="#deploy-wb-with-helm-terraform-module" lang="ja" >}}) ものと置き換えます。以前と同じ変数を設定します。tfvars ファイルを変更しないでください。
2. Terraform run を実行します。terraform init、plan、apply を実行します
3. インストールを確認します。[インストールの確認]({{< relref path="#verify-the-installation" lang="ja" >}}) のステップに従って、すべてが機能することを確認してください。
4. 古いインストールを削除します。古い Helm チャートをアンインストールするか、マニフェストで作成されたリソースを削除します。

## W&B サーバーの設定リファレンス

このセクションでは W&B サーバーアプリケーションの設定オプションについて説明します。アプリケーションは [WeightsAndBiases]({{< relref path="#how-it-works" lang="ja" >}}) という名前のカスタムリソース定義として設定を受け取ります。一部の設定オプションは以下の構成で公開されており、一部は環境変数として設定する必要があります。

ドキュメントには 2 つの環境変数のリストがあります: [基本]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}) と [高度]({{< relref path="/guides/hosting/iam/advanced_env_vars/" lang="ja" >}})。Helm Chart を使用して必要な設定オプションが公開されていない場合にのみ、環境変数を使用してください。

プロダクションデプロイメントのための W&B サーバーアプリケーション設定ファイルには以下の内容が必要です。この YAML ファイルは、W&B デプロイメントの望ましい状態を定義し、バージョン、環境変数、外部リソース（データベースなど）、および他の必要な設定を含みます。

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

変更が必要な値だけを上書きするために、[W&B Helm repository](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml) の完全な set を見つけてください。