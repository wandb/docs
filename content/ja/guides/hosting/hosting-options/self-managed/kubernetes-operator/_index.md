---
title: Run W&B Server on Kubernetes
description: Kubernetes Operator を使用して W&B Platform をデプロイする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-kubernetes-operator-_index
    parent: self-managed
url: guides/hosting/operator
weight: 2
---

## W&B Kubernetes Operator

W&B Kubernetes Operatorを使用すると、Kubernetes上でのW&B Serverのデプロイ、管理、トラブルシューティング、およびスケーリングを簡素化できます。このOperatorは、W&Bインスタンスのスマートアシスタントとして考えることができます。

W&B Serverのアーキテクチャと設計は、AI開発者向け ツール の機能を拡張し、高パフォーマンス、優れたスケーラビリティ、および容易な管理のための適切なプリミティブを提供するために、継続的に進化しています。この進化は、コンピューティングサービス、関連するストレージ、およびそれらの間の接続に適用されます。デプロイメントタイプ全体での継続的な更新と改善を促進するために、W&B は Kubernetes operator を使用します。

{{% alert %}}
W&B は、この operator を使用して、AWS、GCP、および Azure パブリック クラウド上に Dedicated cloud インスタンスをデプロイおよび管理します。
{{% /alert %}}

Kubernetes operator の詳細については、Kubernetesドキュメントの[Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)を参照してください。

### アーキテクチャ移行の理由
従来、W&B アプリケーションは、Kubernetes クラスター内の単一のデプロイメントおよび pod として、または単一の Docker コンテナとしてデプロイされていました。W&B は、データベースと Object Store を外部化することを推奨しており、今後も推奨していきます。データベースと Object Store を外部化すると、アプリケーションの状態が分離されます。

アプリケーションの成長に伴い、モノリシックなコンテナから分散システム（マイクロサービス）に進化する必要性が明らかになりました。この変更により、バックエンドロジックの処理が容易になり、Kubernetes インフラストラクチャの機能がシームレスに組み込まれます。分散システムは、W&B が依存する追加機能に不可欠な新しいサービスのデプロイもサポートします。

2024年以前は、Kubernetes関連の変更を行うには、[terraform-kubernetes-wandb](https://github.com/wandb/terraform-kubernetes-wandb) Terraformモジュールを手動で更新する必要がありました。Terraformモジュールを更新することで、クラウドプロバイダー間での互換性が確保され、必要なTerraform変数が構成され、バックエンドまたはKubernetesレベルの変更ごとにTerraformが適用されます。

このプロセスは、W&Bサポートが各顧客のTerraformモジュールのアップグレードを支援する必要があったため、スケーラブルではありませんでした。

解決策は、中央の[deploy.wandb.ai](https://deploy.wandb.ai)サーバーに接続して、特定のリリースチャネルの最新の仕様変更をリクエストし、適用する operator を実装することでした。ライセンスが有効である限り、更新が受信されます。[Helm](https://helm.sh/)は、W&B operator のデプロイメントメカニズムと、W&B Kubernetesスタックのすべての構成テンプレートを operator が処理する手段の両方として使用されます（Helm-ception）。

### 仕組み
Operator は、helm またはソースからインストールできます。詳細な手順については、[charts/operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)を参照してください。

インストールプロセスでは、`controller-manager`というデプロイメントが作成され、`weightsandbiases.apps.wandb.com`という[カスタムリソース](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)定義（shortName：`wandb`）が使用されます。これは単一の`spec`を取得し、それをクラスターに適用します。

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: weightsandbiases.apps.wandb.com
```

`controller-manager`は、カスタムリソースの仕様、リリースチャネル、およびユーザー定義の構成に基づいて、[charts/operator-wandb](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)をインストールします。構成仕様の階層により、ユーザー側で最大限の構成の柔軟性が実現し、W&B は新しいイメージ、構成、機能、および Helm の更新を自動的にリリースできます。

構成オプションについては、[構成仕様の階層]({{< relref path="#configuration-specification-hierarchy" lang="ja" >}})および[構成リファレンス]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}})を参照してください。

### 構成仕様の階層
構成仕様は、高レベルの仕様が低レベルの仕様をオーバーライドする階層モデルに従います。その仕組みは次のとおりです。

- **リリースチャネルの値**: この基本レベルの構成では、W&B によってデプロイメント用に設定されたリリースチャネルに基づいて、デフォルト値と構成が設定されます。
- **ユーザー入力値**: ユーザーは、システムコンソールを介して、リリースチャネルの仕様によって提供されるデフォルト設定をオーバーライドできます。
- **カスタムリソースの値**: ユーザーからの仕様の最上位レベル。ここで指定された値は、ユーザー入力とリリースチャネルの両方の仕様をオーバーライドします。構成オプションの詳細については、[構成リファレンス]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}})を参照してください。

この階層モデルにより、構成は柔軟で、さまざまなニーズに合わせてカスタマイズでき、アップグレードと変更に対する管理可能で体系的なアプローチを維持できます。

### W&B Kubernetes Operatorを使用するための要件
W&B Kubernetes operator で W&B をデプロイするには、次の要件を満たす必要があります。

[リファレンスアーキテクチャ]({{< relref path="../ref-arch.md#infrastructure-requirements" lang="ja" >}})を参照してください。さらに、[有効な W&B Server ライセンスを取得]({{< relref path="../#obtain-your-wb-server-license" lang="ja" >}})してください。

自己管理型インストールを設定および構成する方法の詳細な説明については、[こちら]({{< relref path="../bare-metal.md" lang="ja" >}})のガイドを参照してください。

インストール方法によっては、次の要件を満たす必要がある場合があります。
* Kubectl がインストールされ、正しい Kubernetes クラスターコンテキストで構成されている。
* Helm がインストールされている。

### エアギャップ環境へのインストール
エアギャップ環境に W&B Kubernetes Operator をインストールする方法については、[Kubernetes を使用したエアギャップ環境での W&B のデプロイ]({{< relref path="operator-airgapped.md" lang="ja" >}})のチュートリアルを参照してください。

## W&B Server アプリケーションのデプロイ
このセクションでは、W&B Kubernetes operator をデプロイするさまざまな方法について説明します。
{{% alert %}}
W&B Operator は、W&B Server のデフォルトであり、推奨されるインストール方法です。
{{% /alert %}}

**次のいずれかを選択してください。**
- 必要な外部サービスをすべてプロビジョニングし、Helm CLI を使用して W&B を Kubernetes にデプロイする場合は、[こちら]({{< relref path="#deploy-wb-with-helm-cli" lang="ja" >}})に進んでください。
- インフラストラクチャと W&B Server を Terraform で管理する場合は、[こちら]({{< relref path="#deploy-wb-with-helm-terraform-module" lang="ja" >}})に進んでください。
- W&B Cloud Terraform Modules を利用する場合は、[こちら]({{< relref path="#deploy-wb-with-wb-cloud-terraform-modules" lang="ja" >}})に進んでください。

### Helm CLI を使用した W&B のデプロイ
W&B は、W&B Kubernetes operator を Kubernetes クラスターにデプロイするための Helm Chart を提供します。このアプローチを使用すると、Helm CLI または ArgoCD などの継続的デリバリー ツールで W&B Server をデプロイできます。上記の要件が満たされていることを確認してください。

Helm CLI を使用して W&B Kubernetes Operator をインストールするには、次の手順に従います。

1. W&B Helm リポジトリを追加します。W&B Helm チャートは、W&B Helm リポジトリで入手できます。次のコマンドを使用してリポジトリを追加します。
```shell
helm repo add wandb https://charts.wandb.ai
helm repo update
```
2. Kubernetes クラスターに Operator をインストールします。以下をコピーして貼り付けます。
```shell
helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
```
3. W&B Server のインストールをトリガーするように W&B operator カスタムリソースを構成します。この構成例を `operator.yaml` というファイルにコピーして、W&B デプロイメントをカスタマイズできるようにします。[構成リファレンス]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}})を参照してください。

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
   ```

    W&B Server アプリケーションをインストールおよび構成できるように、カスタム構成で Operator を起動します。

    ```shell
    kubectl apply -f operator.yaml
    ```

    デプロイメントが完了するまで待ちます。これには数分かかります。

5. Web UI を使用してインストールを検証するには、最初 の 管理 ユーザー アカウントを作成し、[インストールの検証]({{< relref path="#verify-the-installation" lang="ja" >}})に概説されている検証手順に従います。


### Helm Terraform モジュールを使用した W&B のデプロイ

この方法では、Terraform の Infrastructure-as-Code アプローチを活用して、一貫性と再現性を実現し、特定の要件に合わせてカスタマイズされたデプロイメントが可能です。公式の W&B Helm ベースの Terraform モジュールは、[こちら](https://registry.terraform.io/modules/wandb/wandb/helm/latest)にあります。

次のコードは、開始点として使用でき、本番環境グレードのデプロイメントに必要なすべての構成オプションが含まれています。

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

構成オプションは[構成リファレンス]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}})で説明されているものと同じですが、構文は HashiCorp Configuration Language（HCL）に従う必要があることに注意してください。Terraform モジュールは、W&B カスタムリソース定義（CRD）を作成します。

W&B&Biases 自体が Helm Terraform モジュールを使用して顧客向けの「Dedicated cloud」インストールをデプロイする方法については、次のリンクを参照してください。
- [AWS](https://github.com/wandb/terraform-aws-wandb/blob/45e1d746f53e78e73e68f911a1f8cad5408e74b6/main.tf#L225)
- [Azure](https://github.com/wandb/terraform-azurerm-wandb/blob/170e03136b6b6fc758102d59dacda99768854045/main.tf#L155)
- [GCP](https://github.com/wandb/terraform-google-wandb/blob/49ddc3383df4cefc04337a2ae784f57ce2a2c699/main.tf#L189)

### W&B Cloud Terraform モジュールを使用した W&B のデプロイ

W&B は、AWS、GCP、および Azure 用の一連の Terraform モジュールを提供します。これらのモジュールは、Kubernetes クラスター、ロードバランサー、MySQL データベースなど、インフラストラクチャ全体と W&B Server アプリケーションをデプロイします。W&B Kubernetes Operator は、次のバージョンの公式 W&B クラウド固有の Terraform モジュールですでに事前構築されています。

| Terraform Registry                                                  | Source Code                                      | Version |
| ------------------------------------------------------------------- | ------------------------------------------------ | ------- |
| [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest) | https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| [Azure](https://registry.terraform.io/modules/wandb/terraform-azurerm-wandb)           | https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| [GCP](https://github.com/wandb/terraform-google-wandb)              | https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

この統合により、最小限のセットアップでインスタンスに W&B Kubernetes Operator を使用する準備が整い、クラウド環境での W&B Server のデプロイと管理への合理化されたパスが提供されます。

これらのモジュールの使用方法の詳細については、ドキュメントの自己管理型インストールセクションの[このセクション]({{< relref path="../#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}})を参照してください。

### インストールの検証

インストールを検証するために、W&B は[W&B CLI]({{< relref path="/ref/cli/" lang="ja" >}})を使用することをお勧めします。verify コマンドは、すべてのコンポーネントと構成を検証するいくつかのテストを実行します。

{{% alert %}}
この手順では、最初の 管理 ユーザー アカウントがブラウザーで作成されていることを前提としています。
{{% /alert %}}

インストールを検証するには、次の手順に従います。

1. W&B CLI をインストールします。
    ```shell
    pip install wandb
    ```
2. W&B にログインします。
    ```shell
    wandb login --host=https://YOUR_DNS_DOMAIN
    ```

    例：
    ```shell
    wandb login --host=https://wandb.company-name.com
    ```

3. インストールを検証します。
    ```shell
    wandb verify
    ```

インストールが成功し、完全に動作する W&B デプロイメントでは、次の出力が表示されます。

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
W&B Kubernetes operator には、管理コンソールが付属しています。これは`${HOST_URI}/console`にあります。たとえば、`https://wandb.company-name.com/` console などです。

管理コンソールにログインするには、次の2つの方法があります。

{{< tabpane text=true >}}
{{% tab header="オプション 1 (推奨)" value="option1" %}}
1. ブラウザで W&B アプリケーションを開き、ログインします。`${HOST_URI}/`で W&B アプリケーションにログインします。たとえば、`https://wandb.company-name.com/`
2. コンソールにアクセスします。右上隅のアイコンをクリックし、次に**システムコンソール**をクリックします。管理者権限を持つユーザーのみが**システムコンソール**エントリを表示できます。

    {{< img src="/images/hosting/access_system_console_via_main_app.png" alt="" >}}
{{% /tab %}}

{{% tab header="オプション 2" value="option2"%}}
{{% alert %}}
オプション 1 が機能しない場合にのみ、次の手順を使用してコンソールにアクセスすることをお勧めします。
{{% /alert %}}

1. ブラウザでコンソールアプリケーションを開きます。上記の URL を開き、ログイン画面にリダイレクトします。
    {{< img src="/images/hosting/access_system_console_directly.png" alt="" >}}
2. インストールによって生成される Kubernetes シークレットからパスワードを取得します。
    ```shell
    kubectl get secret wandb-password -o jsonpath='{.data.password}' | base64 -d
    ```
    パスワードをコピーします。
3. コンソールにログインします。コピーしたパスワードを貼り付け、次に**ログイン**をクリックします。
{{% /tab %}}
{{< /tabpane >}}

## W&B Kubernetes operator の更新
このセクションでは、W&B Kubernetes operator を更新する方法について説明します。

{{% alert %}}
* W&B Kubernetes operator を更新しても、W&B サーバーアプリケーションは更新されません。
* W&B operator を更新する手順に進む前に、W&B Kubernetes operator を使用しない Helm チャートを使用している場合は、[こちら]({{< relref path="#migrate-self-managed-instances-to-wb-operator" lang="ja" >}})の手順を参照してください。
{{% /alert %}}

以下のコードスニペットをコピーしてターミナルに貼り付けます。

1. まず、[`helm repo update`](https://helm.sh/docs/helm/helm_repo_update/)でリポジトリを更新します。
    ```shell
    helm repo update
    ```

2. 次に、[`helm upgrade`](https://helm.sh/docs/helm/helm_upgrade/)で Helm チャートを更新します。
    ```shell
    helm upgrade operator wandb/operator -n wandb-cr --reuse-values
    ```

## W&B Server アプリケーションの更新
W&B Kubernetes operator を使用する場合は、W&B Server アプリケーションを更新する必要はなくなりました。

operator は、W&B のソフトウェアの新しいバージョンがリリースされると、W&B Server アプリケーションを自動的に更新します。

## 自己管理型インスタンスの W&B Operator への移行
次のセクションでは、独自の W&B Server インストールを自己管理することから、W&B Operator を使用してこれを行うように移行する方法について説明します。移行プロセスは、W&B Server のインストール方法によって異なります。

{{% alert %}}
W&B Operator は、W&B Server のデフォルトであり、推奨されるインストール方法です。ご不明な点がございましたら、[カスタマーサポート](mailto:support@wandb.com)または W&B チームにお問い合わせください。
{{% /alert %}}

- 公式の W&B Cloud Terraform Modules を使用した場合は、適切なドキュメントに移動し、そこに記載されている手順に従ってください。
  - [AWS]({{< relref path="#migrate-to-operator-based-aws-terraform-modules" lang="ja" >}})
  - [GCP]({{< relref path="#migrate-to-operator-based-gcp-terraform-modules" lang="ja" >}})
  - [Azure]({{< relref path="#migrate-to-operator-based-azure-terraform-modules" lang="ja" >}})
- [W&B Non-Operator Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/wandb) を使用した場合は、[こちら]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ja" >}})に進んでください。
- [Terraform を使用した W&B Non-Operator Helm chart](https://registry.terraform.io/modules/wandb/wandb/kubernetes/latest) を使用した場合は、[こちら]({{< relref path="#migrate-to-operator-based-terraform-helm-chart" lang="ja" >}})に進んでください。
- マニフェストを使用して Kubernetes リソースを作成した場合は、[こちら]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ja" >}})に進んでください。

### Operator ベースの AWS Terraform Modules への移行

移行プロセスの詳細については、[こちら]({{< relref path="../install-on-public-cloud/aws-tf.md#migrate-to-operator-based-aws-terraform-modules" lang="ja" >}})に進んでください。

### Operator ベースの GCP Terraform Modules への移行

ご不明な点がある場合や、サポートが必要な場合は、[カスタマーサポート](mailto:support@wandb.com)または W&B チームにお問い合わせください。

### Operator ベースの Azure Terraform Modules への移行

ご不明な点がある場合や、サポートが必要な場合は、[カスタマーサポート](mailto:support@wandb.com)または W&B チームにお問い合わせください。

### Operator ベースの Helm チャートへの移行

Operator ベースの Helm チャートに移行するには、次の手順に従います。

1. 現在の W&B 構成を取得します。W&B が非 operator ベースのバージョンの Helm チャートでデプロイされた場合は、次のように値をエクスポートします。
    ```shell
    helm get values wandb
    ```
    W&B が Kubernetes マニフェストでデプロイされた場合は、次のように値をエクスポートします。
    ```shell
    kubectl get deployment wandb -o yaml
    ```
    これで、次のステップに必要なすべての構成値が揃いました。

2. `operator.yaml` というファイルを作成します。[構成リファレンス]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}})で説明されている形式に従ってください。ステップ 1 の値を使用します。

3. 現在のデプロイメントを 0 pod にスケールします。このステップでは、現在のデプロイメントを停止します。
    ```shell
    kubectl scale --replicas=0 deployment wandb
    ```
4. Helm チャートリポジトリを更新します。
    ```shell
    helm repo update
    ```
5. 新しい Helm チャートをインストールします。
    ```shell
    helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
    ```
6. 新しい Helm チャートを構成し、W&B アプリケーションのデプロイメントをトリガーします。新しい構成を適用します。
    ```shell
    kubectl apply -f operator.yaml
    ```
    デプロイメントが完了するまでに数分かかります。

7. インストールを検証します。[インストールの検証]({{< relref path="#verify-the-installation" lang="ja" >}})の手順に従って、すべてが正常に動作することを確認します。

8. 古いインストールを削除します。古い Helm チャートをアンインストールするか、マニフェストで作成されたリソースを削除します。

### Operator ベースの Terraform Helm チャートへの移行

Operator ベースの Helm チャートに移行するには、次の手順に従います。

1. Terraform 構成を準備します。Terraform 構成で古いデプロイメントの Terraform コードを[こちら]({{< relref path="#deploy-wb-with-helm-terraform-module" lang="ja" >}})で説明されているコードに置き換えます。以前と同じ変数を設定します。tfvars ファイルがある場合は変更しないでください。
2. Terraform run を実行します。terraform init、plan、および apply を実行します。
3. インストールを検証します。[インストールの検証]({{< relref path="#verify-the-installation" lang="ja" >}})の手順に従って、すべてが正常に動作することを確認します。
4. 古いインストールを削除します。古い Helm チャートをアンインストールするか、マニフェストで作成されたリソースを削除します。

## W&B Server の構成リファレンス

このセクションでは、W&B Server アプリケーションの構成オプションについて説明します。アプリケーションは、[WeightsAndBiases]({{< relref path="#how-it-works" lang="ja" >}})というカスタムリソース定義として構成を受け取ります。一部の構成オプションは以下の構成で公開され、一部は環境変数として設定する必要があります。

ドキュメントには、[基本]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}})および[詳細]({{< relref path="/guides/hosting/iam/advanced_env_vars/" lang="ja" >}})の2つの環境変数リストがあります。Helm Chart を使用して必要な構成オプションが公開されていない場合にのみ、環境変数を使用してください。

本番環境デプロイメント用の W&B Server アプリケーション構成ファイルには、次の内容が必要です。この YAML ファイルは、バージョン、環境変数、データベースなどの外部リソース、およびその他の必要な設定を含む、W&B デプロイメントの目的の状態を定義します。

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

[W&B Helm リポジトリ](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml)で値の完全なセットを見つけ、オーバーライドする必要がある値のみを変更します。

### 完全な例
これは、GCP Ingress と GCS（GCP Object storage）を備えた GCP Kubernetes を使用する構成例です。

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

### ホスト
```yaml
 # プロトコルを含む FQDN を指定します
global:
  # ホスト名の例、独自のものに置き換えます
  host: https://wandb.example.com
```

### オブジェクトストレージ (バケット)

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

**その他のプロバイダー (Minio、Ceph など)**

その他の S3 互換プロバイダーの場合は、次のようにバケット構成を設定します。
```yaml
global:
  bucket:
    # 値の例、独自のものに置き換えます
    provider: s3
    name: storage.example.com
    kmsKey: null
    path: wandb
    region: default
    accessKey: 5WOA500...P5DK7I
    secretKey: HDKYe4Q...JAp1YyjysnX
```

AWS の外部でホストされている S3 互換ストレージの場合、`kmsKey` は `null` である必要があります。

シークレットから `accessKey` と `secretKey` を参照するには:
```yaml
global:
  bucket:
    # 値の例、独自のものに置き換えます
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
     # 値の例、独自のものに置き換えます
     host: db.example.com
     port: 3306
     database: wandb_local
     user: wandb
     password: 8wtX6cJH...ZcUarK4zZGjpV
```

シークレットから `password` を参照するには:
```yaml
global:
   mysql:
     # 値の例、独自のものに置き換えます
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
  # ライセンスの例、独自のものに置き換えます
  license: eyJhbGnUzaHgyQjQy...VFnPS_KETXg1hi
```

シークレットから `license` を参照するには:
```yaml
global:
  licenseSecret:
    name: license-secret
    key: CUSTOMER_WANDB_LICENSE
```

### Ingress

Ingress クラスを特定するには、この FAQ[エントリ]({{< relref path="#how-to-identify-the-kubernetes-ingress-class" lang="ja" >}})を参照してください。

**TLS なし**

```yaml
global:
# 重要: Ingress は YAML で ‘global’ と同じレベルにあります (子ではありません)
ingress:
  class: ""
```

**TLS あり**

証明書を含むシークレットを作成します

```console
kubectl create secret tls wandb-ingress-tls --key wandb-ingress-tls.key --cert wandb-ingress-tls.crt
```

Ingress 構成でシークレットを参照します
```yaml
global:
# 重要: Ingress は YAML で ‘global’ と同じレベルにあります (子ではありません)
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

Nginx の場合は、次の注釈を追加する必要がある場合があります。

```
ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: 64m
```

### カスタム Kubernetes ServiceAccount

カスタム Kubernetes サービスアカウントを指定して、W&B pod を実行します。

次のスニペットは、指定された名前でデプロイメントの一部としてサービスアカウントを作成します。

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
サブシステム "app" と "parquet" は、指定されたサービスアカウントで実行されます。他のサブシステムは、デフォルトのサービスアカウントで実行されます。

サービスアカウントがクラスターにすでに存在する場合は、`create: false` を設定します。

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

app、parquet、console など、さまざまなサブシステムでサービスアカウントを指定できます。

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

サービスアカウントは、サブシステム間で異なる場合があります。

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

シークレットから `password` を参照するには:

```console
kubectl create secret generic redis-secret --from-literal=redis-password=supersecret
```

以下の構成で参照します。
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
    # ユーザーの検索に使用する LDAP 検索ベース
    baseDN:
    # バインドする LDAP ユーザー (匿名バインドを使用しない場合)
    bindDN:
    # バインドする LDAP パスワードを含むシークレット名とキー (匿名バインドを使用しない場合)
    bindPW:
    # 電子メールとグループ ID 属性の LDAP 属性名をコンマ区切りの文字列値として指定します。
    attributes:
    # LDAP グループ許可リスト
    groupAllowList:
    # LDAP TLS を有効にする
    tls: false
```

**TLS あり**

LDAP TLS 証明書構成には、証明書コンテンツで事前に作成された構成マップが必要です。

構成マップを作成するには、次のコマンドを使用します。

```console
kubectl create configmap ldap-tls-cert --from-file=certificate.crt
```

次の例のように、YAML で構成マップを使用します。

```yaml
global:
  ldap:
    enabled: true
    # "ldap://" または "ldaps://" を含む LDAP サーバーアドレス
    host:
    # ユーザーの検索に使用する LDAP 検索ベース
    baseDN:
    # バインドする LDAP ユーザー (匿名バインドを使用しない場合)
    bindDN:
    # バインドする LDAP パスワードを含むシークレット名とキー (匿名バインドを使用しない場合)
    bindPW:
    # 電子メールとグループ ID 属性の LDAP 属性名をコンマ区切りの文字列値として指定します。
    attributes:
    # LDAP グループ許可リスト
    groupAllowList:
    # LDAP TLS を有効にする
    tls: true
    # LDAP サーバーの CA 証明書を含む ConfigMap 名とキー
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
      # IdP が必要な場合にのみ含めます。
      authMethod: ""
      issuer: ""
```

`authMethod` はオプションです。

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

### カスタム認証局
`customCACerts` はリストであり、多くの証明書を使用できます。`customCACerts` で指定された認証局は、W&B Server アプリケーションにのみ適用されます。

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

CA 証明書は、ConfigMap にも保存できます。
```yaml
global:
  caCertsConfigMap: custom-ca-certs
```

ConfigMap は次のようになります。
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
ConfigMap を使用する場合、ConfigMap の各キーは `.crt` で終わる必要があります (例: `my-cert.crt` または `ca-cert1.crt`)。この命名規則は、`update-ca-certificates` が各証明書を解析してシステムの CA ストアに追加するために必要です。
{{% /alert %}}

### カスタムセキュリティコンテキスト

各 W&B コンポーネントは、次の形式のカスタムセキュリティコンテキスト構成をサポートしています。

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
`runAsGroup:` の有効な値は `0` のみです。他の値はエラーです。
{{% /alert %}}

たとえば、アプリケーション pod を構成するには、構成にセクション `app` を追加します。

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

同じ概念が、`console`、`weave`、`weave-trace`、および `parquet` にも適用されます。

## W&B Operator の構成リファレンス

このセクションでは、W&B Kubernetes operator（`wandb-controller-manager`）の構成オプションについて説明します。operator は、構成を YAML ファイルの形式で受け取ります。

デフォルトでは、W&B Kubernetes operator に構成ファイルは必要ありません。必要な場合は、構成ファイルを作成します。たとえば、カスタム認証局を指定したり、エアギャップ環境にデプロイしたりするために、構成ファイルが必要になる場合があります。

仕様のカスタマイズの完全なリストについては、[Helm リポジトリ](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)を参照してください。

### カスタム CA
カスタム認証局（`customCACerts`）はリストであり、多くの証明書を使用できます。これらの認証局を追加すると、W&B Kubernetes operator（`wandb-controller-manager`）にのみ適用されます。

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

CA 証明書は、ConfigMap にも保存できます。
```yaml
caCertsConfigMap: custom-ca-certs
```

ConfigMap は次のようになります。
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
ConfigMap の各キーは `.crt` で終わる必要があります (例: `my-cert.crt` または `ca-cert1.crt`)。この命名規則は、`update-ca-certificates` が各証明書を解析してシステムの CA ストアに追加するために必要です。
{{% /alert %}}

## FAQ

### 個々の pod の目的/役割は何ですか?
* **`wandb-app`**: W&B のコア。GraphQL API とフロントエンドアプリケーションが含まれています。プラットフォームの機能のほとんどを強化します。
* **`wandb-console`**: 管理コンソール。`/console` 経由でアクセスします。
* **`wandb-otel`**: OpenTelemetry エージェント。Kubernetes レイヤーのリソースからメトリクスとログを収集し、管理コンソールに表示します。
* **`wandb-prometheus`**: Prometheus サーバー。さまざまなコンポーネントからメトリクスを取得し、管理コンソールに表示します。
* **`wandb-parquet`**: `wandb-app` pod とは別のバックエンドマイクロサービス。データベースデータを Parquet 形式でオブジェクトストレージにエクスポートします。
* **`wandb-weave`**: UI でクエリテーブルをロードし、さまざまなコアアプリ機能をサポートする別のバックエンドマイクロサービス。
* **`wandb-weave-trace`**: LLM ベースのアプリケーションを追跡、実験、評価、デプロイ、および改善するためのフレームワーク。フレームワークは `wandb-app` pod 経由でアクセスします。

### W&B Operator Console のパスワードを取得する方法
[W&B Kubernetes Operator 管理コンソールへのアクセス]({{< relref path="#access-the-wb-management-console" lang="ja" >}})を参照してください。

### Ingress が機能しない場合に W&B Operator Console にアクセスする方法

Kubernetes クラスターに到達できるホストで、次のコマンドを実行します。

```console
kubectl port-forward svc/wandb-console 8082
```

ブラウザで `https://localhost:8082/` console を使用してコンソールにアクセスします。

パスワードの取得方法については、[W&B Kubernetes Operator 管理コンソールへのアクセス]({{< relref path="#access-the-wb-management-console" lang="ja" >}})（オプション 2）を参照してください。

### W&B Server のログを表示する方法

アプリケーション pod の名前は **wandb-app-xxx** です。

```console
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

### Kubernetes Ingress クラスを識別する方法

次のコマンドを実行して、クラスターにインストールされている Ingress クラスを取得できます。

```console
kubectl get ingressclass
```
```