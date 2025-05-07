---
title: W&B サーバーを Kubernetes で実行する
description: W&B プラットフォーム を Kubernetes Operator でデプロイする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-kubernetes-operator-_index
    parent: self-managed
url: /ja/guides/hosting/operator
weight: 2
---

## W&B Kubernetes オペレーター

W&B Kubernetes オペレーターを使用して、Kubernetes 上の W&B Server デプロイメントを展開、管理、トラブルシューティング、およびスケーリングを簡素化します。このオペレーターは、W&B インスタンス用のスマートアシスタントと考えることができます。

W&B Server のアーキテクチャと設計は、AI 開発者のツール提供能力を拡張し、高性能でより優れたスケーラビリティと簡易な管理を提供するために進化し続けています。この進化は、コンピューティングサービス、関連ストレージ、およびそれらの接続性に適用されます。デプロイメントタイプ全体での継続的な更新と改善を促進するために、W&B は Kubernetes オペレーターを使用しています。

{{% alert %}}
W&B はオペレーターを使用して、AWS、GCP、および Azure のパブリッククラウド上で専用クラウドインスタンスをデプロイおよび管理します。
{{% /alert %}}

Kubernetes オペレーターに関する詳細情報は、Kubernetes のドキュメントにある[オペレーターパターン](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)を参照してください。

### アーキテクチャの変更理由

歴史的に、W&B アプリケーションは Kubernetes クラスター内の単一デプロイメントおよびポッド、または単一の Docker コンテナとしてデプロイされていました。W&B は引き続き、データベースおよびオブジェクトストアを外部化することを推奨しています。データベースとオブジェクトストアの外部化は、アプリケーションの状態を切り離します。

アプリケーションが成長するにつれて、モノリシックコンテナから分散システム（マイクロサービス）へ進化するニーズが明らかになりました。この変更はバックエンドロジックの処理を容易にし、組み込みの Kubernetes インフラストラクチャ能力をスムーズに導入します。分散システムはまた、新しいサービスの展開をサポートし、W&B が依存する追加の機能を提供します。

2024年以前、Kubernetes 関連の変更は、[terraform-kubernetes-wandb](https://github.com/wandb/terraform-kubernetes-wandb) Terraform モジュールを手動で更新する必要がありました。Terraform モジュールを更新することで、クラウドプロバイダー間の互換性が確保され、必要な Terraform 変数が設定され、すべてのバックエンドまたは Kubernetes レベルの変更ごとに Terraform を適用することが保証されました。

このプロセスはスケーラブルではありませんでした。なぜなら、W&B サポートが各顧客に対して Terraform モジュールのアップグレードを支援しなければならなかったからです。

その解決策は、中央の [deploy.wandb.ai](https://deploy.wandb.ai) サーバーに接続するオペレーターを実装し、特定のリリースチャンネルに対する最新の仕様変更を要求して適用することでした。ライセンスが有効な限り、更新が受け取れます。[Helm](https://helm.sh/) は、W&B オペレーターのデプロイメントメカニズムとして、また W&B Kubernetes スタックのすべての設定テンプレート処理を行う手段として使用され、Helm-セプションを実現します。

### 仕組み

オペレーターを helm でインストールするか、ソースからインストールすることができます。詳細な手順は[charts/operator](https://github.com/wandb/helm-charts/tree/main/charts/operator) を参照してください。

インストールプロセスは `controller-manager` という名前のデプロイメントを作成し、`spec` をクラスターに適用する `weightsandbiases.apps.wandb.com` (shortName: `wandb`) という名前の[カスタムリソース](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)定義を使用します。

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: weightsandbiases.apps.wandb.com
```

`controller-manager` は、カスタムリソース、リリースチャンネル、およびユーザー定義の設定の spec に基づいて [charts/operator-wandb](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb) をインストールします。設定の仕様の階層は、ユーザー側での最大限の設定の柔軟性を実現し、新しい画像、設定、機能、および Helm 更新を自動的にリリースすることが可能です。

設定オプションについては、[設定仕様階層]({{< relref path="#configuration-specification-hierarchy" lang="ja" >}})および[設定参照]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}})を参照してください。

### 設定仕様階層

設定仕様は、上位レベルの仕様が下位レベルのものをオーバーライドする階層モデルに従います。以下はその仕組みです：

- **リリースチャンネル値**: これは基本レベルの設定で、デプロイメントに対する W&B によって設定されたリリースチャンネルに基づいてデフォルトの値と設定を設定します。
- **ユーザー入力値**: システムコンソールを通じて、ユーザーはリリースチャンネル Spec によって提供されるデフォルト設定をオーバーライドすることができます。
- **カスタムリソース値**: ユーザーから提供される最高レベルの仕様です。ここで指定された値は、ユーザー入力およびリリースチャンネルの仕様の両方をオーバーライドします。設定オプションの詳細な説明については、[設定参照]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}})を参照してください。

この階層モデルは、さまざまなニーズに合わせて柔軟でカスタマイズ可能な設定を保証し、管理可能で体系的なアップグレードと変更のアプローチを維持します。

### W&B Kubernetes オペレーターを使用するための要件

W&B を W&B Kubernetes オペレーターでデプロイするために、次の要件を満たしてください:

[リファレンスアーキテクチャ]({{< relref path="../ref-arch.md#infrastructure-requirements" lang="ja" >}})を参照してください。また、[有効な W&B サーバーライセンスを取得]({{< relref path="../#obtain-your-wb-server-license" lang="ja" >}})します。

セルフマネージドインストールのセットアップと構成方法についての詳細な説明は、こちらの[ガイド]({{< relref path="../bare-metal.md" lang="ja" >}})を参照してください。

インストール方法によっては、次の要件を満たす必要がある場合があります:
* 正しい Kubernetes クラスターコンテキストでインストール済みかつ構成済みの Kubectl。
* Helm がインストールされていること。

### エアギャップインストール

エアギャップ環境での W&B Kubernetes オペレーターのインストール方法については、[Deploy W&B in airgapped environment with Kubernetes]({{< relref path="operator-airgapped.md" lang="ja" >}}) チュートリアルを参照してください。

## W&B Server アプリケーションのデプロイ

このセクションでは、W&B Kubernetes オペレーターをデプロイするさまざまな方法を説明しています。
{{% alert %}}
W&B Operator は、W&B Server のデフォルトで推奨されるインストール方法です
{{% /alert %}}

**以下のいずれかを選択してください:**
- 必要なすべての外部サービスをプロビジョニング済みで、Helm CLI を使用して W&B を Kubernetes にデプロイしたい場合は[こちら]({{< relref path="#deploy-wb-with-helm-cli" lang="ja" >}})を参照してください。
- インフラストラクチャと W&B Server を Terraform で管理することを好む場合は[こちら]({{< relref path="#deploy-wb-with-helm-terraform-module" lang="ja" >}})を参照してください。
- W&B Cloud Terraform Modules を利用したい場合は[こちら]({{< relref path="#deploy-wb-with-wb-cloud-terraform-modules" lang="ja" >}})を参照してください。

### Helm CLI で W&B をデプロイする

W&B は W&B Kubernetes オペレーターを Kubernetes クラスターにデプロイするための Helm Chart を提供しています。この方法により、Helm CLI または ArgoCD などの継続的デリバリーツールを使用して W&B Server をデプロイできます。上記の要件が満たされていることを確認してください。

次の手順に従って、Helm CLI を使用して W&B Kubernetes オペレーターをインストールします:

1. W&B Helm リポジトリを追加します。W&B Helm チャートは W&B Helm リポジトリで利用可能です。以下のコマンドでリポジトリを追加します:
```shell
helm repo add wandb https://charts.wandb.ai
helm repo update
```
2. Kubernetes クラスターにオペレーターをインストールします。以下をコピーして貼り付けます:
```shell
helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
```
3. W&B オペレーターのカスタムリソースを構成して W&B Server のインストールをトリガーします。この設定の例を `operator.yaml` というファイルにコピーし、W&B デプロイメントをカスタマイズできるようにします。[設定参照]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}})を参照してください。

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

    独自の設定でオペレーターを開始して、W&B Server アプリケーションをインストールおよび構成できるようにします。

    ```shell
    kubectl apply -f operator.yaml
    ```

    デプロイメントが完了するまで待ちます。これには数分かかります。

5. Web UI を使用してインストールを検証するには、最初の管理者ユーザーアカウントを作成し、[インストールの検証]({{< relref path="#verify-the-installation" lang="ja" >}})で説明されている検証手順に従います。

### Helm Terraform Module で W&B をデプロイする

この方法は、特定の要件に合わせたカスタマイズされたデプロイメントを可能にし、Terraform のインフラストラクチャ-as-code アプローチを活用して一貫性と再現性を実現します。公式の W&B Helm ベースの Terraform Module は[こちら](https://registry.terraform.io/modules/wandb/wandb/helm/latest)にあります。

以下のコードを出発点として使用し、本番グレードのデプロイメントに必要な設定オプションをすべて含めることができます。

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

設定オプションは[設定参照]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}})に記載されているものと同じですが、構文は HashiCorp Configuration Language (HCL) に従う必要があります。Terraform モジュールは、W&B カスタムリソース定義 (CRD) を作成します。

W&B&Biases 自身が「Dedicated cloud」インストールをデプロイするために Helm Terraform モジュールをどのように活用しているかを知るには、次のリンクをたどってください：
- [AWS](https://github.com/wandb/terraform-aws-wandb/blob/45e1d746f53e78e73e68f911a1f8cad5408e74b6/main.tf#L225)
- [Azure](https://github.com/wandb/terraform-azurerm-wandb/blob/170e03136b6b6fc758102d59dacda99768854045/main.tf#L155)
- [GCP](https://github.com/wandb/terraform-google-wandb/blob/49ddc3383df4cefc04337a2ae784f57ce2a2c699/main.tf#L189)

### W&B Cloud Terraform Modules で W&B をデプロイする

W&B は AWS、GCP、および Azure のための Terraform Modules を提供しています。これらのモジュールは、Kubernetes クラスター、ロードバランサー、MySQL データベースなどのインフラ全体と同様に W&B Server アプリケーションをデプロイします。これらの公式 W&B クラウド固有の Terraform Modules には、W&B Kubernetes オペレーターが既に組み込まれています。

| Terraform Registry                                                  | Source Code                                      | Version |
| ------------------------------------------------------------------- | ------------------------------------------------ | ------- |
| [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest) | https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| [Azure](https://github.com/wandb/terraform-azurerm-wandb)           | https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| [GCP](https://github.com/wandb/terraform-google-wandb)              | https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

この統合により、最小限のセットアップで W&B インスタンス用の W&B Kubernetes オペレーターの準備が整い、クラウド環境での W&B Server のデプロイと管理がスムーズに行えます。

これらのモジュールの使用方法の詳細な説明については、これを[セクション]({{< relref path="../#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}})のセルフマネージドインストールセクションのドキュメントを参照してください。

### インストールを検証する

インストールを検証するには、W&B は [W&B CLI]({{< relref path="/ref/cli/" lang="ja" >}}) を使用することを推奨しています。検証コマンドは、すべてのコンポーネントと設定を検証するいくつかのテストを実行します。

{{% alert %}}
このステップは、最初の管理者ユーザーアカウントをブラウザで作成してあることを前提としています。
{{% /alert %}}

インストールを検証するために以下の手順に従います:

1. W&B CLI をインストールします:
    ```shell
    pip install wandb
    ```
2. W&B にログインします:
    ```shell
    wandb login --host=https://YOUR_DNS_DOMAIN
    ```

    例:
    ```shell
    wandb login --host=https://wandb.company-name.com
    ```

3. インストールを検証します:
    ```shell
    wandb verify
    ```

正常なインストールと完全に機能する W&B デプロイメントは、次の出力を示します:

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

W&B Kubernetes オペレーターには管理コンソールが付属しています。 `${HOST_URI}/console` にあり、例えば `https://wandb.company-name.com/` です。

管理コンソールにログインする方法は2つあります:

{{< tabpane text=true >}}
{{% tab header="Option 1 (推奨)" value="option1" %}}
1. W&B アプリケーションをブラウザで開き、ログインします。W&B アプリケーションには `${HOST_URI}/` でログインします。例えば `https://wandb.company-name.com/`
2. コンソールにアクセスします。右上のアイコンをクリックし、次に **System console** をクリックします。管理者権限を持つユーザーだけが **System console** エントリを見ることができます。

    {{< img src="/images/hosting/access_system_console_via_main_app.png" alt="" >}}
{{% /tab %}}

{{% tab header="Option 2" value="option2"%}}
{{% alert %}}
W&B は、Option 1 が機能しない場合のみ、以下の手順を使用してコンソールにアクセスすることを推奨します。
{{% /alert %}}

1. ブラウザでコンソールアプリケーションを開きます。上記で説明されている URL を開くと、ログイン画面にリダイレクトされます:
    {{< img src="/images/hosting/access_system_console_directly.png" alt="" >}}
2. インストールが生成する Kubernetes シークレットからパスワードを取得します:
    ```shell
    kubectl get secret wandb-password -o jsonpath='{.data.password}' | base64 -d
    ```
    パスワードをコピーします。
3. コンソールにログインします。コピーしたパスワードを貼り付け、次に **Login** をクリックします。
{{% /tab %}}
{{< /tabpane >}}

## W&B Kubernetes オペレーターの更新

このセクションでは、W&B Kubernetes オペレーターを更新する方法を説明します。

{{% alert %}}
* W&B Kubernetes オペレーターを更新しても、W&B サーバーアプリケーションは更新されません。
* W&B Kubernetes オペレーターを使用していない helm chart を使用している場合は、続いて W&B オペレーターを更新する手順を実行する前に[こちら]({{< relref path="#migrate-self-managed-instances-to-wb-operator" lang="ja" >}})の指示を参照してください。
{{% /alert %}}

以下のコードスニペットをターミナルにコピーして貼り付けます。

1. まず、 [`helm repo update`](https://helm.sh/docs/helm/helm_repo_update/) でリポジトリを更新します:
    ```shell
    helm repo update
    ```

2. 次に、 [`helm upgrade`](https://helm.sh/docs/helm/helm_upgrade/) で Helm チャートを更新します:
    ```shell
    helm upgrade operator wandb/operator -n wandb-cr --reuse-values
    ```

## W&B Server アプリケーションの更新

W&B Kubernetes オペレーターを使用する場合、W&B Server アプリケーションの更新は不要です。

オペレーターは、W&B のソフトウェアの新しいバージョンがリリースされると、W&B Server アプリケーションを自動的に更新します。

## W&B オペレーターへのセルフマネージドインスタンスの移行

このセクションでは、自分自身で W&B Server インストールを管理することから、W&B オペレーターを使用してこれを実行するための移行プロセスを説明しています。移行プロセスは、W&B Server をインストールした方法によって異なります:

{{% alert %}}
W&B オペレーターは、W&B Server のデフォルトで推奨されるインストール方法です。質問がある場合や不明点がある場合は、[カスタマーサポート](mailto:support@wandb.com) または W&B チームに問い合わせてください。
{{% /alert %}}

- 公式の W&B Cloud Terraform Modules を使用した場合は、適切なドキュメントを参照し、次の手順に従ってください:
  - [AWS]({{< relref path="#migrate-to-operator-based-aws-terraform-modules" lang="ja" >}})
  - [GCP]({{< relref path="#migrate-to-operator-based-gcp-terraform-modules" lang="ja" >}})
  - [Azure]({{< relref path="#migrate-to-operator-based-azure-terraform-modules" lang="ja" >}})
- [W&B Non-Operator Helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/wandb)を使用した場合は[こちら]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ja" >}})を続けてください。
- [W&B Non-Operator Helm チャート with Terraform](https://registry.terraform.io/modules/wandb/wandb/kubernetes/latest) を使用した場合は[こちら]({{< relref path="#migrate-to-operator-based-terraform-helm-chart" lang="ja" >}})を続けてください。
- Kubernetes マニフェストでリソースを作成した場合は[こちら]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ja" >}})を続けてください。

### オペレーターを基にした AWS Terraform Modules への移行

移行プロセスの詳細な説明については、こちら [here]({{< relref path="../install-on-public-cloud/aws-tf.md#migrate-to-operator-based-aws-terraform-modules" lang="ja" >}})を参照してください。

### オペレーターを基にした GCP Terraform Modules への移行

質問がある場合や支援が必要な際は、[カスタマーサポート](mailto:support@wandb.com)または W&B チームにお問い合わせください。

### オペレーターを基にした Azure Terraform Modules への移行

質問がある場合や支援が必要な際は、[カスタマーサポート](mailto:support@wandb.com)または W&B チームにお問い合わせください。

### オペレーターを基にした Helm チャートへの移行

オペレーターを基にした Helm チャートへの移行手順は次のとおりです:

1. 現在の W&B 設定を取得します。W&B がオペレーターを基にしていないバージョンの Helm チャートでデプロイされている場合、次のように値をエクスポートします:
    ```shell
    helm get values wandb
    ```
    W&B が Kubernetes マニフェストでデプロイされている場合、次のように値をエクスポートします:
    ```shell
    kubectl get deployment wandb -o yaml
    ```
    これで、次のステップで必要なすべての設定値が手元にあります。

2. `operator.yaml` というファイルを作成します。[設定参照]({{< relref path="#configuration-reference-for-wb-operator" lang="ja" >}}) で説明されている形式に従ってください。ステップ 1 の値を使用します。

3. 現在のデプロイメントを 0 ポッドにスケールします。このステップで現在のデプロイメントを停止します。
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
6. 新しい helm チャートを構成し、W&B アプリケーションのデプロイメントをトリガーします。新しい設定を適用します。
    ```shell
    kubectl apply -f operator.yaml
    ```
    デプロイメントが完了するまでに数分かかります。

7. インストールを検証します。すべてが正常に動作することを確認するために、[インストールの検証]({{< relref path="#verify-the-installation" lang="ja" >}})の手順に従います。

8. 古いインストールの削除。古い helm チャートをアンインストールするか、マニフェストで作成されたリソースを削除します。

### オペレーターを基にした Terraform Helm チャートへの移行

オペレーターを基にした Helm チャートへの移行手順は次のとおりです:

1. Terraform 設定を準備します。Terraform 設定内の古いデプロイメントの Terraform コードを、[こちら]({{< relref path="#deploy-wb-with-helm-terraform-module" lang="ja" >}})で説明されているものに置き換えます。以前と同じ変数を設定します。.tfvars ファイルがある場合、それを変更しないでください。
2. Terraform run を実行します。terraform init、plan、および apply を実行します。
3. インストールを検証します。すべてが正常に動作することを確認するために、[インストールの検証]({{< relref path="#verify-the-installation" lang="ja" >}})の手順に従います。
4. 古いインストールの削除。古い helm チャートをアンインストールするか、マニフェストで作成されたリソースを削除します。

## Configuration Reference for W&B Server

このセクションでは、W&B サーバーアプリケーションの設定オプションについて説明します。アプリケーションは、[WeightsAndBiases]({{< relref path="#how-it-works" lang="ja" >}})というカスタムリソース定義としてその設定を受け取ります。一部の設定オプションは以下の設定で公開され、他は環境変数として設定する必要があります。

ドキュメントには環境変数が2つのリストに分かれています：[basic]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}) および [advanced]({{< relref path="/guides/hosting/iam/advanced_env_vars/" lang="ja" >}})。必要な設定オプションが Helm Chart を使用して公開されていない場合にのみ環境変数を使用してください。

本番展開用の W&B サーバーアプリケーションの設定ファイルには、以下の内容が必要です。この YAML ファイルは、W&B デプロイメントの望ましい状態を定義し、バージョン、環境変数、データベースなどの外部リソース、およびその他必要な設定を含みます。

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

完全な値セットは [W&B Helm リポジトリ](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml)にあります。オーバーライドする必要がある値のみを変更してください。

### 完全な例

これは、GCP Kubernetes を使用した GCP Ingress および GCS（GCP オブジェクトストレージ）を使用した設定例です：

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
 # プロトコルと共に完全修飾ドメイン名を提供
global:
  # ホスト名の例、独自のものに置き換え
  host: https://wandb example com
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

**その他のプロバイダー（Minio、Ceph、など）**

他の S3 互換プロバイダーの場合、バケットの設定は次のようにします：

```yaml
global:
  bucket:
    # 例の値、独自のものに置き換え
    provider: s3
    name: storage.example.com
    kmsKey: null
    path: wandb
    region: default
    accessKey: 5WOA500...P5DK7I
    secretKey: HDKYe4Q...JAp1YyjysnX
```

AWS 外部でホスティングされている S3 互換ストレージの場合、`kmsKey` は `null` にする必要があります。

`accessKey` および `secretKey` をシークレットから参照するには：

```yaml
global:
  bucket:
    # 例の値、独自のものに置き換え
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
     # 例の値、独自のものに置き換え
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
     # 例の値、独自のものに置き換え
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
  # 例のライセンス、独自のものに置き換え
  license: eyJhbGnUzaHgyQjQy...VFnPS_KETXg1hi
```

`license` をシークレットから参照するには：

```yaml
global:
  licenseSecret:
    name: license-secret
    key: CUSTOMER_WANDB_LICENSE
```

### Ingress

Kubernetes ingress クラスを識別する方法については、FAQ [エントリ]({{< relref path="#how-to-identify-the-kubernetes-ingress-class" lang="ja" >}})を参照してください。

**TLS なし**

```yaml
global:
# 重要: Ingress は YAML の `global` と同じレベルにあります（子ではありません）
ingress:
  class: ""
```

**TLS 使用**

証明書が含まれるシークレットを作成します

```console
kubectl create secret tls wandb-ingress-tls --key wandb-ingress-tls.key --cert wandb-ingress-tls.crt
```

Ingress 設定でシークレットを参照します

```yaml
global:
# 重要: Ingress は YAML の `global` と同じレベルにあります（子ではありません）
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

Nginx の場合、次の注釈を追加する必要があるかもしれません：

```
ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: 64m
```

### カスタム Kubernetes ServiceAccounts

W&B ポッドを実行するためにカスタム Kubernetes Service Account を指定します。

次のスニペットは、指定された名前でデプロイメントの一部としてサービスアカウントを作成します：

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

サブシステム "app" および "parquet" は指定されたサービスアカウントの下で実行されます。他のサブシステムはデフォルトのサービスアカウントで実行されます。

サービスアカウントがクラスター上で既に存在する場合、`create: false` を設定します：

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

app, parquet, console, その他の様々なサブシステム上にサービスアカウントを指定できます：

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

サブシステム間でサービスアカウントを異なるものにすることができます：

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

`password` をシークレットから参照するには：

```console
kubectl create secret generic redis-secret --from-literal=redis-password=supersecret
```

下記の設定で参照します：

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

**TLS を使用しない場合**

```yaml
global:
  ldap:
    enabled: true
    # LDAP サーバーアドレスには "ldap://" または "ldaps://" を含める
    host:
    # ユーザーを見つけるために使用する LDAP 検索ベース
    baseDN:
    # バインドに使用する LDAP ユーザー（匿名バインドを使用しない場合）
    bindDN:
    # バインドに使用する LDAP パスワードを含むシークレットの名前とキー（匿名バインドを使用しない場合）
    bindPW:
    # 電子メールおよびグループ ID 属性名の LDAP 属性をカンマ区切りの文字列で指定
    attributes:
    # LDAP グループ許可リスト
    groupAllowList:
    # LDAP TLS の有効化
    tls: false
```

**TLS 使用**

LDAP TLS 証明書の設定には、証明書内容をプレ作成した config map が必要です。

config map を作成するには、次のコマンドを使用できます：

```console
kubectl create configmap ldap-tls-cert --from-file=certificate.crt
```

そして、下記の例のように YAML 内で config map を使用します：

```yaml
global:
  ldap:
    enabled: true
    # LDAP サーバーアドレスには "ldap://" または "ldaps://" を含める
    host:
    # ユーザーを見つけるために使用する LDAP 検索ベース
    baseDN:
    # バインドに使用する LDAP ユーザー（匿名バインドを使用しない場合）
    bindDN:
    # バインドに使用する LDAP パスワードを含むシークレットの名前とキー（匿名バインドを使用しない場合）
    bindPW:
    # 電子メールおよびグループ ID 属性名の LDAP 属性をカンマ区切りの文字列で指定
    attributes:
    # LDAP グループ許可リスト
    groupAllowList:
    # LDAP TLS の有効化
    tls: true
    # LDAP サーバーの CA 証明書を含む ConfigMap の名前とキー
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
      # IdP が要求する場合のみ含める。
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

### カスタム証明書機関

`customCACerts` はリストであり、複数の証明書を含むことができます。`customCACerts` に指定された証明書機関は W&B サーバーアプリケーションのみに適用されます。

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

証明書機関を ConfigMap に保存することもできます：

```yaml
global:
  caCertsConfigMap: custom-ca-certs
```

ConfigMap は次のようになっている必要があります：

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
ConfigMap を使用する場合、ConfigMap 内の各キーは `.crt` で終わる必要があります（例：`my-cert.crt` または `ca-cert1.crt`）。この名前付け規約は、`update-ca-certificates` が各証明書をシステム CA ストアに解析して追加するために必要です。
{{% /alert %}}

### カスタムセキュリティコンテキスト

各 W&B コンポーネントは、以下の形式のカスタムセキュリティコンテキスト設定をサポートしています：

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
`runAsGroup:`には `0` だけが有効な値です。 他の値はエラーです。
{{% /alert %}}

アプリケーションポッドを設定するには、設定に `app` セクションを追加します：

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

同じ概念は `console`、`weave`、`weave-trace`、`parquet` にも適用されます。

## Configuration Reference for W&B Operator

このセクションでは、W&B Kubernetes オペレーター（`wandb-controller-manager`）の設定オプションを説明しています。オペレーターは、YAML ファイルの形式でその設定を受け取ります。

デフォルトでは、W&B Kubernetes オペレーターには設定ファイルは必要ありません。必要な場合にだけ設定ファイルを作成します。たとえば、カスタム証明書機関を指定したり、エアギャップ環境にデプロイしたりする必要がある場合などです。

仕様のカスタマイズの完全なリストは、[Helm リポジトリ](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)で確認できます。

### カスタム CA

カスタム証明書機関（`customCACerts`）はリストであり、複数の証明書を含むことができます。それらの証明書機関が追加されると、W&B Kubernetes オペレーター（`wandb-controller-manager`）のみに適用されます。

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

CA 証明書を ConfigMap に保存することもできます：

```yaml
caCertsConfigMap: custom-ca-certs
```

ConfigMap は次のようになっている必要があります：

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
ConfigMap の各キーは `.crt` で終わる必要があります（例：`my-cert.crt` または `ca-cert1.crt`）。この命名規則は、`update-ca-certificates` が各証明書をシステム CA ストアに解析して追加するために必要です。
{{% /alert %}}

## FAQ

### 各個別のポッドの役割/目的は何ですか？

* **`wandb-app`**: W&B の中枢であり、GraphQL API およびフロントエンドアプリケーションを含みます。これは私たちのプラットフォームの大部分の機能を提供します。
* **`wandb-console`**: 管理コンソールであり、`/console` を通じてアクセスできます。
* **`wandb-otel`**: OpenTelemetry エージェントであり、Kubernetes レイヤーでのリソースからメトリクスおよびログを収集して管理コンソールに表示します。
* **`wandb-prometheus`**: Prometheus サーバーであり、管理コンソールに表示するためにさまざまなコンポーネントからメトリクスを収集します。
* **`wandb-parquet`**: `wandb-app` ポッドとは別のバックエンドマイクロサービスであり、データベースデータを Parquet 形式でオブジェクトストレージにエクスポートします。
* **`wandb-weave`**: UI でクエリテーブルをロードし、さまざまなコアアプリ機能をサポートする別のバックエンドマイクロサービス。
* **`wandb-weave-trace`**: LLM ベースのアプリケーションを追跡、実験、評価、展開、および改善するためのフレームワーク。このフレームワークは `wandb-app` ポッドを介してアクセスできます。

### W&B オペレーターコンソールパスワードの取得方法
[W&B Kubernetes オペレーターマネジメントコンソールへのアクセス]({{< relref path="#access-the-wb-management-console" lang="ja" >}})を参照してください。

### Ingress が機能しない場合に W&B Operator Console にアクセスする方法

Kubernetes クラスターに到達可能なホストで以下のコマンドを実行してください：

```console
kubectl port-forward svc/wandb-console 8082
```

`https://localhost:8082/` console でブラウザからコンソールにアクセスしてください。

コンソールのパスワードの取得方法については、[W&B Kubernetes オペレーターマネジメントコンソールへのアクセス]({{< relref path="#access-the-wb-management-console" lang="ja" >}})（Option 2）を参照してください。

### W&B Server のログを表示する方法

アプリケーションポッドの名前は **wandb-app-xxx** です。

```console
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

### Kubernetes ingress クラスを識別する方法

クラスターにインストールされている ingress クラスを取得するには、次のコマンドを実行します:

```console
kubectl get ingressclass
```