---
title: Launch terms and concepts
menu:
  launch:
    identifier: ja-launch-launch-terminology
    parent: launch
url: guides/launch/launch-terminology
weight: 2
---

W&B Launch を使用すると、[ジョブ]({{< relref path="#launch-job" lang="ja" >}}) を [キュー]({{< relref path="#launch-queue" lang="ja" >}}) にエンキューして run を作成できます。ジョブは、W&B で計測された Python スクリプトです。キューは、[ターゲットリソース]({{< relref path="#target-resources" lang="ja" >}}) で実行するジョブのリストを保持します。[エージェント]({{< relref path="#launch-agent" lang="ja" >}}) は、キューからジョブを取得し、ターゲットリソースでジョブを実行します。W&B は、[run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を追跡する方法と同様に、Launch ジョブを追跡します。

### Launch job
Launch job とは、完了するタスクを表す特定のタイプの [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) です。たとえば、一般的な Launch job には、モデルのトレーニングやモデルの評価のトリガーなどがあります。ジョブ定義には以下が含まれます。

- 少なくとも 1 つの実行可能なエントリポイントを含む、Python コードおよびその他のファイルアセット。
- 入力 (config パラメータ) および出力 (記録されたメトリクス) に関する情報。
- 環境に関する情報 (例: `requirements.txt`、ベース `Dockerfile`)。

ジョブ定義には、主に次の 3 種類があります。

| ジョブタイプ | 定義 | このジョブタイプの実行方法 |
| ---------- | --------- | -------------- |
| Artifact ベース (またはコードベース) のジョブ | コードおよびその他のアセットは、W&B Artifact として保存されます。 | Artifact ベースのジョブを実行するには、Launch agent をビルダーで構成する必要があります。 |
| Git ベースのジョブ | コードおよびその他のアセットは、git リポジトリの特定のコミット、ブランチ、またはタグから複製されます。 | Git ベースのジョブを実行するには、Launch agent をビルダーおよび git リポジトリの認証情報で構成する必要があります。 |
| イメージベースのジョブ | コードおよびその他のアセットは、Docker イメージに組み込まれます。 | イメージベースのジョブを実行するには、Launch agent をイメージリポジトリの認証情報で構成する必要がある場合があります。 |

{{% alert %}}
Launch job はモデルトレーニングに関連しないアクティビティ (たとえば、モデルを Triton 推論サーバーにデプロイするなど) を実行できますが、すべてのジョブは正常に完了するために `wandb.init` を呼び出す必要があります。これにより、W&B Workspace で追跡するための run が作成されます。
{{% /alert %}}

作成したジョブは、プロジェクト Workspace の [Jobs] タブの W&B アプリで確認できます。そこから、ジョブを構成し、[Launch キュー]({{< relref path="#launch-queue" lang="ja" >}}) に送信して、さまざまな [ターゲットリソース]({{< relref path="#target-resources" lang="ja" >}}) で実行できます。

### Launch queue
Launch *キュー* は、特定のターゲットリソースで実行するジョブの順序付きリストです。Launch キューは、先入れ先出し (FIFO) です。キューの数に実質的な制限はありませんが、目安として、ターゲットリソースごとに 1 つのキューを設定することをお勧めします。ジョブは、W&B アプリ UI、W&B CLI、または Python SDK でエンキューできます。次に、1 つまたは複数の Launch エージェントを構成して、キューからアイテムを取得し、キューのターゲットリソースで実行できます。

### ターゲットリソース
Launch キューがジョブを実行するように構成されているコンピューティング環境は、*ターゲットリソース* と呼ばれます。

W&B Launch は、次のターゲットリソースをサポートしています。

- [Docker]({{< relref path="./set-up-launch/setup-launch-docker.md" lang="ja" >}})
- [Kubernetes]({{< relref path="./set-up-launch/setup-launch-kubernetes.md" lang="ja" >}})
- [AWS SageMaker]({{< relref path="./set-up-launch/setup-launch-sagemaker.md" lang="ja" >}})
- [GCP Vertex]({{< relref path="./set-up-launch/setup-vertex.md" lang="ja" >}})

各ターゲットリソースは、*リソース構成* と呼ばれるさまざまな設定パラメータを受け入れます。リソース構成は、各 Launch キューによって定義されたデフォルト値を想定しますが、各ジョブによって個別にオーバーライドできます。詳細については、各ターゲットリソースのドキュメントを参照してください。

### Launch agent
Launch agent は、Launch キューを定期的にチェックして実行するジョブがないか確認する、軽量で永続的なプログラムです。Launch agent がジョブを受信すると、最初にジョブ定義からイメージを構築またはプルし、ターゲットリソースで実行します。

1 つのエージェントが複数のキューをポーリングできますが、エージェントは、ポーリングする各キューのバッキングターゲットリソースをすべてサポートするように適切に構成する必要があります。

### Launch agent 環境
エージェント環境とは、Launch agent が実行され、ジョブをポーリングする環境のことです。

{{% alert %}}
エージェントのランタイム環境は、キューのターゲットリソースとは独立しています。言い換えれば、エージェントは、必要なターゲットリソースにアクセスできるように十分に構成されていれば、どこにでもデプロイできます。
{{% /alert %}}
