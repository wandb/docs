---
title: Launch terms and concepts
menu:
  launch:
    identifier: ja-launch-launch-terminology
    parent: launch
url: guides/launch/launch-terminology
weight: 2
---

W&B Launch を使用すると、[ジョブ]({{< relref path="#launch-job" lang="ja" >}})を[キュー]({{< relref path="#launch-queue" lang="ja" >}})にエンキューして run を作成できます。ジョブは、W&B で計測された Python スクリプトです。キューは、[ターゲットリソース]({{< relref path="#target-resources" lang="ja" >}})で実行するジョブのリストを保持します。[エージェント]({{< relref path="#launch-agent" lang="ja" >}})は、キューからジョブをプルし、ターゲットリソースでジョブを実行します。W&B は、W&B が[run]({{< relref path="/guides/models/track/runs/" lang="ja" >}})を追跡するのと同様に、Launch ジョブを追跡します。

### Launch ジョブ
Launch ジョブは、完了するタスクを表す特定のタイプの [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}})です。たとえば、一般的な Launch ジョブには、モデルのトレーニングやモデルの評価のトリガーなどがあります。ジョブの定義には以下が含まれます。

- 少なくとも1つの実行可能なエントリポイントを含む、Python コードおよびその他のファイルアセット。
- 入力 (config パラメータ) と出力 (記録されたメトリクス) に関する情報。
- 環境に関する情報 (例: `requirements.txt`、ベース `Dockerfile`)。

ジョブの定義には、主に次の3種類があります。

| ジョブタイプ | 定義 | このジョブタイプの実行方法 | 
| ---------- | --------- | -------------- |
|Artifact ベース (またはコードベース) のジョブ| コードおよびその他のアセットは、W&B artifact として保存されます。| artifact ベースのジョブを実行するには、Launch エージェントをビルダーで構成する必要があります。 |
|Git ベースのジョブ| コードおよびその他のアセットは、git リポジトリ内の特定のコミット、ブランチ、またはタグから複製されます。 | git ベースのジョブを実行するには、Launch エージェントをビルダーおよび git リポジトリの認証情報で構成する必要があります。 |
|イメージベースのジョブ| コードおよびその他のアセットは、Docker イメージにベイクされます。 | イメージベースのジョブを実行するには、Launch エージェントをイメージリポジトリの認証情報で構成する必要がある場合があります。 |

{{% alert %}}
Launch ジョブは、モデルトレーニングに関係のないアクティビティ (たとえば、Triton 推論サーバーにモデルをデプロイするなど) を実行できますが、すべてのジョブは正常に完了するために `wandb.init` を呼び出す必要があります。これにより、W&B ワークスペースで追跡するための run が作成されます。
{{% /alert %}}

作成したジョブは、W&B アプリのプロジェクト ワークスペースの [Jobs] タブにあります。そこから、ジョブを構成して [Launch キュー]({{< relref path="#launch-queue" lang="ja" >}})に送信し、さまざまな[ターゲットリソース]({{< relref path="#target-resources" lang="ja" >}})で実行できます。

### Launch キュー
Launch *キュー* は、特定のターゲットリソースで実行するジョブの順序付きリストです。Launch キューは先入れ先出し (FIFO) です。キューの数に実用的な制限はありませんが、適切なガイドラインはターゲットリソースごとに1つのキューです。ジョブは、W&B アプリ UI、W&B CLI、または Python SDK でエンキューできます。次に、1つ以上の Launch エージェントを構成して、キューからアイテムをプルし、キューのターゲットリソースで実行できます。

### ターゲットリソース
Launch キューがジョブの実行を構成するように設定されているコンピューティング環境は、*ターゲットリソース* と呼ばれます。

W&B Launch は、次のターゲットリソースをサポートしています。

- [Docker]({{< relref path="/launch/set-up-launch/setup-launch-docker.md" lang="ja" >}})
- [Kubernetes]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ja" >}})
- [AWS SageMaker]({{< relref path="/launch/set-up-launch/setup-launch-sagemaker.md" lang="ja" >}})
- [GCP Vertex]({{< relref path="/launch/set-up-launch/setup-vertex.md" lang="ja" >}})

各ターゲットリソースは、*リソース構成* と呼ばれる異なる設定パラメータのセットを受け入れます。リソース構成は、各 Launch キューによって定義されたデフォルト値を採用しますが、各ジョブによって個別に上書きできます。詳細については、各ターゲットリソースのドキュメントを参照してください。

### Launch エージェント
Launch エージェントは、Launch キューで実行するジョブを定期的にチェックする、軽量で永続的なプログラムです。Launch エージェントがジョブを受信すると、最初にジョブ定義からイメージを構築またはプルし、ターゲットリソースで実行します。

1つのエージェントが複数のキューをポーリングする場合がありますが、エージェントは、ポーリングする各キューのバッキングターゲットリソースをすべてサポートするように適切に構成する必要があります。

### Launch エージェント環境
エージェント環境は、Launch エージェントがジョブをポーリングして実行されている環境です。

{{% alert %}}
エージェントのランタイム環境は、キューのターゲットリソースとは独立しています。つまり、エージェントは、必要なターゲットリソースにアクセスできるように十分に構成されていれば、どこにでもデプロイできます。
{{% /alert %}}
