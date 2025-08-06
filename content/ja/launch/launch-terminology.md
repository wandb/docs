---
title: ローンチ の用語とコンセプト
menu:
  launch:
    identifier: ja-launch-launch-terminology
    parent: launch
url: guides/launch/launch-terminology
weight: 2
---

W&B Launch では、[jobs]({{< relref path="#launch-job" lang="ja" >}}) を [queues]({{< relref path="#launch-queue" lang="ja" >}}) にエンキューし、run を作成します。ジョブは W&B でインスツルメントされた Python スクリプトです。キューはジョブのリストを保持し、[target resource]({{< relref path="#target-resources" lang="ja" >}}) 上で実行されます。[Agents]({{< relref path="#launch-agent" lang="ja" >}}) はキューからジョブを取得し、ターゲットリソース上でジョブを実行します。W&B は Launch ジョブを、[runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) のトラッキングと同様に記録します。

### Launch job

Launch job とは、完了すべきタスクを表す特定の種類の [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) です。たとえば、よくある Launch job にはモデルのトレーニングやモデル評価の実行などがあります。ジョブ定義には以下が含まれます:

- Python コードおよびその他のファイルアセット（少なくとも1つの実行可能なエントリーポイントを含む）。
- 入力情報（設定パラメータ）や出力情報（記録されたメトリクス）。
- 環境に関する情報（例: `requirements.txt` やベースとなる `Dockerfile` など）。

ジョブ定義には主に 3 種類あります:

| Job types | 定義 | このジョブタイプの実行方法 | 
| ---------- | --------- | -------------- |
|Artifact-based (または code-based) jobs| コードとその他のアセットを W&B Artifact として保存します。| Artifact-based ジョブを実行するには、Launch agent をビルダーで設定する必要があります。|
|Git-based jobs| コードおよびアセットを git リポジトリの特定のコミット、ブランチ、タグからクローンします。 | Git-based ジョブの実行には、Launch agent をビルダーと git リポジトリの認証情報で設定する必要があります。|
|Image-based jobs| コードやアセットを Docker イメージに組み込みます。 | Image-based ジョブを実行するには、Launch agent をイメージリポジトリの認証情報で設定する場合があります。| 

{{% alert %}}
Launch job はモデルのトレーニング以外の処理（例: モデルを Triton 推論サーバーにデプロイ）も可能ですが、すべてのジョブは `wandb.init` を呼び出す必要があります。これにより、W&B Workspace にトラッキング用の run が作成されます。
{{% /alert %}}

作成したジョブは W&B アプリのプロジェクトワークスペース内「Jobs」タブで確認できます。そこからジョブを設定し、実行のために [launch queue]({{< relref path="#launch-queue" lang="ja" >}}) に送信できます。さまざまな [target resources]({{< relref path="#target-resources" lang="ja" >}}) で実行できます。

### Launch queue

Launch *queues* は、特定のターゲットリソースで実行されるジョブの順序付きリストです。Launch queue は「先入れ先出し」（FIFO）方式です。キューの数に実質的な上限はありませんが、ターゲットリソースごとにキューを 1 つ用意するのが良い目安です。ジョブは W&B アプリの UI、W&B CLI、または Python SDK でエンキューできます。その後、1つ以上の Launch agent を設定し、キューからアイテムを取得して、そのキューのターゲットリソース上で実行できます。

### Target resources

Launch queue がジョブを実行するように設定されている計算環境のことを *target resource* と呼びます。

W&B Launch は以下のターゲットリソースをサポートしています:

- [Docker]({{< relref path="/launch/set-up-launch/setup-launch-docker.md" lang="ja" >}})
- [Kubernetes]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ja" >}})
- [AWS SageMaker]({{< relref path="/launch/set-up-launch/setup-launch-sagemaker.md" lang="ja" >}})
- [GCP Vertex]({{< relref path="/launch/set-up-launch/setup-vertex.md" lang="ja" >}})

それぞれのターゲットリソースは「リソース設定」と呼ばれる固有の設定パラメータを受け取ります。リソース設定には各 Launch queue ごとのデフォルト値がありますが、各ジョブごとに個別に上書き可能です。詳細は各ターゲットリソースのドキュメントを参照してください。

### Launch agent

Launch agent は、小型で常駐型のプログラムで、定期的に Launch queue をチェックして実行可能なジョブがないか監視します。Launch agent がジョブを受信すると、まずジョブ定義からイメージをビルドまたはプルし、その後ターゲットリソース上で実行します。

1つの agent で複数の queue を監視することも可能ですが、各 queue のバックエンドとなる target resource すべてをサポートできるよう、適切に設定する必要があります。

### Launch agent environment

Agent environment とは、Launch agent が動作し、ジョブを監視する環境を指します。

{{% alert %}}
Agent の実行環境は queue の target resource とは独立しています。つまり、agent は必要な target resource に十分アクセスできるよう設定されていれば、どこにでもデプロイ可能です。
{{% /alert %}}