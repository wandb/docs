---
title: ローンチ用語およびコンセプト
menu:
  launch:
    identifier: launch-terminology
    parent: launch
url: guides/launch/launch-terminology
weight: 2
---

W&B Launch では、[jobs]({{< relref "#launch-job" >}}) を [queues]({{< relref "#launch-queue" >}}) に投入して run を作成します。Job は W&B で計測された Python スクリプトです。Queue には実行すべき job のリストが格納され、[target resource]({{< relref "#target-resources" >}}) 上で動作します。[Agents]({{< relref "#launch-agent" >}}) は queues から job を取り出し、target resources 上で実行します。W&B は launch job を [runs]({{< relref "/guides/models/track/runs/" >}}) と同じようにトラッキングします。

### Launch job
Launch job は、完了すべきタスクを表す特定タイプの [W&B Artifact]({{< relref "/guides/core/artifacts/" >}}) です。例えば一般的な launch job には、モデルのトレーニングやモデルの評価実行などがあります。Job 定義には次のものが含まれます。

- Python コードやその他のファイルアセット（少なくとも 1 つの実行可能 entrypoint を含む）。
- 入力（設定パラメータ）と出力（ログされるメトリクス）に関する情報。
- 実行環境についての情報（例: `requirements.txt`、ベースとなる `Dockerfile` など）。

Job 定義には主に 3 種類あります。

| Job types | 定義 | この job タイプの実行方法 |
| ---------- | --------- | -------------- |
|Artifact-based（あるいは code-based） jobs| コードやその他のアセットを W&B artifact として保存します。| Artifact-based job を実行するには、Launch agent に builder の設定が必要です。|
|Git-based jobs| コードやその他のアセットを git リポジトリの特定のコミットやブランチ、タグからクローンします。 | Git-based job を実行するには、Launch agent に builder と git リポジトリの認証情報の設定が必要です。|
|Image-based jobs| コードやその他のアセットを Docker イメージにパッケージ化します。| Image-based job を実行するには、Launch agent にイメージリポジトリ認証情報の設定が必要になることがあります。|

{{% alert %}}
Launch jobs はモデルのトレーニング以外のアクティビティ（例: モデルを Triton 推論サーバーにデプロイするなど）も実行できますが、全ての job で `wandb.init` の呼び出しが必須です。これにより W&B workspace 上でトラッキング用の run が作成されます。
{{% /alert %}}

作成した job は W&B App のプロジェクト workspace 内 `Jobs` タブから確認できます。そこから job を設定し、さまざまな [launch queue]({{< relref "#launch-queue" >}}) に送信して、いろいろな [target resources]({{< relref "#target-resources" >}}) 上で実行できます。

### Launch queue
Launch *queues* は、特定の target resource 上で実行される job の順序付きリストです。Launch queue は先入れ先出し（FIFO）方式です。Queue の数に実質的な制限はありませんが、リソースごとに 1 つの queue が目安です。Job は W&B App UI、W&B CLI、Python SDK のいずれかから queue に投入できます。その後、1 つ以上の Launch agent を設定し、queue から項目を取得して対応する target resource で実行することができます。

### Target resources
Launch queue が job を実行するために設定されている計算環境を *target resource* と呼びます。

W&B Launch は以下の target resource をサポートしています。

- [Docker]({{< relref "/launch/set-up-launch/setup-launch-docker.md" >}})
- [Kubernetes]({{< relref "/launch/set-up-launch/setup-launch-kubernetes.md" >}})
- [AWS SageMaker]({{< relref "/launch/set-up-launch/setup-launch-sagemaker.md" >}})
- [GCP Vertex]({{< relref "/launch/set-up-launch/setup-vertex.md" >}})

各 target resource は、それぞれ異なる設定パラメータ（*resource configurations*）を受け付けます。Resource configurations には、各 Launch queue で定義されたデフォルト値が適用されますが、各 job ごとに個別に上書きすることもできます。詳細は各 target resource のドキュメントを参照してください。

### Launch agent
Launch agent は、Launch queue を定期的にチェックして job を実行する軽量で持続的なプログラムです。Agent が job を受け取ると、まず job 定義に基づきイメージをビルドまたは取得し、その後 target resource 上で実行します。

1 つの agent で複数の queue を監視できますが、その場合には agent が監視する全ての queue の target resource をサポートできるよう適切に設定する必要があります。

### Launch agent environment
Agent environment とは、Launch agent が起動して job を監視している環境のことです。

{{% alert %}}
Agent の実行環境は、queue の target resource とは独立しています。つまり、agent は必要な target resource に適切にアクセスできるよう設定さえされていれば、どこからでもデプロイ可能です。
{{% /alert %}}