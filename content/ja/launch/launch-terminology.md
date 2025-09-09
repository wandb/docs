---
title: Launch の 用語 と 概念
menu:
  launch:
    identifier: ja-launch-launch-terminology
    parent: launch
url: guides/launch/launch-terminology
weight: 2
---

W&B Launch では、[ジョブ]({{< relref path="#launch-job" lang="ja" >}}) を [キュー]({{< relref path="#launch-queue" lang="ja" >}}) にエンキューして run を作成します。ジョブは W&B で計装された Python スクリプトです。キューは、[ターゲット リソース]({{< relref path="#target-resources" lang="ja" >}}) 上で実行するジョブの一覧を保持します。[エージェント]({{< relref path="#launch-agent" lang="ja" >}}) はキューからジョブを取得し、ターゲット リソース上で実行します。W&B は Launch ジョブを、W&B が [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を追跡するのと同様の方法でトラッキングします。

### Launch ジョブ
Launch ジョブは、完了すべきタスクを表す [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の一種です。例えば、一般的な Launch ジョブには モデル のトレーニングや モデルの評価 のトリガーなどがあります。ジョブ定義には次が含まれます:
- Python コードとその他のファイル アセット。少なくとも 1 つの実行可能なエントリポイントを含みます。
- 入力 (設定 パラメータ) と出力 (ログされる メトリクス) に関する情報。
- 環境 に関する情報 (例: `requirements.txt`、ベースの `Dockerfile`)。

ジョブ定義には主に 3 種類あります:

| ジョブタイプ | 定義 | この ジョブタイプ の実行方法 | 
| ---------- | --------- | -------------- |
| Artifact ベース (または コード ベース) の ジョブ | コードやその他のアセットは W&B Artifact として保存されます。 | Artifact ベースのジョブを実行するには、Launch エージェント に ビルダー を 設定 する必要があります。 |
| Git ベースの ジョブ | コードやその他のアセットは Git リポジトリの特定のコミット、ブランチ、またはタグからクローンされます。 | Git ベースのジョブを実行するには、Launch エージェント に ビルダー と Git リポジトリの認証情報 を 設定 する必要があります。 |
| イメージ ベースの ジョブ | コードやその他のアセットは Docker イメージに焼き込まれます。 | イメージ ベースのジョブを実行するには、必要に応じて Launch エージェント に イメージ リポジトリの認証情報 を 設定 する必要があります。 | 

{{% alert %}}
Launch ジョブは モデル のトレーニング 以外の処理も実行できます。例えば、Triton 推論 サーバーに モデル をデプロイするなどです。ただし、すべてのジョブは成功裏に完了するために `wandb.init` を呼び出す必要があります。これにより、W&B Workspace にトラッキング用の run が作成されます。
{{% /alert %}}

作成したジョブは W&B App の Project の Workspace にある `Jobs` タブで確認できます。そこから、ジョブを 設定 して [Launch キュー]({{< relref path="#launch-queue" lang="ja" >}}) に送信し、さまざまな [ターゲット リソース]({{< relref path="#target-resources" lang="ja" >}}) で実行できます。

### Launch キュー
Launch *キュー* は、特定の ターゲット リソース で実行するジョブの順序付きリストです。Launch キューは先入れ先出し (FIFO) です。作成できるキューの数に実質的な制限はありませんが、目安は ターゲット リソース ごとに 1 つのキューです。ジョブは W&B App の UI、W&B CLI、または Python SDK からエンキューできます。次に、1 つ以上の Launch エージェント を 設定 して、キューからアイテムを取得し、そのキューの ターゲット リソース 上で実行できます。

### ターゲット リソース
Launch キューがジョブを実行するように設定された計算 環境 を、 *ターゲット リソース* と呼びます。

W&B Launch は次の ターゲット リソース をサポートします:
- [Docker]({{< relref path="/launch/set-up-launch/setup-launch-docker.md" lang="ja" >}})
- [Kubernetes]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ja" >}})
- [AWS SageMaker]({{< relref path="/launch/set-up-launch/setup-launch-sagemaker.md" lang="ja" >}})
- [GCP Vertex]({{< relref path="/launch/set-up-launch/setup-vertex.md" lang="ja" >}})

各 ターゲット リソース は、 *リソース 設定* と呼ばれる異なる一連の 設定 パラメータ を受け付けます。リソース 設定 には Launch キューごとに既定値が定義されていますが、ジョブごとに個別に上書きできます。詳細は各 ターゲット リソース のドキュメントを参照してください。

### Launch エージェント
Launch エージェント は、実行すべきジョブがないか定期的に Launch キューを確認する、軽量な常駐プログラムです。Launch エージェント がジョブを受け取ると、まずジョブ定義からイメージをビルドまたはプルし、その後 ターゲット リソース 上で実行します。

1 つのエージェントが複数のキューをポーリングすることもできますが、ポーリング対象の各キューを支えるすべての ターゲット リソース をサポートできるように、エージェントを適切に 設定 しておく必要があります。

### Launch エージェント 環境
エージェント 環境 とは、Launch エージェント が稼働し、ジョブをポーリングしている 環境 のことです。

{{% alert %}}
エージェントの実行 環境 は、キューの ターゲット リソース とは独立しています。言い換えると、必要な ターゲット リソース に アクセス できるよう十分に 設定 されている限り、エージェントはどこにデプロイしても構いません。
{{% /alert %}}