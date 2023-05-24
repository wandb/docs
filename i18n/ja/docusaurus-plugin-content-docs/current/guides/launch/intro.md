---
slug: /guides/launch
description: Easily scale and manage ML jobs using W&B Launch.
displayed_sidebar: default
---
# ローンチ

W&B Launchは、機械学習開発者と現代の機械学習ワークフローを支える高スケールの特 specialized ハードウェアとの間に接続層を導入します。デスクトップからGPUへのトレーニングの実行を簡単にスケールアップし、モデル評価スイートを迅速にスピンアップし、複雑なインフラストラクチャーの摩擦なしでプロダクション推論の準備ができます。

![](/images/launch/ready_to_launch.png)

## 仕組み

Launchワークフローは、**ジョブ、キュー、エージェント**という3つの基本要素によって支えられています。

![](/images/launch/Launch_Diagram.png)

* **ジョブ**は、MLワークフロー内のタスクを構成および実行するための設計図です。ジョブは、実際にはW&Bでrunをトラッキングすると自動的に作成される[アーティファクト](../../guides/artifacts/intro.md)です。各ジョブには、それが作成されるrunに関する文脈情報が含まれています。これには、ソースコード、エントリーポイント、ソフトウェアの依存関係、ハイパーパラメーター、データセットのバージョンなどがあります。

* **Launchキュー**は、ユーザーがジョブを特定の計算リソースに構成および送信できる先入れ先出し（FIFO）キューです。ローンチキュー内の各項目は、ジョブとそのジョブのパラメータ設定で構成されています。

* **Launchエージェント**は、ジョブを実行するために1つ以上のローンチキューでポーリングを行う長時間実行プロセスです。エージェントは、ジョブの元々の環境を再現するためにコンテナイメージを構築できるようになります。次に、エージェントは、構築したイメージ（または事前に作成されたイメージ）を取得し、このジョブが取得されたキューでターゲットとされるシステム上で実行できます。

## はじめ方

:::info
SDKバージョン0.14.0以上であることを確認してください。実行方法は、```
wandb --version``` です。
あなたがW&B Dedicated CloudまたはCustomer-Managed W&B展開を使用している場合、W&Bサーバーのバージョン0.30以上をご利用いただくようにしてください。

::::

ユースケースに応じて、以下のリソースを参照して、Weights & Biases Launchを開始してください。

* W&B Launchを初めて使用する場合は、[クイックスタート](./quickstart.md)ガイドを参照してください。

* この開発者ガイドでW&B Launchに関する以下のトピックを探索してください。

    * [ジョブを作成する](../launch/create-job.md)

    * [キューを作成する](../launch/create-queue.md)

    * [ジョブを実行する](../launch/launch-jobs.md)

    * [エージェントを実行する](../launch/run-agent.md)

* CLIリファレンスで[`wandb launch`](../../ref/cli/wandb-launch.md)コマンドと[`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md)コマンドを確認してください。

:::info

W&Bの営業チームに問い合わせて、あなたのビジネスにW&B Launchを設定してもらってください: [https://wandb.ai/site/pricing](https://wandb.ai/site/pricing)。

:::