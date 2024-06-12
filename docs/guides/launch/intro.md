---
description: "W&B Launch \u3092\u4F7F\u7528\u3057\u3066\u3001ML \u30B8\u30E7\u30D6\
  \u3092\u7C21\u5358\u306B\u30B9\u30B1\u30FC\u30EB\u304A\u3088\u3073\u7BA1\u7406\u3057\
  \u307E\u3059\u3002"
slug: /guides/launch
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Launch

<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

W&B Launchを使用すると、デスクトップからAmazon SageMakerやKubernetesなどのコンピュートリソースにトレーニングの[runs](../runs/intro.md)を簡単にスケールできます。W&B Launchが設定されると、数回のクリックとコマンドでトレーニングスクリプト、モデルの評価スイート、プロダクション推論のためのモデルの準備などを迅速に実行できます。

## 仕組み

Launchは、**launch jobs**、**queues**、および**agents**の3つの基本コンポーネントで構成されています。

[*launch job*](./launch-terminology.md#launch-job)は、MLワークフローでタスクを設定し実行するための設計図です。launch jobを持っていると、それを[*launch queue*](./launch-terminology.md#launch-queue)に追加できます。launch queueは、Amazon SageMakerやKubernetesクラスターなどの特定のコンピュートターゲットリソースにジョブを設定し送信できる先入れ先出し（FIFO）キューです。

ジョブがキューに追加されると、1つ以上の[*launch agents*](./launch-terminology.md#launch-agent)がそのキューをポーリングし、キューがターゲットとするシステムでジョブを実行します。

![](/images/launch/launch_overview.png)

ユースケースに基づいて、あなた（またはチームの誰か）が選択した[コンピュートリソースターゲット](./launch-terminology.md#target-resources)（例えばAmazon SageMaker）に従ってlaunch queueを設定し、自身のインフラストラクチャーにlaunch agentをデプロイします。

Launch jobs、queuesの仕組み、launch agents、その他W&B Launchの詳細については、[Terms and concepts](./launch-terminology.md)ページをご覧ください。

## 開始方法

ユースケースに応じて、W&B Launchの開始に役立つ以下のリソースを参照してください：

* 初めてW&B Launchを使用する場合は、[Walkthrough](./walkthrough.md)ガイドを参照することをお勧めします。
* [W&B Launch](./setup-launch.md)の設定方法を学びます。
* [launch job](./create-launch-job.md)を作成します。
* TritonへのデプロイやLLMの評価などの一般的なタスクのテンプレートについては、W&B Launchの[public jobs GitHubリポジトリ](https://github.com/wandb/launch-jobs)をチェックしてください。
    * このリポジトリから作成されたlaunch jobsは、W&Bのこの公開[`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs)プロジェクトで確認できます。