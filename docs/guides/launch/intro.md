---
description: W&B Launch を使用して、ML ジョブを簡単にスケールおよび管理します。
slug: /guides/launch
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Launch

<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

W&B Launchを使用すると、デスクトップからAmazon SageMakerやKubernetesなどの計算リソースにトレーニング[runs](../runs/intro.md)を簡単にスケールできます。W&B Launchが設定されると、数回のクリックとコマンドでトレーニングスクリプトの実行、モデルの評価スイートの実行、プロダクション推論のためのモデルの準備などが迅速に行えます。

## 仕組み

Launchは、**launch jobs**、**queues**、**agents**の3つの基本コンポーネントで構成されています。

[*launch job*](./launch-terminology.md#launch-job)は、MLワークフロー内のタスクを設定して実行するための設計図です。launch jobができたら、それを[*launch queue*](./launch-terminology.md#launch-queue)に追加できます。launch queueは、Amazon SageMakerやKubernetesクラスターなどの特定の計算ターゲットリソースにジョブを設定して送信できる先入れ先出し（FIFO）キューです。

ジョブがキューに追加されると、1つ以上の[*launch agents*](./launch-terminology.md#launch-agent)がそのキューをポーリングし、キューがターゲットとするシステムでジョブを実行します。

![](/images/launch/launch_overview.png)

ユースケースに基づいて、あなた（またはチームの誰か）が選択した[計算リソースターゲット](./launch-terminology.md#target-resources)（例：Amazon SageMaker）に従ってlaunch queueを設定し、自分のインフラストラクチャーにlaunch agentをデプロイします。

Launch jobs、queuesの仕組み、launch agents、W&B Launchの詳細については、[Terms and concepts](./launch-terminology.md)ページをご覧ください。

## 開始方法

ユースケースに応じて、W&B Launchの開始に役立つ以下のリソースを参照してください：

* 初めてW&B Launchを使用する場合は、[Walkthrough](./walkthrough.md)ガイドを参照することをお勧めします。
* [W&B Launch](./setup-launch.md)の設定方法を学びます。
* [launch job](./create-launch-job.md)を作成します。
* TritonへのデプロイやLLMの評価などの一般的なタスクのテンプレートについては、W&B Launchの[public jobs GitHub repository](https://github.com/wandb/launch-jobs)をチェックしてください。
    * このリポジトリから作成されたlaunch jobsは、公開されている[`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) W&Bプロジェクトで確認できます。