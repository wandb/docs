---
description: W&B Launch を使用して、ML ジョブを簡単にスケールおよび管理できます。
slug: /guides/launch
displayed_sidebar: default
---

import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Launch

<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

W&B Launch を使えば、デスクトップから Amazon SageMaker や Kubernetes などのコンピュータリソースにトレーニングの [runs](../runs/intro.md) を簡単にスケールできます。W&B Launch を設定したら、数回のクリックとコマンドでトレーニングスクリプトの実行、モデルの評価スイートの実行、プロダクション推論のためのモデルの準備などが迅速に行えます。

## 仕組み

Launch は、**launch jobs**、**queues**、**agents** の3つの基本コンポーネントで構成されています。

[*launch job*](./launch-terminology.md#launch-job) は、MLワークフロー内でタスクを設定し実行するための設計図です。launch job ができたら、[*launch queue*](./launch-terminology.md#launch-queue) に追加することができます。launch queue は先入れ先出し (FIFO) のキューで、Amazon SageMaker や Kubernetes クラスターなど、特定のコンピュートターゲットリソースにジョブを設定し提出できます。

ジョブがキューに追加されると、1つまたは複数の [*launch agents*](./launch-terminology.md#launch-agent) がそのキューをポーリングし、キューがターゲットにしたシステムでジョブを実行します。

![](/images/launch/launch_overview.png)

ユースケースに基づいて、あなた（またはチームの誰か）が選んだ [コンピュートリソースターゲット](./launch-terminology.md#target-resources)（例：Amazon SageMaker）に従って launch queue を設定し、自分のインフラストラクチャーに launch agent をデプロイします。

Launch job やキューの仕組み、launch agent、W&B Launch の詳細については、[Terms and concepts](./launch-terminology.md) ページをご覧ください。

## 開始方法

ユースケースに応じて、W&B Launch を始めるための以下のリソースをご参照ください：

* 初めて W&B Launch を使用する場合は、[Walkthrough](./walkthrough.md) ガイドをお勧めします。
* [W&B Launch の設定方法](./setup-launch.md) を学びましょう。
* [launch job](./create-launch-job.md) を作成します。
* 一般的なタスクのテンプレート（[Triton へのデプロイ](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton) や [LLM の評価](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals) など）を確認するために、W&B Launch の [public jobs GitHub repository](https://github.com/wandb/launch-jobs) をチェックしてください。
    * このリポジトリから作成された launch jobs を、この公共の [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) W&B project で確認できます。