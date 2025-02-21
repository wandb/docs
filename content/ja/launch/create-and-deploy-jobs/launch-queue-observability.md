---
title: Monitor launch queue
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-launch-queue-observability
    parent: create-and-deploy-jobs
url: guides/launch/launch-queue-observability
---

インタラクティブな **Queue monitoring dashboard** を使用して、launch キューが多忙であるかアイドルであるかを確認し、実行中のワークロードを視覚化し、非効率的なジョブを発見できます。launch queue dashboard は、コンピュートハードウェアやクラウドリソースを効果的に使用しているかどうかを判断するのに特に役立ちます。

より深い分析のために、このページは W&B の実験管理ワークスペースや、Datadog、NVIDIA Base Command、クラウドコンソールなどの外部インフラストラクチャーモニタリングプロバイダーへのリンクを提供します。

{{% alert %}}
Queue monitoring dashboards は、現在 W&B Multi-tenant Cloud デプロイメントオプションでのみ利用可能です。
{{% /alert %}}

## ダッシュボードとプロット
**Monitor** タブを使用して、過去7日間に発生したキューのアクティビティを表示できます。左側のパネルを使用して、時間範囲、グループ化、フィルターを制御します。

ダッシュボードには、パフォーマンスと効率に関するよくある質問に答えるいくつかのプロットが含まれています。次節では、キューダッシュボードの UI 要素について説明します。

### ジョブステータス
**Job status** プロットは、各時間間隔において実行中、保留中、キュー待ち、または完了したジョブの数を示します。 **Job status** プロットを使用して、キューのアイドル期間を特定します。 

{{< img src="/images/launch/launch_obs_jobstatus.png" alt="" >}}

例えば、固定リソース (例えば DGX BasePod) があるとしましょう。この固定リソースでキューがアイドルである場合、低優先度のプリエンプション可能な launch ジョブ（sweeps など）を実行する機会があるかもしれません。

一方、クラウドリソースを使用し、活動の周期的な急増を観察した場合、特定の時間帯にリソースを予約することで費用を節約する機会があるかもしれません。

プロットの右側には、どの色が launch ジョブの [ステータス]({{< relref path="./launch-view-jobs.md#check-the-status-of-a-job" lang="ja" >}}) を表しているか示すキーがあります。

{{% alert %}}
`Queued` の項目は、ワークロードを他のキューにシフトする機会を示しているかもしれません。失敗の急増は、launch ジョブの設定で助けが必要なユーザーを特定するのに役立ちます。
{{% /alert %}}

### キュー待ち時間

**Queued time** プロットは、指定された日付または時間範囲で launch ジョブがキューにあった時間（秒）を示します。

{{< img src="/images/launch/launch_obs_queuedtime.png" alt="" >}}

x 軸は指定した時間枠を示し、y 軸は launch ジョブが launch キューにあった時間（秒）を示します。例えば、特定の日に 10 の launch ジョブがキュー待ちになっているとしましょう。 **Queue time** プロットは、それらの 10 の launch ジョブがそれぞれ平均 60 秒待機している場合、600 秒を示します。

{{% alert %}}
**Queued time** プロットを使用して、長いキュー待ち時間に影響を受けているユーザーを特定します。
{{% /alert %}}

左側のバーにある `Grouping` コントロールで各ジョブの色をカスタマイズできます。これは、特にキューの容量が限られているときにどのユーザーとジョブが影響を受けているかを特定するのに役立ちます。

### Job runs

{{< img src="/images/launch/launch_obs_jobruns2.png" alt="" >}}

このプロットは、特定の期間に実行されたすべてのジョブの開始と終了を表示し、各 run に異なる色を付けています。これにより、特定の時間にキューが処理していたワークロードを一目で確認できます。

パネルの右下にある Select ツールを使用して、ジョブをブラシオーバーし、以下のテーブルに詳細を入力します。

### CPU と GPU の使用率
**GPU use by a job**, **CPU use by a job**, **GPU memory by job**, **System memory by job** を使用して、launch ジョブの効率を確認します。

{{< img src="/images/launch/launch_obs_gpu.png" alt="" >}}

例えば、**GPU memory by job** を使用して、W&B run が完了するまでに長時間かかり、CPU コアの使用率が低かったかどうかを確認できます。

各プロットの x 軸は、launch ジョブによって作成された W&B run の持続時間（秒）を示しています。データポイントにマウスをホバーさせると、run ID、run が所属する project、W&B run を生成する launch ジョブなどの情報が表示されます。

### Errors

**Errors** パネルは、特定の launch キューで発生したエラーを表示します。より具体的には、Errors パネルは、エラーが発生した時点のタイムスタンプ、エラーの発生元である launch ジョブの名前、生成されたエラーメッセージを表示します。デフォルトでは、エラーは最新から古い順に並べられています。

{{< img src="/images/launch/launch_obs_errors.png" alt="" >}}

**Errors** パネルを使用して、ユーザーを特定し、ブロックを解除します。

## 外部リンク

キューの可視性ダッシュボードのビューはすべてのキュータイプで一貫していますが、多くの場合、環境固有のモニタへ直接ジャンプすると便利です。これを達成するために、キューの可視性ダッシュボードからコンソールへのリンクを追加します。

ページの下部で `Manage Links` をクリックしてパネルを開きます。目的のページのフル URL を追加し、ラベルを追加します。追加したリンクは **External Links** セクションに表示されます。