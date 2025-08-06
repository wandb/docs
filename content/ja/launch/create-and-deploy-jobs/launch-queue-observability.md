---
title: ローンチキューを監視する
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-launch-queue-observability
    parent: create-and-deploy-jobs
url: guides/launch/launch-queue-observability
---

インタラクティブな **Queue monitoring dashboard** を使うことで、launch queue がどのくらい使われているか（高負荷かアイドルか）、実行中のワークロードの可視化、非効率なジョブの特定ができます。launch queue dashboard は、計算用ハードウェアやクラウドリソースを有効活用できているかどうかを判断する際に特に役立ちます。

さらに詳細な分析を行いたい場合は、そのページから W&B 実験管理ワークスペースや Datadog、NVIDIA Base Command やクラウドコンソールのような外部のインフラストラクチャーモニタリングサービスへリンクできます。

{{% alert %}}
Queue monitoring dashboard は、現在 W&B Multi-tenant Cloud デプロイメントオプションでのみ利用可能です。
{{% /alert %}}

## ダッシュボードとプロット
**Monitor** タブを使って、過去 7 日間でキュー上で発生したアクティビティを確認できます。左側のパネルで、期間、グループ分け、フィルタを選択できます。

ダッシュボードには、パフォーマンスや効率に関するよくある質問に答える様々なプロットが用意されています。以下のセクションでは、queue ダッシュボードの UI 要素について説明します。

### ジョブステータス
**Job status** プロットは、各時間ごとに実行中、保留中、キューイング中、完了したジョブの数を表示します。**Job status** プロットを使うと、queue がアイドル状態だった期間を特定できます。

{{< img src="/images/launch/launch_obs_jobstatus.png" alt="ジョブステータスのタイムライン" >}}

例えば、固定リソース（DGX BasePod など）を持っている場合、そのリソースがアイドルだった時間帯が見えれば、sweeps のような優先度の低いプリエンプティブな launch job を追加で走らせるチャンスかもしれません。

一方で、クラウドリソースを利用しており、定期的にアクティビティのバーストが起こっている場合、特定の時間帯だけリソースを予約してコストを抑える、といった運用も検討できます。

プロットの右側には、[launch ジョブのステータス]({{< relref path="./launch-view-jobs.md#check-the-status-of-a-job" lang="ja" >}})を示す色分けのキーが表示されます。

{{% alert %}}
`Queued` の件数は他の queue へのワークロードのシフト機会になる可能性があります。また、失敗が急増した場合は、launch ジョブの設定でサポートを必要としているユーザーがいるかもしれません。
{{% /alert %}}


### Queued time

**Queued time** プロットは、指定した日付または期間で、launch ジョブが queue 上に待機していた時間（秒単位）を表示します。

{{< img src="/images/launch/launch_obs_queuedtime.png" alt="Queued time メトリクス" >}}

x軸は指定した期間、y軸は launch ジョブが queue で待っていた時間（秒数) を表します。たとえば、ある日に 10 個の launch ジョブが待機し、それぞれ平均 60 秒待った場合、**Queue time** プロットには 600 秒と表示されます。

{{% alert %}}
**Queued time** プロットを活用して、queue の待ち時間が長く影響を受けているユーザーを特定しましょう。
{{% /alert %}}

左側の `Grouping` コントロールを使えば、ジョブごとに色を変えることができます。

どのユーザーやジョブが queue 容量不足の影響を受けているかを把握するのに役立ちます。

### Job runs

{{< img src="/images/launch/launch_obs_jobruns2.png" alt="Job runs タイムライン" >}}

このプロットは、指定した時間帯に実行されたすべてのジョブの開始・終了を run ごとに色分けして表示します。これにより、任意のタイミングで queue がどのワークロードを処理していたか、一目で把握できます。

パネル右下にある Select ツールを使って気になるジョブを範囲選択すると、下部のテーブルに詳細が表示されます。


### CPU ・ GPU 使用状況
**GPU use by a job**、**CPU use by a job**、**GPU memory by job**、**System memory by job** を使うと、launch ジョブの効率性を可視化できます。

{{< img src="/images/launch/launch_obs_gpu.png" alt="GPU 使用メトリクス" >}}

たとえば、**GPU memory by job** を使えば、W&B run の完了までに長い時間がかかっていた場合、CPU コアの使用率が低かったかどうかを確認できます。

どのプロットも、x軸は（launch ジョブで作られた）W&B run の継続時間（秒）、y軸は各メトリクスを表します。データポイントにカーソルをあわせると、run ID 、紐づく project、launch ジョブ名など run 情報が確認できます。

### エラー

**Errors** パネルには、該当 launch queue 上で発生したエラーが表示されます。具体的には、エラー発生時刻、エラーが発生した launch ジョブ名、エラーメッセージが確認できます。デフォルトでは、新しい順に並んでいます。

{{< img src="/images/launch/launch_obs_errors.png" alt="エラーログパネル" >}}

**Errors** パネルを活用して、ユーザーの問題特定・解決をサポートしましょう。

## 外部リンク

queue observability dashboard のビューは全ての queue タイプで共通ですが、実際の運用では環境ごとのモニター画面に直接アクセスしたい場面も多いでしょう。その場合は、queue observability dashboard からコンソールへ直接リンクを追加できます。

ページ下部の `Manage Links` をクリックするとパネルが表示されます。リンクしたいページの URL を入力し、ラベルを追加すれば、**External Links** セクションに表示されます。