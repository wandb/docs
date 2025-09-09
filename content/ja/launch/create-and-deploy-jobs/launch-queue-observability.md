---
title: Launch キューを監視する
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-launch-queue-observability
    parent: create-and-deploy-jobs
url: guides/launch/launch-queue-observability
---

インタラクティブな **Queue monitoring dashboard** を使って、Launch キューが高負荷かアイドルかを把握し、実行中のワークロードを可視化し、非効率なジョブを見つけましょう。Launch キュー ダッシュボードは、計算ハードウェアやクラウド リソースを効果的に使えているかどうかを判断するのに特に有用です。

より深い 分析 のために、このページから W&B 実験管理 Workspace や、Datadog、NVIDIA Base Command、クラウド コンソールといった外部のインフラストラクチャー監視プロバイダーへのリンクを辿れます。

{{% alert %}}
Queue 監視ダッシュボードは現在、W&B マルチテナント クラウド デプロイメント オプションでのみ利用可能です。
{{% /alert %}}

## ダッシュボードとプロット
**Monitor** タブで、直近 7 日間にそのキューで発生したアクティビティを確認できます。左側のパネルで、時間範囲、グルーピング、フィルターを制御します。

このダッシュボードには、パフォーマンスと効率に関する よくある質問 に答えるプロットが複数含まれます。以下のセクションでは、キュー ダッシュボードの UI 要素を説明します。

### Job status
**Job status** プロットは、各時間間隔において実行中、保留、キュー中、完了のジョブ数を示します。キューがアイドルだった期間を特定するのに **Job status** プロットが役立ちます。 

{{< img src="/images/launch/launch_obs_jobstatus.png" alt="ジョブ ステータスのタイムライン" >}}

例えば、固定リソース（DGX BasePod など）を持っているとします。固定リソースでキューがアイドルになっているのを観測した場合、Sweeps のような優先度の低いプリエンプティブルな Launch ジョブを実行する好機かもしれません。

一方で、クラウド リソースを使っていて周期的にアクティビティのバーストが見られるなら、特定の時間帯にリソースを予約することでコストを節約できる可能性があります。

プロットの右側には、[Launch ジョブのステータス]({{< relref path="./launch-view-jobs.md#check-the-status-of-a-job" lang="ja" >}}) をどの色が表しているかを示すキーがあります。

{{% alert %}}
`Queued` の項目は、ワークロードを他のキューに移す機会を示しているかもしれません。失敗のスパイクは、Launch ジョブのセットアップで支援を必要としているユーザーを特定する手がかりになります。
{{% /alert %}}

### Queued time
**Queued time** プロットは、指定した日付または時間範囲において、Launch ジョブがキュー上で待機していた時間（秒）を示します。 

{{< img src="/images/launch/launch_obs_queuedtime.png" alt="キュー待機時間のメトリクス" >}}

x 軸は指定した時間枠、y 軸は Launch ジョブが Launch キューで待機していた時間（秒）です。例えば、ある日に 10 件の Launch ジョブがキュー入りしており、それぞれ平均 60 秒待っていたなら、**Queue time** プロットには 600 秒と表示されます。

{{% alert %}}
**Queued time** プロットを使って、長いキュー待ち時間の影響を受けているユーザーを特定しましょう。 
{{% /alert %}}

左側のバーにある `Grouping` コントロールで、ジョブごとの色をカスタマイズできます。これは特に、キュー容量が不足しているときに、どのユーザーやジョブが影響を受けているかを見極めるのに役立ちます。

### Job runs
{{< img src="/images/launch/launch_obs_jobruns2.png" alt="ジョブ実行のタイムライン" >}}

このプロットは、指定期間に実行されたすべてのジョブの開始と終了を、run ごとに異なる色で表示します。これにより、ある時点でキューがどのワークロードをプロセッシングしていたかを一目で把握できます。  

パネル右下の Select ツールでジョブをブラッシングすると、下部のテーブルに詳細が表示されます。

### CPU and GPU usage
**GPU use by a job**、**CPU use by a job**、**GPU memory by job**、**System memory by job** を使って、Launch ジョブの効率を確認します。 

{{< img src="/images/launch/launch_obs_gpu.png" alt="GPU 使用率メトリクス" >}}

例えば、**GPU memory by job** を使うと、ある W&B run の完了に時間がかかったかどうか、また CPU コアの使用率が低かったかどうかを確認できます。

各プロットの x 軸は、Launch ジョブによって作成された W&B run の継続時間（秒）です。データ点にマウスオーバーすると、run ID、run が属する Project、その W&B run を作成した Launch ジョブなどの情報を表示します。

### Errors
**Errors** パネルは、指定した Launch キューで発生したエラーを表示します。具体的には、エラーの発生時刻、そのエラーが発生した Launch ジョブ名、そして生成されたエラーメッセージを表示します。デフォルトでは、新しいものから古いものの順に並びます。 

{{< img src="/images/launch/launch_obs_errors.png" alt="エラー ログ パネル" >}}

**Errors** パネルを使って、ブロックされているユーザーを特定し、アンブロックしましょう。 

## 外部リンク
Queue 可観測性ダッシュボードの表示はすべてのキュー種別で一貫していますが、多くの場合、環境 固有のモニターに直接ジャンプできると便利です。そのために、Queue 可観測性ダッシュボードからコンソールへのリンクを追加します。

ページ下部の `Manage Links` をクリックしてパネルを開きます。リンクしたいページのフル URL を追加し、次にラベルを追加します。追加したリンクは **External Links** セクションに表示されます。