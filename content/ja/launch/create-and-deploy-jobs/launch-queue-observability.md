---
title: ローンチキューを監視する
menu:
  launch:
    identifier: launch-queue-observability
    parent: create-and-deploy-jobs
url: guides/launch/launch-queue-observability
---

インタラクティブな **Queue monitoring dashboard** を使うことで、ローンチキューが高頻度で使われているかアイドル状態かを可視化し、稼働中のワークロードや非効率なジョブを特定できます。Launch キューダッシュボードは、計算リソースやクラウドリソースを効果的に活用できているかどうかの判断に特に役立ちます。

さらに詳細な分析のために、このページから W&B experiment tracking workspace や Datadog、NVIDIA Base Command、クラウドコンソールなど外部インフラストラクチャーモニタリングプロバイダーへのリンクが用意されています。

{{% alert %}}
Queue monitoring dashboard は現在、W&B Multi-tenant Cloud デプロイメントオプションのみでご利用いただけます。
{{% /alert %}}

## ダッシュボードとプロット
**Monitor** タブを使うと、過去 7 日間に発生したキューのアクティビティを確認できます。左側のパネルで時間範囲、グルーピング、フィルターの設定が可能です。

ダッシュボードには、パフォーマンスや効率に関するよくある質問へ回答する複数のプロットが用意されています。以降のセクションでは、キューダッシュボードの UI 要素について説明します。

### Job status
**Job status** プロットは、各時間区間で実行中、保留中、キューイング中、完了したジョブ数を示します。**Job status** プロットを活用すると、キューがアイドル状態になっている期間を特定できます。

{{< img src="/images/launch/launch_obs_jobstatus.png" alt="Job status timeline" >}}

例えば、固定リソース（例えば DGX BasePod など）を持っている場合、そのリソースがアイドルになっているキューを見つけたら、優先度の低いプリエンプティブルな Launch ジョブや Sweeps を実行するチャンスかもしれません。

一方で、クラウドリソースを利用していて周期的にアクティビティが集中していれば、その時間帯にリソースを予約することでコスト削減の機会になる可能性があります。

プロットの右側には色分けで [launch job のステータス]({{< relref "./launch-view-jobs.md#check-the-status-of-a-job" >}}) を確認できるキーが表示されます。

{{% alert %}}
`Queued` 項目が多い場合、他のキューにワークロードを移すことで効率を上げることができます。また、失敗が急増している場合は、Launch ジョブのセットアップでサポートが必要なユーザーを特定できます。
{{% /alert %}}


### Queued time

**Queued time** プロットは、指定した日付または期間にローンチジョブがキューに並んでいた時間（秒単位）を表示します。

{{< img src="/images/launch/launch_obs_queuedtime.png" alt="Queued time metrics" >}}

横軸は指定した時間範囲を、縦軸はローンチジョブが Launch キュー上にあった秒数を示します。例として、ある日に 10 個の Launch ジョブがキューにあり、それぞれが平均 60 秒待っていた場合、**Queue time** プロットは 600 秒と表示されます。

{{% alert %}}
**Queued time** プロットを使って、待ち時間が長いユーザーを把握しましょう。
{{% /alert %}}

左側バーの `Grouping` コントロールを使えば、各ジョブごとに色分けをカスタマイズできます。

これにより、キューのキャパシティ不足で影響を受けているユーザーやジョブを特定するのに特に役立ちます。

### Job runs

{{< img src="/images/launch/launch_obs_jobruns2.png" alt="Job runs timeline" >}}

このプロットは、指定した期間に実行されたすべてのジョブの開始と終了をそれぞれ異なる色で表示します。これによって、いつどのワークロードがキューで処理されていたか一目で把握できます。

パネル右下の Select ツールを使い、ジョブをドラッグして選択することで、下部のテーブルに詳細を表示できます。

### CPU や GPU の使用状況
**GPU use by a job**、**CPU use by a job**、**GPU memory by job**、**System memory by job** の各プロットを使い、Launch ジョブのリソース利用効率を確認できます。

{{< img src="/images/launch/launch_obs_gpu.png" alt="GPU usage metrics" >}}

たとえば、**GPU memory by job** を用いることで、W&B run の完了までに長時間を要した際に、CPU コアやメモリがどれくらい活用されていたかを確認できます。

各プロットの横軸は、W&B run（Launch ジョブによって生成）の継続時間（秒）を示します。データポイントの上にマウスカーソルを置くと、その W&B run の ID、所属プロジェクト、対応する Launch ジョブの情報など詳細が表示されます。

### Errors

**Errors** パネルには、指定した Launch キューで発生したエラーが表示されます。具体的には、エラー発生時刻、該当する Launch ジョブ名、エラーメッセージが表示されます。デフォルトでは、新しいエラー順で並んでいます。

{{< img src="/images/launch/launch_obs_errors.png" alt="Error logs panel" >}}

**Errors** パネルを活用し、トラブルシュートが必要なユーザーの特定やブロック解除に役立ててください。

## 外部リンク

キューのオブザーバビリティダッシュボードの表示は、すべてのキュータイプで共通ですが、場合によっては環境固有のモニターに直接アクセスすると便利です。その場合は、キューオブザーバビリティダッシュボードからコンソールへのリンクを追加しましょう。

ページ下部の `Manage Links` をクリックすると、パネルが開きます。リンクしたいページの完全な URL を入力し、ラベルを追加します。追加したリンクは **External Links** セクションに表示されます。