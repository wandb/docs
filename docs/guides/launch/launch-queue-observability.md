---
displayed_sidebar: default
---


# Queue monitoring dashboard (ベータ)

インタラクティブな **Queue monitoring dashboard** を使って、launch キューの使用状況（集中利用やアイドル状態）を確認し、稼働中のワークロードを可視化し、非効率なジョブを特定できます。launch キューダッシュボードは、計算ハードウェアやクラウドリソースを効果的に使用しているかどうかを判断するのに特に有用です。

より深い分析を行うために、このページは W&B 実験管理ワークスペースおよび Datadog、NVIDIA Base Command、クラウドコンソールなどの外部インフラストラクチャー監視プロバイダーへのリンクを提供します。

:::info
Queue monitoring dashboards には W&B Weave が必要です。 W&B Weave は、Customer-managed や AWS/GCP 専用クラウドデプロイメントではまだ利用できません。詳細については、W&B の担当者にお問い合わせください。
:::

## ダッシュボードとプロット
**Monitor** タブを使用して、過去7日間のキューの活動を確認します。左側のパネルを使用して、時間範囲、グループ化、およびフィルタを制御します。

ダッシュボードには、パフォーマンスと効率性に関するよくある質問に答えるための多くのプロットがあります。以下のセクションでは、queue dashboard の UI 要素について説明します。

### Job status
**Job status** プロットは、各時間間隔で実行中、保留中、キューに入れられた、または完了したジョブの数を表示します。**Job status** プロットを使用して、キューのアイドル期間を識別します。

![](/images/launch/launch_obs_jobstatus.png)

例えば、固定リソース（DGX BasePod など）があるとします。この固定リソースでキューがアイドル状態である場合、優先度の低い pre-emptible の launch job（例えば Sweeps）を実行する機会を示唆しているかもしれません。

一方、クラウドリソースを使用している場合、定期的なアクティビティのバーストが見えることがあります。定期的なアクティビティのバーストは、特定の時間にリソースを予約してお金を節約する機会を示唆しているかもしれません。

プロットの右側には、[launch job のステータス](./launch-view-jobs.md#check-the-status-of-a-job)を示す色分けがあります。

:::tip
`Queued` 項目は、他のキューにワークロードを移す機会を示しているかもしれません。失敗のスパイクは、launch job セットアップに問題があるユーザーを特定する手掛かりになります。
:::

### Queued time

**Queued time** プロットは、指定された日付または時間範囲に対して、launch job がキューにあった時間（秒単位）を表示します。

![](/images/launch/launch_obs_queuedtime.png)

x軸には指定した期間が表示され、y軸には launch job が launch キューにあった時間（秒単位）が表示されます。例えば、ある日に10の launch job がキューに入っているとします。この場合、**Queue time** プロットは、それらの10の launch job がそれぞれ平均60秒待機していた場合、600秒と表示されます。

:::tip
**Queued time** プロットを使用して、長いキュー待ち時間に影響を受けたユーザーを特定します。
:::

カラーは左側のバーにある `Grouping` コントロールでカスタマイズできます。

これにより、キュー容量の不足で困っているユーザーやジョブを特定するのに特に役立ちます。

### Job runs

![](/images/launch/launch_obs_jobruns2.png)

このプロットは、指定した期間で実行された各ジョブの開始と終了を示し、各 run は異なる色で表示されます。これにより、特定の時間にキューが処理していたワークロードを一目で確認できます。

パネルの右下にある Select ツールを使用して、ジョブをブラッシングしてテーブルに詳細を表示します。

### CPU および GPU 使用率
**GPU use by a job**、**CPU use by a job**、**GPU memory by job**、**System memory by job** を使用して、launch jobs の効率を確認します。

![](/images/launch/launch_obs_gpu.png)

例えば、**GPU memory by job** を使用して、W&B run が完了するのに長い時間がかかり、CPU コアの低い割合しか使用していないかどうかを確認できます。

各プロットの x軸には、launch job によって作成された W&B run の持続時間（秒単位）が表示されます。データポイントにマウスをホバーすると、W&B run の情報（run ID、run が属するプロジェクト、W&B run を作成した launch job など）を表示できます。

### Errors

**Errors** パネルは、指定された launch queue で発生したエラーを表示します。具体的には、エラーが発生したタイムスタンプ、エラーが発生した launch job の名前、および生成されたエラーメッセージが表示されます。デフォルトでは、エラーは最新から古い順に並べられます。

![](/images/launch/launch_obs_errors.png)

**Errors** パネルを使用して、ユーザーの問題を特定して解決します。

## 外部リンク

queue observability dashboard のビューはすべてのキュータイプで一貫していますが、多くの場合、環境固有のモニターに直接ジャンプすることが有用です。これを達成するために、コンソールから queue observability dashboard へのリンクを追加します。

ページの下部にある `Manage Links` をクリックしてパネルを開き、リンクしたいページの完全な URL を追加します。次に、ラベルを追加します。追加したリンクは **External Links** セクションに表示されます。