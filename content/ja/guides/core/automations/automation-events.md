---
title: 自動化イベントとスコープ
menu:
  default:
    identifier: ja-guides-core-automations-automation-events
    parent: automations
weight: 2
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

オートメーションは、プロジェクトやレジストリ内で特定のイベントが発生したときに開始できます。このページでは、各スコープごとにオートメーションをトリガーできるイベントについて説明します。オートメーションの詳細は [Automations の概要]({{< relref path="/guides/core/automations/" lang="ja" >}}) または [オートメーションの作成]({{< relref path="create-automations/" lang="ja" >}}) をご覧ください。

## Registry
このセクションでは [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) におけるオートメーションのスコープとイベントについて説明します。

1. https://wandb.ai/registry/ の **Registry** アプリにアクセスします。
1. レジストリ名をクリックし、**Automations** タブでオートメーションの確認・作成ができます。

![Registry Automations タブでオートメーションが設定されたスクリーンショット](/images/automations/registry_automations_tab.png)

[オートメーションの作成方法]({{< relref path="create-automations/" lang="ja" >}}) についてさらに詳しく学ぶことができます。

### スコープ
Registry でオートメーションを作成できるスコープは以下の通りです：
- [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) レベル: 特定のレジストリ内のすべてのコレクション（将来的に追加されたものも含む）でイベントが発生した際に監視します。
- コレクションレベル: 特定レジストリ内の単一コレクション。

### イベント
Registry オートメーションで監視できるイベントは次の通りです：
- **新しいバージョンがコレクションにリンクされたとき**：新しいモデルやデータセットが Registry に追加された際にテストや検証を実行します。
- **Artifact エイリアスが追加されたとき**：新しい artifact バージョンに特定のエイリアスが付与された際、ワークフローの特定ステップを実行します。例: `production` エイリアスが付与されたタイミングでモデルをデプロイするなど。

## Project
このセクションでは [project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) におけるオートメーションのスコープとイベントについて説明します。

1. W&B アプリの `https://wandb.ai/<team>/<project-name>` ページで目的の W&B Project にアクセスします。
1. **Automations** タブでオートメーションの確認・作成ができます。

![Project Automations タブでオートメーションが設定されたスクリーンショット](/images/automations/project_automations_tab.png)

[オートメーションの作成方法]({{< relref path="create-automations/" lang="ja" >}}) についてさらに詳しく学ぶことができます。

### スコープ
Project でオートメーションを作成できるスコープは以下の通りです：
- Project レベル: プロジェクト内のすべてのコレクションでイベントが発生した場合に監視します。
- コレクションレベル: 指定したフィルターに一致するプロジェクト内すべてのコレクション。

### Artifact 関連のイベント
artifact がオートメーションをトリガーできるイベントについて説明します。

- **artifact に新しいバージョンが追加されたとき**：artifact の各バージョンごとに定期的な処理を実行します。例: 新しい dataset artifact バージョンが作成された時にトレーニングジョブを開始するなど。
- **artifact エイリアスが追加されたとき**：プロジェクトやコレクション内の新しい artifact バージョンに特定のエイリアスが付与された際、ワークフローの特定ステップを実行します。例: `test-set-quality-check` エイリアスがついたときに一連の後処理を走らせたり、`latest` エイリアスが新規バージョンに付与されるたびにワークフローを実行できます。あるエイリアスは同時に 1 つの artifact バージョンにのみつけられます。
- **artifact タグが追加されたとき**：プロジェクトやコレクション内の artifact バージョンに特定のタグが付与された際、ワークフローの特定ステップを実行します。例: artifact バージョンに "europe" というタグがついた場合に、その地域特有のワークフローを動かします。artifact タグはグルーピングやフィルタリングに使用され、同じタグを複数の artifact バージョンに同時につけることができます。

### Run 関連のイベント
[run のステータス]({{< relref path="/guides/models/track/runs/#run-states" lang="ja" >}})が変化した場合、または [メトリクス値]({{< relref path="/guides/models/track/log/#what-data-is-logged-with-specific-wb-api-calls" lang="ja" >}})が変化した場合にオートメーションを実行できます。

#### Run ステータス変更
{{% alert %}}
- 現在は [W&B マルチテナントクラウド]({{< relref path="/guides/hosting/#wb-multi-tenant-cloud" lang="ja" >}}) のみで利用可能です。
- **Killed** ステータスの run ではオートメーションはトリガーされません。このステータスは、管理者ユーザーにより強制停止された run を意味します。
{{% /alert %}}

run の [ステータス]({{< relref path="/guides/models/track/runs/_index.md#run-states" lang="ja" >}}) が **Running**、**Finished**、**Failed** のいずれかに変わった際にワークフローを実行できます。さらに、run を開始したユーザーや run 名でフィルタリングすることで、オートメーションをトリガーする対象を絞り込むことも可能です。

![run ステータス変更によるオートメーションの例](/images/automations/run_status_change.png)

run ステータスは run 全体のプロパティであるため、run ステータスのオートメーションは **Automations** ページからのみ作成でき、ワークスペースからは作成できません。

#### Run メトリクス値の変化
{{% alert %}}
現在は [W&B マルチテナントクラウド]({{< relref path="/guides/hosting/#wb-multi-tenant-cloud" lang="ja" >}}) のみで利用可能です。
{{% /alert %}}

run の履歴に記録されたメトリクスや、`cpu` などの [システムメトリクス]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}})、つまり CPU 使用率のような値でワークフローを実行できます。W&B はシステムメトリクスを 15 秒ごとに自動でログします。

run メトリクスのオートメーションは、プロジェクトの **Automations** タブ、またはワークスペース内の折れ線パネルから直接作成できます。

run メトリックオートメーションの設定時には、指定したしきい値とどのように比較するかを構成します。選択肢はイベントタイプと指定したフィルターによって異なります。

さらに、run を開始したユーザーや run 名でフィルタリングし、オートメーションをトリガーする run を絞り込むことができます。

##### しきい値
**Run metrics threshold met** イベントの場合、以下を設定します：

1. 判定に利用する直近記録値のウィンドウサイズ（デフォルトは 5）
1. ウィンドウ内で計算する **平均**、**最小**、**最大** のいずれか
1. 比較方法の種類
      - より大きい
      - 以上
      - 未満
      - 以下
      - 等しくない
      - 等しい

例：平均 `accuracy` が `.6` を超えた時にオートメーションをトリガーするなど。

![run メトリクスしきい値オートメーションの例](/images/automations/run_metrics_threshold_automation.png)

##### 変化しきい値
**Run metrics change threshold met** イベントの場合、2つの「ウィンドウ」の値を使って判定します。

- _現在ウィンドウ_：判定に使う直近記録値（デフォルト 10）
- _直前ウィンドウ_：判定に使う直前の記録値（デフォルト 50）

現在ウィンドウと直前ウィンドウは連続しており、重複しません。

オートメーションを作成する際は以下を設定します：

1. 現在ウィンドウ（デフォルト 10）
1. 直前ウィンドウ（デフォルト 50）
1. 値の評価方法が **相対値** か **絶対値** か（デフォルトは **相対値**）
1. 比較方法の種類
      - 少なくともこれだけ増加
      - 少なくともこれだけ減少
      - 少なくともこれだけ増減

例：平均 `loss` が少なくとも `.25` 減少したときにオートメーションをトリガーするなど。

![run メトリクス変化しきい値オートメーションの例](/images/automations/run_metrics_change_threshold_automation.png)

#### Run フィルター
このセクションでは、オートメーションがどのように対象の run を絞り込むかについて説明します。

- デフォルトでは、プロジェクト内のどの run でもイベント発生時にオートメーションが走ります。特定の run だけを対象としたい場合、run フィルターを指定します。
- 各 run ごとに個別に判定され、オートメーションをトリガーできます。
- 各 run の値は別々のウィンドウに格納され、しきい値と個別に比較されます。
- 1つのオートメーションは 24 時間の間に同じ run で1回だけ発火します。

## 次のステップ
- [Slack オートメーションの作成]({{< relref path="create-automations/slack.md" lang="ja" >}})
- [Webhook オートメーションの作成]({{< relref path="create-automations/webhook.md" lang="ja" >}})