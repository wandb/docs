---
title: Runs を Experiments に グループ化する
description: トレーニングや評価の run を、より大規模な実験にまとめる
menu:
  default:
    identifier: ja-guides-models-track-runs-grouping
    parent: what-are-runs
---

個々のジョブを実験にまとめるには、**wandb.init()** に一意の **group** 名を渡します。

## ユースケース

1. **分散トレーニング:** 実験が学習用と評価用などの別々のスクリプトに分かれていて、それらをひとつのまとまりとして扱いたい場合にグループ化を使います。
2. **複数プロセス**: 複数の小さなプロセスを 1 つの実験としてまとめます。
3. **k-fold クロスバリデーション**: 乱数シードが異なる run をまとめて、より大きな実験として把握します。こちらは、Sweeps とグループ化を使った k-fold クロスバリデーションの[例](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation)です。

グループ化を設定する方法はいくつかあります:

### 1. スクリプトで group を設定

省略可能な group と `job_type` を `wandb.init()` に渡します。これにより、各実験ごとに専用の group ページが作成され、その中に個々の run が含まれます。例:`wandb.init(group="experiment_1", job_type="eval")`

### 2. 環境変数で group を設定

`WANDB_RUN_GROUP` を使って、環境変数として run の group を指定できます。詳しくはドキュメントの [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を参照してください。**Group** はプロジェクト内で一意で、同じ group のすべての run で共有される必要があります。`wandb.util.generate_id()` を使うと、一意な 8 文字の文字列を生成してすべてのプロセスで使えます。例: `os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()`

### 3. UI で group を設定


run が初期化された後、Workspace もしくはその **Runs** ページから新しい group へ移動できます。

1. W&B のプロジェクトに移動します。
1. プロジェクトのサイドバーから **Workspace** または **Runs** タブを選択します。
1. 移動したい run を検索するかスクロールして見つけます。

    run 名にカーソルを合わせて縦三点リーダーをクリックし、**Move to another group** をクリックします。
1. 新しい group を作成するには **New group** をクリックします。group 名を入力し、フォームを送信します。
1. 一覧からその run の新しい group を選び、**Move** をクリックします。

### 4. UI で列ごとのグループ化を切り替える

非表示の列も含め、任意の列で動的にグループ化できます。たとえば `wandb.Run.config` を使って バッチサイズ や 学習率 をログしていれば、Web アプリでそれらのハイパーパラメーターごとに動的にグループ化できます。**Group by** 機能は [run の run group]({{< relref path="grouping.md" lang="ja" >}}) とは別物です。run を run group でグループ化できます。別の run group へ移動するには、[UI で group を設定]({{< relref path="#set-a-group-in-the-ui" lang="ja" >}}) を参照してください。

{{% alert %}}
run の一覧では **Group** 列はデフォルトで非表示です。
{{% /alert %}}

1 つ以上の列で run をグループ化するには:

1. **Group** をクリックします。
1. 1 つ以上の列名をクリックします。
1. 複数の列を選んだ場合は、ドラッグしてグループ化の順序を変更します。
1. フォームの外をクリックして閉じます。

### run の表示方法をカスタマイズする
プロジェクトの **Workspace** または **Runs** タブから、run の表示方法をカスタマイズできます。どちらのタブでも同じ表示設定が使われます。

表示する列をカスタマイズするには:
1. run の一覧の上で **Columns** をクリックします。
1. 非表示の列名をクリックすると表示されます。表示中の列名をクリックすると非表示になります。
  
    列名での検索は、あいまい検索、完全一致、正規表現が使えます。列をドラッグして並び順を変更できます。
1. **Done** をクリックしてカラムブラウザを閉じます。

任意の表示列で run の一覧をソートするには:

1. 列名にカーソルを合わせ、アクション `...` メニューをクリックします。
1. **Sort ascending** または **Sort descending** をクリックします。

ピン留めされた列は右側に表示されます。列をピン留め / 解除するには:
1. 列名にカーソルを合わせ、アクション `...` メニューをクリックします。
1. **Pin column** または **Unpin column** をクリックします。

デフォルトでは、読みやすさのため長い run 名は中央で省略表示されます。run 名の省略位置をカスタマイズするには:

1. run の一覧上部のアクション `...` メニューをクリックします。
1. **Run name cropping** で「末尾 / 中央 / 先頭」から選択します。

## グループ化による分散トレーニング

`wandb.init()` でグループ化を設定すると、UI ではデフォルトで run がグループ化されます。テーブル上部の **Group** ボタンでオン / オフを切り替えられます。こちらはグループ化を設定した[サンプルのプロジェクト](https://wandb.ai/carey/group-demo?workspace=user-carey)で、[サンプルコード](https://wandb.me/grouping)から生成されたものです。サイドバーの各 "Group" 行をクリックすると、その実験の専用の group ページに移動します。

{{< img src="/images/track/distributed_training_wgrouping_1.png" alt="グループ化された run のビュー" >}}

上のプロジェクトページから、左サイドバーの **Group** をクリックすると、[このようなページ](https://wandb.ai/carey/group-demo/groups/exp_5?workspace=user-carey)に移動します:

{{< img src="/images/track/distributed_training_wgrouping_2.png" alt="Group の詳細ページ" >}}

## UI での動的なグループ化

ハイパーパラメーターなど、任意の列で run をグループ化できます。以下はその例です:

* **サイドバー**: エポック数で run がグループ化されています。
* **グラフ**: 各線はその group の平均、シェーディングは分散を表します。この振る舞いはグラフの設定で変更できます。

{{< img src="/images/track/demo_grouping.png" alt="エポックによる動的グループ化" >}}

## グループ化をオフにする

グループ化ボタンをクリックして group フィールドをクリアすれば、いつでもテーブルとグラフを非グループ化の状態に戻せます。

{{< img src="/images/track/demo_no_grouping.png" alt="グループ化されていない run のテーブル" >}}

## グループ化時のグラフ設定

グラフ右上の編集ボタンをクリックし、**Advanced** タブを選んで線とシェーディングを変更します。各 group の線は、平均・最小・最大から選べます。シェーディングは、オフ、最小と最大、標準偏差、標準誤差から選べます。

{{< img src="/images/track/demo_grouping_options_for_line_plots.gif" alt="折れ線グラフのグループ化オプション" >}}