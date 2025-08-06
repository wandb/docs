---
title: run を Experiments にまとめる
description: トレーニングや評価の run をまとめて、より大きな experiment を構成する
menu:
  default:
    identifier: grouping
    parent: what-are-runs
---

個々のジョブを実験としてまとめるには、**wandb.init()** に一意な **group** 名を渡してください。

## ユースケース

1. **分散トレーニング:** 実験が複数のトレーニング・評価スクリプトに分かれている場合、それらをひとまとめの大きな単位として扱いたいときにグループ化が便利です。
2. **複数プロセス:** 複数の小さなプロセスをひとつの実験としてまとめたい場合に利用します。
3. **K-fold クロスバリデーション:** 異なる乱数シードで実行した複数の run をまとめて大きな実験として可視化できます。K-fold クロスバリデーションとグループ化を使った [例](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation) もご覧ください。

グループ化の方法はいくつかあります。

### 1. スクリプト内で group を指定

`wandb.init()` に group と `job_type` をオプションで指定できます。これによりそれぞれの実験ごとに専用の group ページが作成され、その中に個々の run が含まれます。例えば:  
`wandb.init(group="experiment_1", job_type="eval")`

### 2. グループ用の環境変数を設定

環境変数として `WANDB_RUN_GROUP` を使い、run のグループを指定できます。詳しくは [Environment Variables]({{< relref "/guides/models/track/environment-variables.md" >}}) のドキュメントをご覧ください。**Group** 名はプロジェクト内で一意にし、グループに含めるすべての run で共有してください。`wandb.util.generate_id()` を利用して、全プロセスで共通の8文字ユニーク文字列を作ることもできます。例えば:  
`os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()`

### 3. UI で group を設定

run を初期化した後、Workspace や **Runs** ページから新しいグループへ移動できます。

1. W&B の Project にアクセスします。
1. プロジェクトサイドバーから **Workspace** または **Runs** タブを選択します。
1. 名前を変更したい run を検索またはスクロールして見つけます。

    run 名にカーソルを合わせて縦三点リーダー（三つの縦ドット）をクリックし、**Move to another group** を選びます。
1. 新しいグループを作成するには、**New group** をクリックしてグループ名を入力し、フォームを送信します。
1. リストから run の新グループを選び、**Move** をクリックします。

### 4. UI でカラムごとに手動でグループ化

任意のカラムで動的にグループ化できます。非表示のカラムでもグループ化可能です。例として、`wandb.Run.config` でバッチサイズや学習率をログしている場合、それらのハイパーパラメーターでグループ化して表示できます。**Group by** 機能は [run の run group]({{< relref "grouping.md" >}}) とは別物です。run group でも run をグループ化できます。run を他の run group へ移動する方法は [UI でのグループ設定]({{< relref "#set-a-group-in-the-ui" >}}) を参照してください。

{{% alert %}}
run のリストでは、**Group** カラムはデフォルトで非表示です。
{{% /alert %}}

複数カラムで run をグループ化する手順：

1. **Group** ボタンをクリック。
1. 任意のカラム名をクリックして選択。
1. 2つ以上選択した場合、ドラッグしてグループ化の順序を変更可能。
1. フォーム外をクリックして終了。

### run 表示のカスタマイズ
**Workspace** または **Runs** タブから、run 表示方法をカスタマイズできます。両タブは同じ表示設定を利用します。

表示するカラムのカスタマイズ方法：
1. run リスト上部の **Columns** をクリック。
1. 非表示カラム名をクリックすると表示、表示カラム名をクリックすると非表示化できます。
  
    カラム名の検索には、あいまい・完全一致・正規表現で探せます。ドラッグ＆ドロップで表示順序も変更可能です。
1. **Done** をクリックしてカラムブラウザを閉じます。

可視化しているカラムでリストをソートする方法：

1. カラム名にカーソルを合わせ、アクション `...` メニューをクリック。
1. **Sort ascending** または **Sort descending** をクリック。

ピン留めしたカラムは右側に固定表示されます。ピン留め／解除のやり方：
1. カラム名にカーソルを当ててアクション `...` メニューをクリック。
1. **Pin column** または **Unpin column** を選択。

デフォルトで run 名が長い場合は読みやすさのため中央で省略されます。省略方法をカスタマイズするには：

1. run リスト上部のアクション `...` メニューをクリック。
1. **Run name cropping** 設定で先頭、中央、末尾のいずれで省略するかを選べます。

## グループ化と分散トレーニング

`wandb.init()` でグループ化を指定している場合、UI ではデフォルトで run がグループ化して表示されます。テーブル上部の **Group** ボタンでこの表示はいつでも切り替え可能です。[サンプルコード](https://wandb.me/grouping) を使って生成した [プロジェクト例](https://wandb.ai/carey/group-demo?workspace=user-carey) も参照してください。サイドバーの "Group" 行をクリックすると、その実験の専用 group ページに遷移します。

{{< img src="/images/track/distributed_training_wgrouping_1.png" alt="グループ化された runs のビュー" >}}

プロジェクトページから、左サイドバーの **Group** をクリックすると、[このような](https://wandb.ai/carey/group-demo/groups/exp_5?workspace=user-carey)専用ページに遷移できます。

{{< img src="/images/track/distributed_training_wgrouping_2.png" alt="Group 詳細ページ" >}}

## UI での動的なグループ化

例えばハイパーパラメーターなど、任意のカラムで run をグループ化可能です。以下はその例です：

* **サイドバー**: run がエポック数ごとにグループ分けされています。
* **グラフ**: 各線はグループの平均値を表し、影の部分は分散を示します。この振る舞いはグラフ設定で変更可能です。

{{< img src="/images/track/demo_grouping.png" alt="エポック数による動的グループ化" >}}

## グループ化を解除する

グループ化ボタンをクリックし、グループフィールドを空にすれば、テーブルとグラフはグループ化されていない標準状態に戻ります。

{{< img src="/images/track/demo_no_grouping.png" alt="グループ化解除時の runs テーブル" >}}

## グループごとのグラフ設定

グラフ右上の編集ボタンをクリックし、**Advanced** タブで線や影の表示方法を変更できます。各グループの線で平均値・最小値・最大値から選択できます。影は非表示、min-max、標準偏差、標準誤差のいずれかを表示可能です。

{{< img src="/images/track/demo_grouping_options_for_line_plots.gif" alt="折れ線グループ化オプション" >}}