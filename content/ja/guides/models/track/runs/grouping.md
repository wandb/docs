---
title: run を Experiments にまとめる
description: トレーニングや評価の run をまとめて、より大きな experiment を構成する
menu:
  default:
    identifier: ja-guides-models-track-runs-grouping
    parent: what-are-runs
---

個々のジョブをユニークな **group** 名で **wandb.init()** に渡すことで、Experiments 毎にグループ化できます。

## ユースケース

1. **分散トレーニング:** Experiments をトレーニング用・評価用など複数のスクリプトで分割実行し、それらを一つの大きな Experiments として扱いたい場合にグループ化が便利です。
2. **複数プロセス:** 複数の小さなプロセスをまとめて一つの Experiment にグループ化できます。
3. **K-fold クロスバリデーション:** 異なる乱数シードで行った複数の run をまとめて、一つの大きな Experiment として分析できます。K-fold クロスバリデーションの Sweeps とグループ化の [サンプル](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation) もあります。

グループ化にはいくつかの方法があります：

### 1. スクリプトで group を設定

オプションで group と `job_type` を `wandb.init()` に渡します。これにより、各 experiment ごとに専用の group ページが作成され、その中に個々の run が含まれます。例：`wandb.init(group="experiment_1", job_type="eval")`

### 2. 環境変数でグループを指定

`WANDB_RUN_GROUP` を使って run のグループ名を環境変数で指定できます。詳細は [Environment Variables]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) のドキュメントをご参照ください。**Group** 名はプロジェクト内でユニークであり、同じグループに属する全ての run で共有してください。`wandb.util.generate_id()` を使えばユニークな8文字列を生成できるので、例えば `os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()` のように利用できます。

### 3. UI でグループを設定

run の初期化後、Workspace や **Runs** ページから新しいグループに移動できます。

1. W&B のプロジェクト画面に移動します。
1. プロジェクトのサイドバーから **Workspace** または **Runs** タブを選択します。
1. リネームしたい run を検索、またはスクロールして探します。

    run 名にカーソルを合わせ、縦3点リーダーをクリックして、**Move to another group** を選びます。
1. 新しいグループを作成する場合は **New group** をクリック。グループ名を入力してフォームを送信します。
1. リストから run の新しいグループを選択して **Move** をクリックします。

### 4. UI でカラムごとにグループ化を切り替え

隠れているカラムも含め、任意のカラムで動的にグループ化できます。例えば、`wandb.Run.config` でバッチサイズや学習率を記録した場合、それらのハイパーパラメーターで Web アプリ内で動的にグループ化可能です。**Group by** 機能は [run の run group]({{< relref path="grouping.md" lang="ja" >}}) とは異なるものです。run を run group でグループ化することも可能です。run を別の run group に移動したい場合は、[UI でグループを設定]({{< relref path="#set-a-group-in-the-ui" lang="ja" >}}) を参照してください。

{{% alert %}}
run のリストでは、**Group** カラムはデフォルトで非表示になっています。
{{% /alert %}}

run を1つ以上のカラムでグループ化するには：

1. **Group** をクリックします。
1. 1つ以上のカラム名をクリックします。
1. 複数カラムを選択した場合は、ドラッグしてグループ化の優先順を変更できます。
1. フォームの外をクリックすると、メニューが閉じます。

### run の表示方法をカスタマイズ
**Workspace** または **Runs** タブから、プロジェクト内で run の表示内容をカスタマイズできます。どちらのタブも同じ表示設定を使用します。

表示するカラムをカスタマイズするには：
1. runs リスト上部の **Columns** をクリックします。
1. 非表示のカラム名をクリックすると表示されます。表示中のカラム名をクリックすると非表示になります。
  
    カラム名は、あいまい検索、完全一致検索、正規表現検索で探すこともできます。カラムをドラッグすると並び順も変更可能です。
1. カラムブラウザを閉じるには **Done** をクリックします。

任意の表示カラムで run リストをソートするには：

1. カラム名にカーソルを合わせて、アクションメニュー `...` をクリックします。
1. **Sort ascending** または **Sort descending** をクリックします。

ピン留めしたカラムは右側に表示されます。カラムをピン留め／解除するには：
1. カラム名にカーソルを合わせて、アクションメニュー `...` をクリックします。
1. **Pin column** または **Unpin column** を選んでください。

デフォルトでは、長い run 名は中央が省略表示されます。省略の仕方を変更したい場合は：

1. run リスト上部のアクションメニュー `...` をクリックします。
1. **Run name cropping** で、末尾・中央・先頭のいずれかを選択してください。

## グループ化を使った分散トレーニング

`wandb.init()` で group を設定すると、UI 上で run がデフォルトでグループ化されます。表の上部にある **Group** ボタンで表示の切り替えが可能です。[サンプルコード](https://wandb.me/grouping) で group を設定した場合の [サンプルプロジェクト](https://wandb.ai/carey/group-demo?workspace=user-carey) をご覧ください。サイドバー内の各 "Group" 行をクリックすると、その Experiment の専用 Group ページに遷移できます。

{{< img src="/images/track/distributed_training_wgrouping_1.png" alt="Grouped runs view" >}}

上記のプロジェクトページから、左サイドバーの **Group** をクリックすると、[このような](https://wandb.ai/carey/group-demo/groups/exp_5?workspace=user-carey)専用ページへ遷移します：

{{< img src="/images/track/distributed_training_wgrouping_2.png" alt="Group details page" >}}

## UI での動的グループ化

任意のカラム（たとえばハイパーパラメーター）で run をグループ化できます。以下はその例です：

* **サイドバー**: エポック数で run がグループ化されています。
* **グラフ**: 各線はグループの平均を表し、色の濃淡は分散を示します。この振る舞いはグラフ設定で変更できます。

{{< img src="/images/track/demo_grouping.png" alt="Dynamic grouping by epochs" >}}

## グループ化の解除

いつでもグループ化ボタンをクリックして group フィールドをクリアすれば、表やグラフはグループ化されない通常の状態に戻ります。

{{< img src="/images/track/demo_no_grouping.png" alt="Ungrouped runs table" >}}

## グループ化時のグラフ設定

グラフ右上の編集ボタンをクリックし、**Advanced** タブから線やシェーディングの内容を変更できます。グループごとに、平均・最小値・最大値など、どの値を表示するか選択可能です。シェーディングについても、無効化、最小値＋最大値、標準偏差、標準誤差などを切り替えられます。

{{< img src="/images/track/demo_grouping_options_for_line_plots.gif" alt="Line plot grouping options" >}}