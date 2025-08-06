---
title: run とは何ですか？
description: W&B の基本的な構成要素である Runs について学びましょう。
menu:
  default:
    identifier: what-are-runs
    parent: experiments
weight: 5
url: guides/runs
cascade:
- url: guides/runs/:filename
---

_A run_ とは、W&B が記録する 1 回の計算単位です。W&B の Run は、プロジェクト全体の基本的な要素だと考えてください。つまり、それぞれの run は、モデルのトレーニングや結果のログ、ハイパーパラメーターのスイープなど、特定の計算の記録です。

run を開始する一般的なパターンの例として、以下が挙げられます。

* モデルのトレーニング
* ハイパーパラメーターを変更して新しい実験を行う
* 別のモデルを使って新しい機械学習実験を実施する
* データやモデルを [W&B Artifact]({{< relref "/guides/core/artifacts/" >}}) として記録する
* [W&B Artifact をダウンロードする]({{< relref "/guides/core/artifacts/download-and-use-an-artifact.md" >}})

W&B は、作成した run を [*projects*]({{< relref "/guides/models/track/project-page.md" >}}) に保存します。W&B App 内の各 project の workspace で run の内容やプロパティを確認でき、[`wandb.Api.Run`]({{< relref "/ref/python/sdk/classes/run.md" >}}) オブジェクトを使ってプログラムから run のプロパティへアクセスすることも可能です。

`wandb.Run.log()` で記録した内容は、その run に保存されます。

```python
import wandb

entity = "nico"  # 自分の W&B entity に置き換えてください
project = "awesome-project"

with wandb.init(entity=entity, project=project) as run:
    run.log({"accuracy": 0.9, "loss": 0.1})
```

1 行目で W&B の Python SDK をインポートし、2 行目で entity `nico` の `awesome-project` という project に run を初期化、3 行目で accuracy と loss の値をその run にログします。

ターミナル内では次のような表示が返されます。

```bash
wandb: Syncing run earnest-sunset-1
wandb: ⭐️ View project at https://wandb.ai/nico/awesome-project
wandb: 🚀 View run at https://wandb.ai/nico/awesome-project/runs/1jx1ud12
wandb:                                                                                
wandb: 
wandb: Run history:
wandb: accuracy ▁
wandb:     loss ▁
wandb: 
wandb: Run summary:
wandb: accuracy 0.9
wandb:     loss 0.5
wandb: 
wandb: 🚀 View run earnest-sunset-1 at: https://wandb.ai/nico/awesome-project/runs/1jx1ud12
wandb: ⭐️ View project at: https://wandb.ai/nico/awesome-project
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241105_111006-1jx1ud12/logs
```

ターミナルに出力される URL をクリックすると、その run の workspace（W&B App UI の run 個別ページ）にアクセスできます。workspace 上のパネルは、今回の run で記録した 1 点のみを表します。

{{< img src="/images/runs/single-run-call.png" alt="Single run workspace" >}}

単一時点でのメトリクス記録だけでは十分でない場合もあります。例えば識別モデルのトレーニング時には、定期的な間隔でメトリクスをログするのが一般的です。次のコードスニペットを参考にしてください。

```python
import wandb
import random

config = {
    "epochs": 10,
    "learning_rate": 0.01,
}

with wandb.init(project="awesome-project", config=config) as run:
    print(f"lr: {config['learning_rate']}")
      
    # トレーニング run のシミュレーション
    for epoch in range(config['epochs']):
      offset = random.random() / 5
      acc = 1 - 2**-epoch - random.random() / (epoch + 1) - offset
      loss = 2**-epoch + random.random() / (epoch + 1) + offset
      print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
      run.log({"accuracy": acc, "loss": loss})
```

このコードを実行すると次のような出力が得られます。

```bash
wandb: Syncing run jolly-haze-4
wandb: ⭐️ View project at https://wandb.ai/nico/awesome-project
wandb: 🚀 View run at https://wandb.ai/nico/awesome-project/runs/pdo5110r
lr: 0.01
epoch=0, accuracy=-0.10070974957523078, loss=1.985328507123956
epoch=1, accuracy=0.2884687745057535, loss=0.7374362314407752
epoch=2, accuracy=0.7347387967382066, loss=0.4402409835486663
epoch=3, accuracy=0.7667969248039795, loss=0.26176963846423457
epoch=4, accuracy=0.7446848791003173, loss=0.24808611724405083
epoch=5, accuracy=0.8035095836268268, loss=0.16169791827329466
epoch=6, accuracy=0.861349032371624, loss=0.03432578493587426
epoch=7, accuracy=0.8794926436276016, loss=0.10331872172219471
epoch=8, accuracy=0.9424839917077272, loss=0.07767793473500445
epoch=9, accuracy=0.9584880427028566, loss=0.10531971149250456
wandb: 🚀 View run jolly-haze-4 at: https://wandb.ai/nico/awesome-project/runs/pdo5110r
wandb: Find logs at: wandb/run-20241105_111816-pdo5110r/logs
```

このトレーニングスクリプトは `wandb.Run.log()` を10回呼び出します。呼び出しのたびに、そのエポックの accuracy と loss が記録されます。端末に表示された URL から run の workspace にアクセスできます。

W&B はこの一連のトレーニングループを `jolly-haze-4` という一つの run として記録します。これはスクリプトで `wandb.init()` を 1 回だけ呼び出しているためです。

{{< img src="/images/runs/run_log_example_2.png" alt="Training run with logged metrics" >}}

別の例として、[sweep]({{< relref "/guides/models/sweeps/" >}}) の実行では、W&B が指定したハイパーパラメータ探索空間を探索します。スイープによって作成された新しいハイパーパラメータの組み合わせごとに、W&B はユニークな run を生成します。

## W&B Run の初期化

[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) で W&B Run を初期化します。以下に、W&B の Python SDK をインポートし run を初期化するコード例を示します。

山括弧（`< >`）の値は自身の値に置き換えてください。

```python
import wandb

with wandb.init(entity="<entity>", project="<project>") as run:
    # ここに自身のコード
```

run を初期化すると、W&B は指定した project フィールドの project にその run を記録します（`wandb.init(project="<project>"`）。project が存在しなければ新規作成、既に存在する場合はその project に記録されます。

{{% alert %}}
project 名を指定しない場合は、run は `Uncategorized` という project に保存されます。
{{% /alert %}}

W&B の各 run には [*run ID* というユニークな識別子]({{< relref "#unique-run-identifiers" >}}) が付与されます。[独自のユニーク ID を指定する]({{< relref "#unique-run-identifiers" >}}) ことも、自動生成に任せることもできます。

run には、人が読みやすくユニークではない [run name]({{< relref "#name-your-run" >}}) もあります。run の名前も自分で指定することも、W&B にランダムな名前を生成させることもできます。初期化後にリネームもできます。

例えば次のコード例を参照してください。

```python title="basic.py"
import wandb

run = wandb.init(entity="wandbee", project="awesome-project")
```
このスニペットの出力例は次のようになります。

```bash
🚀 View run exalted-darkness-6 at: 
https://wandb.ai/nico/awesome-project/runs/pgbn9y21
Find logs at: wandb/run-20241106_090747-pgbn9y21/logs
```

このスニペットで id パラメータが指定されていないため、W&B がユニークな run ID を生成します。ここでは `nico` が entity、`awesome-project` が project、`exalted-darkness-6` が run の名前、`pgbn9y21` が run ID です。

{{% alert title="Notebook の場合" %}}
run の最後に `run.finish()` を指定して run の終了を明示してください。これにより、run がプロジェクトに正しく記録され、バックグラウンドで継続するのを防げます。

```python title="notebook.ipynb"
import wandb

run = wandb.init(entity="<entity>", project="<project>")
# トレーニングやログ記録など
run.finish()
```
{{% /alert %}}

[run をグループ化]({{< relref "grouping.md" >}}) して experiments を構成した場合、run をグループ間で移動したり、グループ内外に移動することができます。

各 run には、その現在の状態を示す state があります。run のあらゆる状態については [Run states]({{< relref "#run-states" >}}) を参照してください。

## Run states
run の状態は次のとおりです。

| State | 説明 |
| ----- | ----- |
| `Crashed` | 内部プロセスでハートビートの送信が停止。マシンのクラッシュ等で発生することがあります。 | 
| `Failed` | 非ゼロの終了ステータスで run が終了した場合。 | 
| `Finished`| run が終了し、データが完全に同期された状態。または `wandb.Run.finish()` が呼び出された場合。 |
| `Killed` | run が終了前に強制的に停止された場合。 |
| `Running` | run が現在も稼働中で、最近ハートビートを送信している状態。  |

## Unique run identifiers

run ID は run ごとのユニークな識別子です。デフォルトでは、[新しい run を初期化する際に W&B がランダムな run ID を自動生成します]({{< relref "#autogenerated-run-ids" >}})。または [自分で run ID を指定する]({{< relref "#custom-run-ids" >}}) こともできます。

### Autogenerated run IDs

run の初期化時に run ID を指定しないと、W&B がランダムな run ID を自動生成します。run のユニーク ID は W&B App で確認できます。

1. [W&B App](https://wandb.ai/home) にアクセスします。
2. run の初期化時に指定した W&B project に移動します。
3. project の workspace 内で **Runs** タブを選択します。
4. **Overview** タブを選択します。

W&B は **Run path** 欄にユニーク run ID を表示します。run path は team 名、project 名、run ID で構成され、ID 部分が最後です。

例えば、以下の画像ではユニーク run ID は `9mxi1arc` です。

{{< img src="/images/runs/unique-run-id.png" alt="Run ID location" >}}

### Custom run IDs
独自の run ID を指定する場合は、[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) の `id` パラメータに値を渡してください。

```python 
import wandb

run = wandb.init(entity="<project>", project="<project>", id="<run-id>")
```

run のユニーク ID を使って、直接その run の overview ページにアクセスできます。例えば次のような URL 形式です。

```text title="W&B App URL for a specific run"
https://wandb.ai/<entity>/<project>/<run-id>
```

山括弧 (`< >`) 内の値は、entity・project・run ID の実際の値に置き換えてください。

## run に名前をつける 
run の名前は人が読みやすいユニークでない識別子です。

デフォルトでは、W&B が run の初期化時にランダムな名前を生成します。run 名はプロジェクトの workspace や [run の overview ページ]({{< relref "#overview-tab" >}}) の最上部に表示されます。

{{% alert %}}
run 名は workspace で run を素早く識別するのに役立ちます。
{{% /alert %}}

run の名前は、[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) の `name` パラメータで指定できます。

```python 
import wandb

with wandb.init(entity="<project>", project="<project>", name="<run-name>") as run:
    # ここに自身のコード
```

### run の名前を変更する

run 初期化後は、workspace または **Runs** ページから run 名を変更できます。

1. 該当する W&B プロジェクトに移動します。
1. サイドバーから **Workspace** もしくは **Runs** タブを選択します。
1. 名前変更したい run を検索またはスクロールで見つけます。

    run 名にカーソルを合わせて、縦三点リーダー（…）をクリックし、変更範囲を選択します
    - **Rename run for project**: プロジェクト全体で run 名を変更
    - **Rename run for workspace**: その workspace のみ run 名を変更
1. 新しい名前を入力（ランダム名を生成したい場合は空欄で可）
1. フォームを送信すると新しい run 名が表示されます。workspace 上でカスタム名となった run には情報アイコンが表示され、詳細はマウスオーバーで確認できます。

[レポート]({{< relref "/guides/core/reports/edit-a-report.md" >}}) 内の run set からリネームもできます。

1. レポート内で鉛筆アイコンをクリックし編集モードにします。
1. run set 内でリネームしたい run 名にカーソルを合わせて縦三点リーダー（…）をクリックし、下記いずれかを選択します

  - **Rename run for project**: プロジェクト全体で run 名を変更。ランダム名生成は空欄で OK。
  - **Rename run for panel grid**: レポート内のみ run 名を変更（他のコンテキストでは変更されません）。ランダム名生成は未対応です。

  必要事項記入後、フォーム送信。
1. **Publish report** をクリックして公開。

## run にノートを追加する
特定の run に追加したノートは **Overview** タブや project ページの run 一覧テーブルに表示されます。

1. W&B プロジェクトに移動
2. サイドバーから **Workspace** タブを選択
3. ノートを追加したい run をラン選択器から選択
4. **Overview** タブを選択
5. **Description** 欄右の鉛筆アイコンをクリックしてノートを記入

## run を停止する
run は W&B App から、またはプログラム上で停止できます。

{{< tabpane text=true >}}
  {{% tab header="プログラムから停止" %}}
1. run を初期化したターミナルまたはコードエディターへ移動します。
2. `Ctrl+D` を押して run を停止します。

手順通りに進むと、ターミナルには以下のような表示になります。

```bash
KeyboardInterrupt
wandb: 🚀 View run legendary-meadow-2 at: https://wandb.ai/nico/history-blaster-4/runs/o8sdbztv
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20241106_095857-o8sdbztv/logs
```

W&B App で run が非アクティブであることを確認してください。

1. ログを記録していた project に移動
2. run 名を選択 
  {{% alert %}}
  停止した run 名は端末やエディタの出力にも表示されます（例: `legendary-meadow-2`）。
  {{% /alert %}}
3. サイドバーから **Overview** タブを選択

**State** 欄で run の状態が `running` から `Killed` に変わっていることを確認できます。

{{< img src="/images/runs/stop-run-terminal.png" alt="Run stopped via terminal" >}}  
  {{% /tab %}}
  {{% tab header="W&B App から停止" %}}

1. run を記録している project にアクセス
2. 停止したい run を run 選択器で選択
3. サイドバーから **Overview** タブを選択
4. **State** 欄横の上部ボタンから run を停止
{{< img src="/images/runs/stop-run-manual.png" alt="Manual run stop button" >}}

**State** 欄で run の状態が `running` から `Killed` へ更新されます。

{{< img src="/images/runs/stop-run-manual-status.png" alt="Run status after manual stop" >}}  
  {{% /tab %}}
{{< /tabpane >}}

run の状態の詳細一覧は [State fields]({{< relref "#run-states" >}}) を参照してください。

## ログ済み run を確認する

各 run の現状、記録した artifact、ログファイルなど run に関する様々な情報を確認できます。

{{< img src="/images/runs/demo-project.gif" alt="Project navigation demo" >}}

特定の run を表示する手順は下記の通りです。

1. [W&B App](https://wandb.ai/home) にアクセス
2. 該当する W&B project に移動
3. project のサイドバーから **Workspace** タブを選択
4. run 選択器で見たい run をクリック（または run 名で絞り込み検索）

各 run の URL の形式は次のようになります。

```text
https://wandb.ai/<team-name>/<project-name>/runs/<run-id>
```

山括弧 (`< >`) 内は、team 名、project 名、run ID を表しています。

### run の表示方法をカスタマイズ
ここでは、プロジェクトの **Workspace** と **Runs** タブ（どちらも共通の表示設定）で run 一覧をカスタマイズする方法を紹介します。

{{% alert %}}
ワークスペース 1 つあたりの run 表示上限は最大 1000 件です。
{{% /alert %}}

表示するカラムの変更方法:
1. サイドバーで **Runs** タブを選択
1. run テーブル上部の **Columns** をクリック
1. 非表示カラム名をクリックで表示、表示中カラム名をクリックで非表示化
  
    カラム名による検索（部分一致/完全一致/正規表現）も可能。カラム順はドラッグで並び替えできます。
1. **Done** を押してカラム選択ウインドウを閉じます。

任意のカラムで run 一覧をソートするには:

1. カラム名にマウスを当て、アクション用 `...` メニューをクリック
1. **Sort ascending** または **Sort descending** を選択

ピン留めカラムは右側、未ピン留めカラムは **Runs** タブ左側で、**Workspace** タブでは非表示です。

カラムをピン留めするには:
1. サイドバーで **Runs** タブ
1. **Pin column** をクリック

ピン留め解除は:
1. サイドバーで **Workspace** または **Runs** タブ
1. カラム名にマウスを当て `...` メニューから **Unpin column** をクリック

デフォルトでは run 名が中間で省略表示されます。run 名の省略位置を変えるには:

1. run 一覧上部の `...` メニューをクリック
1. **Run name cropping** で先頭/中間/末尾の省略を設定

詳細は [**Runs** タブ]({{< relref "/guides/models/track/project-page.md#runs-tab" >}}) をご覧ください。

### Overview タブ
**Overview** タブでは、プロジェクト内の run の詳細情報を確認できます:

* **Author**: run の作成 entity
* **Command**: run を初期化したコマンド
* **Description**: run の説明（作成時未指定なら空）。W&B App UI や Python SDK から追記可能
* **Tracked Hours**: 実際に計算・記録していた合計時間（待機・休止時間を除く、実処理時間の指標）
* **Runtime**: run 開始から終了までの合計経過時間（休止・待機も含む、ウォールクロック時間）
* **Git repository**: 関連付けられた Git リポジトリ（[Git 有効化]({{< relref "/guides/models/app/settings-page/user-settings.md#personal-github-integration" >}}) が必要）
* **Host name**: run を計算したマシン名（ローカルの場合は自身のマシン名）
* **Name**: run の名前
* **OS**: run を開始した OS
* **Python executable**: run 開始コマンド
* **Python version**: run を作成した Python バージョン
* **Run path**: `entity/project/run-ID` 形式のユニーク識別名
* **Start time**: run 開始時刻
* **State**: [run の状態]({{< relref "#run-states" >}})
* **System hardware**: run 実行マシンのハードウェア構成
* **Tags**: 文字列リスト（`baseline` や `production` など run の分類や一時ラベルに利用）
* **W&B CLI version**: run 実行コマンドがインストールされた W&B CLI バージョン
* **Git state**: run 作成時のリポジトリや作業ディレクトリの直近コミット SHA（Git 有効化や Git 情報未取得時は空）

Overview セクション下には次の情報が続きます。

* **Artifact Outputs**: run で生成された artifact
* **Config**: [`wandb.Run.config`]({{< relref "/guides/models/track/config.md" >}}) で保存された設定パラメータ一覧
* **Summary**: [`wandb.Run.log()`]({{< relref "/guides/models/track/log/" >}}) で保存されたサマリーパラメータ一覧（デフォルトで最後の値を保持）

{{< img src="/images/app_ui/wandb_run_overview_page.png" alt="W&B Dashboard run overview tab" >}}

サンプルプロジェクト overview の例は [こちら](https://wandb.ai/stacey/deep-drive/overview)。

### Workspace タブ
Workspace タブでは、自動生成／カスタムプロット、システムメトリクスなど様々な可視化を閲覧・検索・グループ化／レイアウトできます。

{{< img src="/images/app_ui/wandb-run-page-workspace-tab.png" alt="Run workspace tab" >}}

サンプルプロジェクト workspace の例は [こちら](https://wandb.ai/stacey/deep-drive/workspace?nw=nwuserstacey)

### Runs タブ

Runs タブでは、run の絞り込み、グループ化、並び替えが行えます。

{{< img src="/images/runs/run-table-example.png" alt="Runs table" >}}

Runs タブで出来る一般的な操作例を下記に示します。

{{< tabpane text=true >}}
   {{% tab header="カラムのカスタマイズ" %}}
Runs タブには run の詳細が多数のカラムで表示されます。

- 全カラム表示は横スクロールで確認。
- カラム順はドラッグで左右移動。
- カラム名にカーソルを置いてアクションメニュー `...` から **Pin column** でピン留め。ピン留め済みは **Name** の右側に表示。**Unpin column** で解除。
- **Hide column** で非表示へ変更。現在の非表示一覧は **Columns** から一覧表示。
- 複数カラムの表示・非表示・ピン留め・解除は **Columns** から一括操作。
  - 非表示カラム名クリックで表示
  - 表示カラム名クリックで非表示
  - 表示カラム横のピンアイコンでピン留め

Runs タブのカスタマイズは [Workspace タブ]({{< relref "#workspace-tab" >}}) の selector にも反映されます。

   {{% /tab %}}

   {{% tab header="並び替え" %}}
任意カラムで Table の全行をソートできます。

1. カラムタイトルにマウスオーバー、ケバブメニュー（三点リーダー）が表示されます。
2. ケバブメニュー（三点リーダー）をクリック。
3. **Sort Asc** か **Sort Desc** で昇順／降順に並び替え。

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="Confident predictions" >}}

この画像は `val_acc` カラムでソートオプションを確認する様子です。
   {{% /tab %}}
   {{% tab header="フィルター" %}}
**Filter** ボタンから一覧を条件で絞り込めます。

{{< img src="/images/data_vis/filter.png" alt="Incorrect predictions filter" >}}

**Add filter** をクリックして条件を追加。左から順にカラム名・演算子・値を指定します。

|                   | カラム名 | 二項関係演算子    | 値       |
| -----------       | ----------- | ----------- | ----------- |
| 入力例            | 文字列       |  =, ≠, ≤, ≥, IN, NOT IN,  | 整数, 浮動小数, 文字列, タイムスタンプ, null |

式エディタはオートコンプリートで候補を提示。"and" "or" （場合によっては括弧）で条件の複合も可能です。

{{< img src="/images/data_vis/filter_example.png" alt="Run filtering example" >}}
この画像では `val_loss` カラムをもとに、検証損失が 1 以下の run を表示しています。
   {{% /tab %}}
   {{% tab header="グループ化" %}}
**Group by** ボタンで行を選択カラムの値ごとにグループ化します。

{{< img src="/images/data_vis/group.png" alt="Error distribution analysis" >}}

他の数値カラムでは、そのグループ内での値分布のヒストグラムが自動生成されます。グループ化は全体傾向の把握を助けます。

{{% alert %}}
**Group by** 機能は [run の run group]({{< relref "grouping.md" >}}) とは別物です。run group ごとのグループ化も可能ですが、run を他の run group に移動する場合は [Assign a group or job type to a run]({{< relref "#assign-a-group-or-job-type-to-a-run" >}}) を参照ください。
{{% /alert %}}

   {{% /tab %}}
{{< /tabpane >}}

### Logs タブ
**Log tab** には run 実行時の標準出力（`stdout`）、標準エラー（`stderr`）などログが表示されます。

右上の **Download** ボタンでログファイルをダウンロードできます。

{{< img src="/images/app_ui/wandb_run_page_log_tab.png" alt="Run logs tab" >}}

サンプルの logs タブは [こちら](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/logs)

### Files タブ
**Files tab** では、特定 run に関連するファイル（モデルチェックポイントや検証セットなど）が閲覧できます。

{{< img src="/images/app_ui/wandb_run_page_files_tab.png" alt="Run files tab" >}}

サンプルの files タブは [こちら](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/files/media/images)

### Artifacts タブ
**Artifacts** タブには該当 run で入出力された [artifacts]({{< relref "/guides/core/artifacts/" >}}) の一覧が表示されます。

{{< img src="/images/app_ui/artifacts_tab.png" alt="Run artifacts tab" >}}

[artifact グラフ例はこちら]({{< relref "/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" >}})

## run の削除

W&B App で project 内の run を削除できます。

1. 削除したい run を含む project に移動
2. サイドバーから **Runs** タブ
3. 削除したい run のチェックボックスを選択
4. テーブル上部の **Delete** ボタン（ゴミ箱アイコン）をクリック
5. モーダルが表示されるので **Delete** を選択

{{% alert %}}
一度削除した run ID は再利用できません。削除済み ID で run 初期化するとエラーとなり中断します。
{{% /alert %}}

{{% alert %}}
大量の run が存在する場合、削除したい run を検索バーで正規表現フィルタや、フィルタボタンで状態・タグ・他プロパティをもとに絞り込み削除できます。
{{% /alert %}}

## run の整理方法

このセクションではグループ／ジョブタイプによる run の整理方法を説明します。例えば、run を実験名ごとにグループ、処理目的（前処理・トレーニング・評価・デバッグなど）で job type に分けることで、ワークフローを効率化しモデル比較がしやすくなります。

### run へのグループやジョブタイプの割り当て

W&B の各 run には **group**（実験区分）と **job type**（処理目的）が割り当てられます:

- **Group**: 実験毎など run の大まかなカテゴリ。整理やフィルタに便利です。
- **Job type**: run の役割（例: `preprocessing`, `training`, `evaluation` など）

[参考 workspace](https://wandb.ai/stacey/model_iterz?workspace=user-stacey) では、Fashion-MNIST データセットのデータ量を増やしながらベースラインモデルを学習しています。データ量によって色で区別しています:

- **黄～濃緑**：ベースラインモデルのデータ増加
- **水色～紫～赤紫**："double" モデル（パラメータ多め）でのデータ増加

W&B のフィルタ・検索バー機能で run を下記条件で比較可能です:
- 同じデータセットでのトレーニング
- 同じテストセットでの評価

フィルターを適用すると **Table** ビューが自動で更新され、例えば「このモデルではどのクラスが特に難しいか」などモデル間の性能差の分析が可能です。