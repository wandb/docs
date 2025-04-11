---
title: runs とは何ですか？
description: W&B の基本的な構成要素である Run について学びましょう。
cascade:
- url: guides/runs/:filename
menu:
  default:
    identifier: ja-guides-models-track-runs-_index
    parent: experiments
url: guides/runs
weight: 5
---

W&B の *run* とは、W&B によってログ記録された単一の計算単位のことです。W&B の run をプロジェクト全体の原子要素として考えることができます。言い換えれば、各 run とは特定の計算の記録であり、モデルのトレーニング、結果のログ記録、ハイパーパラメータースイープなどを含みます。

run を開始する一般的なパターンには以下が含まれますが、これらに限定されません：

* モデルのトレーニング
* ハイパーパラメーターを変更して新しい実験を行う
* 異なるモデルで新しい機械学習実験を行う
* データやモデルを [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) としてログ記録する
* [W&B Artifact をダウンロードする]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ja" >}})

W&B はあなたが作成した run を [projects]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) に保存します。W&B App UI 上でプロジェクトのワークスペース内の run とそのプロパティを表示できます。また、[`wandb.Api.Run`]({{< relref path="/ref/python/public-api/run.md" lang="ja" >}}) オブジェクトを使用してプログラムで run のプロパティにアクセスすることも可能です。

`run.log` でログ記録したものは、その run に記録されます。次のコードスニペットを考えてみてください。

```python
import wandb

run = wandb.init(entity="nico", project="awesome-project")
run.log({"accuracy": 0.9, "loss": 0.1})
```

最初の行は W&B Python SDK をインポートします。2 行目はエンティティ `nico` のプロジェクト `awesome-project` で run を初期化します。3 行目はその run に対するモデルの精度と損失をログ記録します。

ターミナル内で、W&B は次のように返します：

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

W&B がターミナルで返す URL は、W&B App UI 上で run のワークスペースにリダイレクトします。ワークスペースで生成されたパネルは単一のポイントに対応していることに注意してください。

{{< img src="/images/runs/single-run-call.png" alt="" >}}

単一時点でメトリクスをログ記録することはあまり有用ではないかもしれません。識別モデルのトレーニングの場合、定期的な間隔でメトリクスをログ記録することがより現実的です。以下のコードスニペットを考慮してください：

```python
epochs = 10
lr = 0.01

run = wandb.init(
    entity="nico",
    project="awesome-project",
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)

offset = random.random() / 5

# トレーニング run のシミュレーション
for epoch in range(epochs):
    acc = 1 - 2**-epoch - random.random() / (epoch + 1) - offset
    loss = 2**-epoch + random.random() / (epoch + 1) + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    run.log({"accuracy": acc, "loss": loss})
```

これは次のような出力を返します：

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

トレーニングスクリプトは `run.log` を10回呼び出します。スクリプトが `run.log` を呼び出すたびに、W&B はそのエポックの精度と損失をログ記録します。前述の出力から W&B が出力する URL を選択すると、その run のワークスペースに直接アクセスできます。

W&B は、シミュレーションしたトレーニングループを `jolly-haze-4` という単一の run 内にキャプチャします。これは、スクリプトが `wandb.init` メソッドを一度だけ呼び出しているためです。

{{< img src="/images/runs/run_log_example_2.png" alt="" >}}

別の例として、[スイープ]({{< relref path="/guides/models/sweeps/" lang="ja" >}})の際、W&B はあなたが指定したハイパーパラメーター探索空間を探索します。スイープが作成する各新しいハイパーパラメーターの組み合わせを、一意の run として実装します。

## run を初期化する

W&B run は [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) を使用して初期化します。次のコードスニペットは、W&B Python SDK をインポートして run を初期化する方法を示しています。

角括弧 (`< >`) で囲まれた値をあなた自身の値に置き換えるようにしてください：

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
```

run を初期化すると、W&B は指定したプロジェクトに対して run をログに記録します (`wandb.init(project="<project>")`)。プロジェクトがまだ存在しない場合、W&B は新しいプロジェクトを作成します。プロジェクトが既に存在する場合、W&B はそのプロジェクトに run を保存します。

{{% alert %}}
プロジェクト名を指定しない場合、W&B は run を `Uncategorized` というプロジェクトに保存します。
{{% /alert %}}

W&B の各 run には、[*run ID* という一意の識別子]({{< relref path="#unique-run-identifiers" lang="ja" >}})が付与されます。[一意の ID を指定することができます]({{< relref path="#unique-run-identifiers" lang="ja" >}}) し、または [W&B がランダムに生成してくれる]({{< relref path="#autogenerated-run-ids" lang="ja" >}}) ことも可能です。

各 run には、人間が読める[*run 名* という一意でない識別子]({{< relref path="#name-your-run" lang="ja" >}})もあります。run の名前を指定することができますし、W&B にランダムに生成させることもできます。

たとえば、次のコードスニペットを考えてみてください：

```python title="basic.py"
import wandb

run = wandb.init(entity="wandbee", project="awesome-project")
```
このコードスニペットは次の出力を生成します：

```bash
🚀 View run exalted-darkness-6 at: 
https://wandb.ai/nico/awesome-project/runs/pgbn9y21
Find logs at: wandb/run-20241106_090747-pgbn9y21/logs
```

前のコードが id 引数を指定しなかったため、W&B は一意の run ID を作成します。ここで、`nico` は run を記録したエンティティであり、`awesome-project` は run が記録されるプロジェクトの名前、 `exalted-darkness-6` は run の名前、`pgbn9y21` は run ID です。

{{% alert title="ノートブックユーザー" %}}
run の末尾で `run.finish()` を指定して run を終了したことを示してください。これにより、run がプロジェクトに正しくログ記録され、バックグラウンドで継続されないようにするのに役立ちます。

```python title="notebook.ipynb"
import wandb

run = wandb.init(entity="<entity>", project="<project>")
# トレーニングコード、ログ記録など
run.finish()
```
{{% /alert %}}

各 run には、run の現在のステータスを示す状態があります。可能な run 状態の完全なリストについては [Run states]({{< relref path="#run-states" lang="ja" >}}) を参照してください。

## Run states
次の表は、run がとりうる可能な状態を示しています：

| 状態 | 説明 |
| ----- | ----- |
| Finished | Run が終了し、完全にデータを同期した、または `wandb.finish()` を呼び出した |
| Failed | Run が終了し、非ゼロの終了ステータス |
| Crashed | Run は内部プロセスでハートビートを送信するのを停止しました（マシンがクラッシュした場合など） |
| Running | Run がまだ実行中で、最近ハートビートを送信している |

## Unique run identifiers

Run ID は run のための一意の識別子です。デフォルトでは、新しい run を初期化する際に、W&B は[ランダムで一意の run ID を生成します]({{< relref path="#autogenerated-run-ids" lang="ja" >}})。また、run を初期化する際に[独自の一意の run ID を指定することもできます]({{< relref path="#custom-run-ids" lang="ja" >}})。

### Autogenerated run IDs

run を初期化する際に run ID を指定しない場合、W&B はランダムな run ID を生成します。run の一意の ID は W&B App UI で確認できます。

1. [https://wandb.ai/home](https://wandb.ai/home) の W&B App UI にアクセスします。
2. run を初期化した際に指定した W&B プロジェクトに移動します。
3. プロジェクトのワークスペース内で、 **Runs** タブを選択します。
4. **Overview** タブを選択します。

W&B は **Run path** フィールドに一意の run ID を表示します。run path はチーム名、プロジェクト名、run ID で構成されています。一意の ID は run path の最後の部分です。

たとえば、以下の画像では、一意の run ID は `9mxi1arc` です：

{{< img src="/images/runs/unique-run-id.png" alt="" >}}

### Custom run IDs
`id` 引数を[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) メソッドに渡すことで、独自の run ID を指定することができます。

```python 
import wandb

run = wandb.init(entity="<project>", project="<project>", id="<run-id>")
```

run の一意の ID を使用して W&B App UI の run の概要ページに直接移動できます。次のセルは特定の run の URL パスを示しています：

```text title="W&B App URL for a specific run"
https://wandb.ai/<entity>/<project>/<run-id>
```

ここで、角括弧 (`< >`) で囲まれた値は、エンティティ、プロジェクト、および run ID の実際の値のためのプレースホルダーです。

## Name your run 
run の名前は、人間が読める非一意の識別子です。

デフォルトでは、W&B は新しい run を初期化する際にランダムな run 名を生成します。run の名前はプロジェクトのワークスペース内および[run の概要ページ]({{< relref path="#overview-tab" lang="ja" >}})の上部に表示されます。

{{% alert %}}
run の名前を使用してプロジェクトワークスペース内で run を素早く識別する方法として活用してください。
{{% /alert %}}

run の名前を指定するには、`name` 引数を[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) メソッドに渡します。

```python 
import wandb

run = wandb.init(entity="<project>", project="<project>", name="<run-name>")
```

## run にメモを追加

特定の run に追加したメモは、run ページの **Overview** タブやプロジェクトページの run 一覧表に表示されます。

1. あなたの W&B プロジェクトに移動します。
2. プロジェクトのサイドバーから **Workspace** タブを選択します。
3. run セレクタからメモを追加したい run を選択します。
4. **Overview** タブを選択します。
5. **Description** フィールド隣の鉛筆アイコンを選択して、メモを追加します。

## run を停止する

W&B App またはプログラムを使用して run を停止します。

{{< tabpane text=true >}}
  {{% tab header="プログラムによって" %}}
1. run を初期化したターミナルまたはコードエディタに移動します。
2. `Ctrl+D` を押して run を停止します。

たとえば、前述の手順に従うと、ターミナルは次のような状態になるかもしれません：

```bash
KeyboardInterrupt
wandb: 🚀 View run legendary-meadow-2 at: https://wandb.ai/nico/history-blaster-4/runs/o8sdbztv
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20241106_095857-o8sdbztv/logs
```

W&B App UI に移動して run がもはやアクティブではないことを確認します：

1. run のログを記録していたプロジェクトに移動します。
2. run の名前を選択します。 
  {{% alert %}}
  停止した run の名前はターミナルまたはコードエディタの出力から見つけることができます。たとえば、前の例では、run の名前は `legendary-meadow-2` です。
  {{% /alert %}}
3. プロジェクトのサイドバーから **Overview** タブを選択します。

**State** フィールドの隣で、run の状態が `running` から `Killed` に変わります。

{{< img src="/images/runs/stop-run-terminal.png" alt="" >}}  
  {{% /tab %}}
  {{% tab header="W&B App" %}}

1. run のログを記録していたプロジェクトに移動します。
2. run セレクタ内で停止したい run を選択します。
3. プロジェクトのサイドバーから **Overview** タブを選択します。
4. **State** フィールドの隣の上部ボタンを選択します。
{{< img src="/images/runs/stop-run-manual.png" alt="" >}}

**State** フィールドの隣で、run の状態が `running` から `Killed` に変わります。

{{< img src="/images/runs/stop-run-manual-status.png" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}

[State fields]({{< relref path="#run-states" lang="ja" >}}) を参照し、run の可能な状態の完全なリストを確認してください。

## ログ記録された runs を見る

run の状態、run にログ記録されたアーティファクト、run 中に記録されたログファイルなど、特定の run に関する情報を表示します。

{{< img src="/images/runs/demo-project.gif" alt="" >}}

特定の run を表示するには：

1. [https://wandb.ai/home](https://wandb.ai/home) の W&B App UI に移動します。
2. run を初期化した際に指定した W&B プロジェクトに移動します。
3. プロジェクトのサイドバー内で **Workspace** タブを選択します。
4. run セレクタ内で表示したい run をクリックするか、部分的な run 名を入力して一致する runs をフィルターします。

    デフォルトでは、長い run 名は読みやすくするために途中で切り詰められます。run 名を最初または最後で切り詰めるには、run リストの上部のアクション `...` メニューをクリックし、**Run name cropping** を最初、途中、または最後で切り取るように設定します。

特定の run の URL パスの形式は次のとおりです：

```text
https://wandb.ai/<team-name>/<project-name>/runs/<run-id>
```

ここで、角括弧 (`< >`) で囲まれた値は、チーム名、プロジェクト名、および run ID の実際の値のためのプレースホルダーです。

### Overview タブ
プロジェクト内で特定の run 情報を知るために **Overview** タブを使用してください。

* **Author**: run を作成した W&B エンティティ。
* **Command**: run を初期化したコマンド。
* **Description**: 提供した run の説明。このフィールドは、run を作成する際に説明を指定しないと空になります。W&B App UI または Python SDK を使用して run に説明を追加できます。
* **Duration**: run が実際に計算を行っている時間またはデータをログ記録している時間。ただし、任意の中断または待機時間は含まれません。
* **Git repository**: run に関連付けられた git リポジトリ。このフィールドを見るためには[git を有効にする]({{< relref path="/guides/models/app/settings-page/user-settings.md#personal-github-integration" lang="ja" >}})必要があります。
* **Host name**: W&B が run を計算する場所。ローカルマシンで run を初期化した場合、マシンの名前が表示されます。
* **Name**: run の名前。
* **OS**: run を初期化するオペレーティングシステム。
* **Python executable**: run を開始するためのコマンド。
* **Python version**: run を作成する Python バージョンを指定します。
* **Run path**: `entity/project/run-ID` 形式で一意の run ID を識別します。
* **Runtime**: run の開始から終了までの総時間を測定します。run の壁時計時間であり、run が中断したりリソースを待っている間の時間も含まれますが、duration は含みません。
* **Start time**: run を初期化した時点のタイムスタンプ。
* **State**: run の[状態]({{< relref path="#run-states" lang="ja" >}})。
* **System hardware**: W&B が run を計算するために使用するハードウェア。
* **Tags**: 文字列のリスト。タグは関連 run を一緒に整理したり、一時的なラベル（例：`baseline` や `production`）を適用するのに便利です。
* **W&B CLI version**: run コマンドをホストしたマシンにインストールされている W&B CLI バージョン。

W&B は概要セクションの下に次の情報を保存します：

* **Artifact Outputs**: run が生成したアーティファクトの出力。
* **Config**: [`wandb.config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}})で保存された設定パラメータのリスト。
* **Summary**: [`wandb.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}})で保存されたサマリーパラメータのリスト。デフォルトでは、W&B はこの値を最後にログ記録した値に設定します。

{{< img src="/images/app_ui/wandb_run_overview_page.png" alt="W&B Dashboard run overview tab" >}}

こちらでプロジェクトの概要の例を確認できます [here](https://wandb.ai/stacey/deep-drive/overview)。

### Workspace タブ
Workspace タブを使用して、生成されたプロットやカスタムプロット、システムメトリクスなどの可視化を表示、検索、グループ化、および配置してください。

{{< img src="/images/app_ui/wandb-run-page-workspace-tab.png" alt="" >}}

こちらでワークスペースの例を確認できます [here](https://wandb.ai/stacey/deep-drive/workspace?nw=nwuserstacey)

### Runs タブ

Runs タブを使用して、run をフィルタリング、グループ化、並べ替えます。

{{< img src="/images/runs/run-table-example.png" alt="" >}}

Runs タブで実行できる一般的なアクションを以下のタブで示しています。

{{< tabpane text=true >}}
   {{% tab header="カスタム列" %}}
Runs タブには、プロジェクト内の run の詳細が表示されます。デフォルトでは多くの列が表示されます。

- 表示されているすべての列を表示するには、ページを横にスクロールします。
- 列の順序を変更するには、列を左右にドラッグします。
- 列を固定するには、列名の上にカーソルを合わせ、表示されたアクションメニュー `...` をクリックしてから **Pin column** をクリックします。固定された列はページの左側に近い位置に表示されます。固定列を解除するには、**Unpin column** を選択します。
- 列を非表示にするには、列名の上にカーソルを合わせ、表示されたアクションメニュー `...` をクリックしてから **Hide column** をクリックします。現在非表示のすべての列を表示するには、**Columns** をクリックします。
- 一度に複数の列を表示、非表示、固定、または固定解除するには、**Columns** をクリックします。
  - 非表示の列の名前をクリックして表示します。
  - 表示されている列の名前をクリックして非表示にします。
  - 表示された列の横にあるピンアイコンをクリックして固定します。

Runs タブをカスタマイズすると、そのカスタマイズは [Workspace タブ]({{< relref path="#workspace-tab" lang="ja" >}}) の **Runs** セレクタにも反映されます。

   {{% /tab %}}

   {{% tab header="ソート" %}}
Table のある列の値で全行を並べ替えます。

1. 列タイトルにマウスを合わせます。ケバブメニュー（3つの垂直な点）が現れます。
2. ケバブメニュー（3つの垂直な点）を選択します。
3. 並べ替え指定を選択して、降順または昇順で行を並べ替える。

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="See the digits for which the model most confidently guessed '0'." >}}

上記の画像は、`val_acc` と呼ばれる Table 列の並べ替えオプションを表示する方法を示しています。   
   {{% /tab %}}
   {{% tab header="フィルター" %}}
フィルターボタン上の式を使用したすべての行のフィルタリング、ダッシュボード上部のフィルターボタンを使用できます。

{{< img src="/images/data_vis/filter.png" alt="See only examples which the model gets wrong." >}}

行に1つ以上のフィルターを追加するには、**Add filter** を選択します。3 つのドロップダウンメニューが表示されます。左から右へのフィルタータイプは、列名、オペレーター、値に基づいています。る。

|                   | 列名 | 二項関係    | 値       |
| -----------       | ----------- | ----------- | ----------- |
| 受け入れ値   | ストリング      |  &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | 整数、浮動小数点、ストリング、タイムスタンプ、null |


行編集案では、列名と論理述語構造に基づいてオートコンプリートを行い、各項目のオプションを示します。複数の論理述語を使用して「and」または「or」（場合によっては括弧）で1つの式に接続できます。

{{< img src="/images/data_vis/filter_example.png" alt="" >}}
上記の画像は、`val_loss` 列に基づいたフィルターを示しています。フィルターは、検証損失が1以下の run を表示します。   
   {{% /tab %}}
   {{% tab header="グループ" %}}
ダッシュボード上の **Group by** ボタンを使用して、特定の列の値で全行をグループ化します。

{{< img src="/images/data_vis/group.png" alt="The truth distribution shows small errors: 8s and 2s are confused for 7s and 9s for 2s." >}}

デフォルトでは、これにより他の数値列が、その列のグループ全体の値の分布を示すヒストグラムに変わります。グループ化は、データ内のより高水準のパターンを理解するのに役立ちます。   
   {{% /tab %}}
{{< /tabpane >}}

### System タブ
**System tab** には、CPU 使用率、システムメモリ、ディスク I/O、ネットワークトラフィック、GPU 使用率など、特定の run に対して追跡されるシステムメトリクスが表示されます。

W&B が追跡するシステムメトリクスの完全なリストについては、[System metrics]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}}) を参照してください。

{{< img src="/images/app_ui/wandb_system_utilization.png" alt="" >}}

システムタブの例はこちらから見ることができます [here](https://wandb.ai/stacey/deep-drive/runs/ki2biuqy/system?workspace=user-carey)。

### Logs タブ
**Log tab** には、標準出力 (`stdout`) や標準エラー (`stderr`) などのコマンドラインに出力されたものが表示されます。

右上の「ダウンロード」ボタンを選択してログファイルをダウンロードします。

{{< img src="/images/app_ui/wandb_run_page_log_tab.png" alt="" >}}

ログタブの例はこちらから見ることができます [here](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/logs).

### Files タブ
**Files tab** を使用して、特定の run に関連付けられたファイル（モデルチェックポイント、検証セット例など）を表示してください。

{{< img src="/images/app_ui/wandb_run_page_files_tab.png" alt="" >}}

ファイルタブの例はこちらから見ることができます [here](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/files/media/images).

### Artifacts タブ
**Artifacts** タブには、指定した run の入力および出力 [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) が一覧表示されます。

{{< img src="/images/app_ui/artifacts_tab.png" alt="" >}}

Artifacts タブの例はこちらから見ることができます [here](https://wandb.ai/stacey/artifact_july_demo/runs/2cslp2rt/artifacts).

## Run を削除する

W&B App を使用してプロジェクトから 1 つ以上の run を削除します。

1. 削除したい runs を含むプロジェクトに移動します。
2. プロジェクトのサイドバーから **Runs** タブを選択します。
3. 削除したい runs の横のチェックボックスを選択します。
4. テーブルの上部に表示される **Delete** ボタン（ゴミ箱アイコン）を選択します。
5. 表示されたモーダルで **Delete** を選択します。

{{% alert %}}
特定の ID を持つ run が削除された場合、その ID は再び使用されないことに注意してください。削除された ID で run を開始しようとするとエラーが表示され、開始が防止されます。
{{% /alert %}}

{{% alert %}}
多くの run を含むプロジェクトでは、検索バーを使用して削除したい run を正規表現を使用してフィルタリングするか、ステータス、タグ、または他のプロパティに基づいて run をフィルターするためのフィルターボタンを使用することができます。
{{% /alert %}}

## Run を整理する

このセクションでは、グループとジョブタイプを使用して run を整理する方法についての手順を紹介します。 run をグループ（たとえば、実験名）に割り当て、ジョブタイプ（たとえば、前処理、トレーニング、評価、デバッグ）を指定することで、ワークフローを簡素化し、モデルの比較を改善できます。

### Run にグループまたはジョブタイプを割り当てる

W&B の各 run は **グループ** と **ジョブタイプ** で分類できます：

- **グループ**：実験の広範なカテゴリで、run を整理およびフィルタリングするために使用されます。
- **ジョブタイプ**：run の機能で、`preprocessing` や `training`、`evaluation` のようなものです。

次の[ワークスペースの例](https://wandb.ai/stacey/model_iterz?workspace=user-stacey)では、Fashion-MNIST データセットからの増加するデータ量を使用してベースラインモデルをトレーニングしています。ワークスペースは使用されたデータ量を示すために色を使用します：

- **黄色から濃緑** は、ベースラインモデルのためのデータ量の増加を示しています。
- **薄い青から紫、マゼンタ** は、追加パラメーターを持つより複雑な「ダブル」モデルのためのデータ量を示しています。

W&B のフィルタリングオプションや検索バーを使用して、次のような特定の条件に基づいて run を比較します：
- 同じデータセットに対するトレーニング。
- 同じテストセットに対する評価。

フィルターを適用する際、**Table** ビューは自動的に更新されます。これにより、モデル間のパフォーマンスの違い（たとえば、どのクラスが他のモデルと比べてはるかに難しいか）を特定することができます。