---
title: What are runs?
description: W&B の基本的な構成要素である Runs について学びましょう。
cascade:
- url: guides/runs/:filename
menu:
  default:
    identifier: ja-guides-models-track-runs-_index
    parent: experiments
url: guides/runs
weight: 5
---

*run* は、W&B によってログされる計算の単一の単位です。W&B の run は、プロジェクト全体の原子要素と考えることができます。つまり、各 run は、モデルのトレーニングと結果のログ、ハイパーパラメーターの スイープ など、特定の計算の記録です。

run を開始する一般的なパターンには、以下が含まれますが、これらに限定されません。

* モデルのトレーニング
* ハイパーパラメーターを変更して新しい 実験 を行う
* 異なるモデルで新しい 機械学習 実験 を行う
* [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) として データ またはモデルをログする
* [W&B Artifact をダウンロードする]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ja" >}})

W&B は、作成した run を [*プロジェクト*]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) に保存します。run とそのプロパティは、W&B App UI の run の プロジェクト ワークスペース 内で表示できます。また、[`wandb.Api.Run`]({{< relref path="/ref/python/public-api/run.md" lang="ja" >}}) オブジェクトを使用して、プログラムで run のプロパティに アクセス することもできます。

`run.log` でログするものはすべて、その run に記録されます。次のコードスニペットを検討してください。

```python
import wandb

run = wandb.init(entity="nico", project="awesome-project")
run.log({"accuracy": 0.9, "loss": 0.1})
```

最初の行は、W&B Python SDK をインポートします。2 行目は、エンティティ `nico` の下の プロジェクト `awesome-project` で run を初期化します。3 行目は、モデルの 精度 と 損失 をその run にログします。

ターミナル 内で、W&B は以下を返します。

```bash
wandb: Syncing run earnest-sunset-1
wandb: ⭐️ View project at https://wandb.ai/nico/awesome-project
wandb: 🚀 View run at https://wandb.ai/nico/awesome-project/runs/1jx1ud12
wandb:                                                                                
wandb: 
wandb: Run history:
wandb: accuracy  
wandb:     loss  
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

ターミナル で W&B が返す URL は、W&B App UI の run の ワークスペース にリダイレクトします。ワークスペース で生成される パネル は、単一のポイントに対応していることに注意してください。

{{< img src="/images/runs/single-run-call.png" alt="" >}}

単一の時点での メトリクス のログは、それほど役に立たない場合があります。判別モデルのトレーニングの場合のより現実的な例は、一定の間隔で メトリクス をログすることです。たとえば、次のコードスニペットを検討してください。

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

# simulating a training run
for epoch in range(epochs):
    acc = 1 - 2**-epoch - random.random() / (epoch + 1) - offset
    loss = 2**-epoch + random.random() / (epoch + 1) + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    run.log({"accuracy": acc, "loss": loss})
```

これにより、次の出力が返されます。

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

トレーニング スクリプト は `run.log` を 10 回呼び出します。スクリプト が `run.log` を呼び出すたびに、W&B はその エポック の 精度 と 損失 をログします。W&B が前の出力から出力する URL を選択すると、W&B App UI の run の ワークスペース に移動します。

スクリプト が `wandb.init` メソッド を 1 回だけ呼び出すため、W&B はシミュレートされたトレーニング ループ を `jolly-haze-4` という単一の run 内でキャプチャすることに注意してください。

{{< img src="/images/runs/run_log_example_2.png" alt="" >}}

別の例として、[sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) 中に、W&B は指定した ハイパーパラメーター 探索 空間を探索します。W&B は、sweep が作成する新しい ハイパーパラメーター の組み合わせを、一意の run として実装します。

## run を初期化する

[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) を使用して W&B run を初期化します。次のコードスニペットは、W&B Python SDK をインポートして run を初期化する方法を示しています。

山かっこ (`< >`) で囲まれた値を、自分の値に置き換えてください。

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
```

run を初期化すると、W&B は プロジェクト フィールド に指定した プロジェクト (`wandb.init(project="<project>"` に run をログします。W&B は、 プロジェクト がまだ存在しない場合は、新しい プロジェクト を作成します。プロジェクト がすでに存在する場合は、W&B はその プロジェクト に run を保存します。

{{% alert %}}
プロジェクト 名を指定しない場合、W&B は run を `Uncategorized` という プロジェクト に保存します。
{{% /alert %}}

W&B の各 run には、[*run ID* と呼ばれる一意の識別子があります]({{< relref path="#unique-run-identifiers" lang="ja" >}})。[一意の ID を指定する]({{< relref path="#unique-run-identifiers" lang="ja" >}}) か、[W&B に ID をランダムに生成させる]({{< relref path="#autogenerated-run-ids" lang="ja" >}}) ことができます。

各 run には、人間が読める [*run 名* としても知られる一意でない識別子もあります]({{< relref path="#name-your-run" lang="ja" >}})。run の名前を指定するか、W&B にランダムに生成させることができます。

たとえば、次のコードスニペットを考えてみましょう。

```python title="basic.py"
import wandb

run = wandb.init(entity="wandbee", project="awesome-project")
```
コードスニペット は、次の出力を生成します。

```bash
🚀 View run exalted-darkness-6 at: 
https://wandb.ai/nico/awesome-project/runs/pgbn9y21
Find logs at: wandb/run-20241106_090747-pgbn9y21/logs
```

上記の コード が id パラメータ の 引数 を指定しなかったため、W&B は一意の run ID を作成します。`nico` は run をログした エンティティ 、`awesome-project` は run がログされる プロジェクト の名前、`exalted-darkness-6` は run の名前、`pgbn9y21` は run ID です。

{{% alert title="Notebook users" %}}
run の最後に `run.finish()` を指定して、run が完了したことを示します。これにより、run が プロジェクト に適切にログされ、バックグラウンド で継続されないようになります。

```python title="notebook.ipynb"
import wandb

run = wandb.init(entity="<entity>", project="<project>")
# Training code, logging, and so forth
run.finish()
```
{{% /alert %}}

各 run には、run の現在のステータス を記述する 状態 があります。可能な run の 状態 の完全なリストについては、[Run の 状態]({{< relref path="#run-states" lang="ja" >}}) を参照してください。

## Run の 状態
次のテーブルは、run がとりうる 状態 を記述しています。

| 状態 | 説明 |
| ----- | ----- |
| Finished| run が終了し、 完全に データ が同期されたか、`wandb.finish()` が呼び出されました |
| Failed | run が 0 以外の終了ステータス で終了しました |
| Crashed | run が 内部 プロセス で ハートビート の送信を停止しました。これは、 マシン が クラッシュ した場合に発生する可能性があります |
| Running | run はまだ実行中で、最近 ハートビート を送信しました |

## 一意の run 識別子

Run ID は、run の一意の識別子です。デフォルトでは、新しい run を初期化すると、[W&B が ランダム で一意の run ID を生成します]({{< relref path="#autogenerated-run-ids" lang="ja" >}})。run を初期化するときに、[独自の 一意の run ID を指定する]({{< relref path="#custom-run-ids" lang="ja" >}}) こともできます。

### 自動生成された run ID

run を初期化するときに run ID を指定しない場合、W&B は ランダム な run ID を生成します。run の一意の ID は、W&B App UI で確認できます。

1. [https://wandb.ai/home](https://wandb.ai/home) の W&B App UI に移動します。
2. run の初期化時に指定した W&B プロジェクト に移動します。
3. プロジェクト の ワークスペース 内で、[**Runs**] タブ を選択します。
4. [**Overview**] タブ を選択します。

W&B は、[**Run パス**] フィールド に一意の run ID を表示します。run パス は、 チーム の名前、 プロジェクト の名前、run ID で構成されます。一意の ID は、run パス の最後の部分です。

たとえば、次の図では、一意の run ID は `9mxi1arc` です。

{{< img src="/images/runs/unique-run-id.png" alt="" >}}

### カスタム run ID
[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) メソッド に `id` パラメータ を渡すことで、独自の run ID を指定できます。

```python 
import wandb

run = wandb.init(entity="<project>", project="<project>", id="<run-id>")
```

run の一意の ID を使用して、W&B App UI で run の Overview ページ に直接移動できます。次のセルは、特定の run の URL パス を示しています。

```text title="W&B App URL for a specific run"
https://wandb.ai/<entity>/<project>/<run-id>
```

山かっこ (`< >`) で囲まれた値は、エンティティ 、 プロジェクト 、run ID の実際の値の プレースホルダー です。

## run に名前を付ける
run の名前は、人間が読める一意でない識別子です。

デフォルトでは、W&B は新しい run を初期化するときに ランダム な run 名を生成します。run の名前は、 プロジェクト の ワークスペース 内と、[run の Overview ページ]({{< relref path="#overview-tab" lang="ja" >}}) の上部に表示されます。

{{% alert %}}
run 名は、 プロジェクト ワークスペース で run をすばやく識別する方法として使用します。
{{% /alert %}}

[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) メソッド に `name` パラメータ を渡すことで、run の名前を指定できます。

```python 
import wandb

run = wandb.init(entity="<project>", project="<project>", name="<run-name>")
```

## run にメモを追加する
特定の run に追加するメモは、[**Overview**] タブ の run ページ と、 プロジェクト ページ の run のテーブルに表示されます。

1. W&B プロジェクト に移動します
2. プロジェクト サイドバー から [**Workspace**] タブ を選択します
3. run セレクター からメモを追加する run を選択します
4. [**Overview**] タブ を選択します
5. [**Description**] フィールド の横にある 鉛筆 アイコン を選択し、メモを追加します

## run を停止する
W&B App または プログラム で run を停止します。

{{< tabpane text=true >}}
  {{% tab header="Programmatically" %}}
1. run を初期化した ターミナル または コード エディタ に移動します。
2. `Ctrl+D` を押して run を停止します。

たとえば、上記の手順に従うと、 ターミナル は次のようになります。

```bash
KeyboardInterrupt
wandb: 🚀 View run legendary-meadow-2 at: https://wandb.ai/nico/history-blaster-4/runs/o8sdbztv
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20241106_095857-o8sdbztv/logs
```

W&B App UI に移動して、run が アクティブ でなくなったことを確認します。

1. run がログされている プロジェクト に移動します。
2. run の名前を選択します。
  {{% alert %}}
  停止する run の名前は、 ターミナル または コード エディタ の出力から確認できます。たとえば、上記の例では、run の名前は `legendary-meadow-2` です。
  {{% /alert %}}
3. プロジェクト サイドバー から [**Overview**] タブ を選択します。

[**State**] フィールド の横で、run の 状態 が `running` から `Killed` に変わります。

{{< img src="/images/runs/stop-run-terminal.png" alt="" >}}
  {{% /tab %}}
  {{% tab header="W&B App" %}}

1. run がログされている プロジェクト に移動します。
2. run セレクター 内で停止する run を選択します。
3. プロジェクト サイドバー から [**Overview**] タブ を選択します。
4. [**State**] フィールド の横にある上部の ボタン を選択します。
{{< img src="/images/runs/stop-run-manual.png" alt="" >}}

[**State**] フィールド の横で、run の 状態 が `running` から `Killed` に変わります。

{{< img src="/images/runs/stop-run-manual-status.png" alt="" >}}
  {{% /tab %}}
{{< /tabpane >}}

可能な run の 状態 の完全なリストについては、[State フィールド]({{< relref path="#run-states" lang="ja" >}}) を参照してください。

## ログに記録された run を表示する

run の 状態、run にログされた Artifacts、run 中に記録された ログ ファイル など、特定の run に関する情報を表示します。

{{< img src="/images/runs/demo-project.gif" alt="" >}}

特定の run を表示するには:

1. [https://wandb.ai/home](https://wandb.ai/home) の W&B App UI に移動します。
2. run の初期化時に指定した W&B プロジェクト に移動します。
3. プロジェクト サイドバー 内で、[**Workspace**] タブ を選択します。
4. run セレクター 内で、表示する run をクリックするか、run 名の一部を入力して、一致する run を フィルター します。

    デフォルトでは、長い run 名は読みやすくするために中央で切り捨てられます。代わりに、run 名を先頭または末尾で切り捨てるには、run のリストの上部にある アクション `...` メニュー をクリックし、[**Run 名のトリミング**] を設定して、末尾、中央、または先頭をトリミングします。

特定の run の URL パス には、次の形式があることに注意してください。

```text
https://wandb.ai/<team-name>/<project-name>/runs/<run-id>
```

山かっこ (`< >`) で囲まれた値は、 チーム 名、 プロジェクト 名、run ID の実際の値の プレースホルダー です。

### Overviewタブ
[**Overview**] タブ を使用して、 プロジェクト 内の特定の run 情報について学習します。次に例を示します。

* **Author**: run を作成する W&B エンティティ 。
* **Command**: run を初期化する コマンド 。
* **Description**: 提供した run の説明。run の作成時に説明を指定しない場合、このフィールド は空です。W&B App UI を使用するか、Python SDK で プログラム で説明を run に追加できます。
* **Duration**: run が アクティブ に計算または データ をログしている時間。一時停止または待機は除きます。
* **Git リポジトリ**: run に関連付けられている git リポジトリ。[git を有効にする]({{< relref path="/guides/models/app/settings-page/user-settings.md#personal-github-integration" lang="ja" >}}) して、このフィールド を表示する必要があります。
* **Host name**: W&B が run を計算する場所。 マシン で ローカル に run を初期化する場合は、 マシン の名前が表示されます。
* **Name**: run の名前。
* **OS**: run を初期化する オペレーティング システム 。
* **Python 実行可能ファイル**: run を開始する コマンド 。
* **Python バージョン**: run を作成する Python バージョン を指定します。
* **Run パス**: `entity/project/run-ID` の形式で一意の run 識別子を識別します。
* **Runtime**: run の開始から終了までの合計時間を測定します。これは、run の ウォール クロック 時間です。Runtime には、run が一時停止している時間または リソース を待機している時間が含まれますが、Duration は含まれません。
* **Start time**: run を初期化する タイムスタンプ 。
* **State**: [run の 状態]({{< relref path="#run-states" lang="ja" >}})。
* **System hardware**: W&B が run の計算に使用する ハードウェア 。
* **Tags**: 文字列のリスト。タグ は、関連する run をまとめて編成したり、`baseline` や `production` などの一時的なラベル を適用したりするのに役立ちます。
* **W&B CLI バージョン**: run コマンド を ホスト した マシン にインストールされている W&B CLI バージョン 。

W&B は、概要セクション の下に次の情報を保存します。

* **Artifact Outputs**: run によって生成された Artifacts 出力。
* **Config**: [`wandb.config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) で保存された config パラメータ のリスト。
* **Summary**: [`wandb.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) で保存された summary パラメータ のリスト。デフォルトでは、W&B はこの値を最後にログされた値に設定します。

{{< img src="/images/app_ui/wandb_run_overview_page.png" alt="W&B Dashboard run overview tab" >}}

プロジェクト の概要の例は[こちら](https://wandb.ai/stacey/deep-drive/overview)をご覧ください。

### Workspaceタブ
[Workspace] タブ を使用して、自動生成された カスタム プロット 、 システム メトリクス など、 可視化 を表示、検索、 グループ化 、および配置します。

{{< img src="/images/app_ui/wandb-run-page-workspace-tab.png" alt="" >}}

プロジェクト ワークスペース の例は[こちら](https://wandb.ai/stacey/deep-drive/workspace?nw=nwuserstacey)をご覧ください

### Runsタブ

[Runs] タブ を使用して、run を フィルター 、 グループ化 、および並べ替えます。

{{< img src="/images/runs/run-table-example.png" alt="" >}}

次のタブ は、[Runs] タブ で実行できる一般的な アクション の一部を示しています。

{{< tabpane text=true >}}
   {{% tab header="Customize columns" %}}
[Runs] タブ には、 プロジェクト 内の run に関する詳細が表示されます。デフォルトでは、多数の 列 が表示されます。

- 表示されているすべての 列 を表示するには、 ページ を水平方向に スクロール します。
- 列 の順序を変更するには、 列 を左または右に ドラッグ します。
- 列 を ピン留め するには、 列 名の上に カーソル を置き、表示される アクション メニュー `...` をクリックし、[**Pin column**] をクリックします。ピン留め された 列 は、[**Name**] 列 の後、 ページ の左側の近くに表示されます。ピン留め された 列 の ピン留め を解除するには、[**Unpin column**] を選択します
- 列 を非表示にするには、 列 名の上に カーソル を置き、表示される アクション メニュー `...` をクリックし、[**Hide column**] をクリックします。現在非表示になっているすべての 列 を表示するには、[**Columns**] をクリックします。
- 複数の 列 を一度に表示、非表示、 ピン留め 、および ピン留め 解除するには、[**Columns**] をクリックします。
  - 非表示の 列 の名前をクリックして、非表示を解除します。
  - 表示されている 列 の名前をクリックして、非表示にします。
  - 表示されている 列 の横にある ピン アイコン をクリックして ピン留め します。

[Runs] タブ を カスタマイズ すると、 カスタマイズ は[Workspace タブ]({{< relref path="#workspace-tab" lang="ja" >}}) の [**Runs**] セレクター にも反映されます。

   {{% /tab %}}

   {{% tab header="Sort" %}}
指定された 列 の値で テーブル 内のすべての行を並べ替えます。

1. マウス を 列 タイトル の上に移動します。ケバブ メニュー (3 つの垂直 ドット) が表示されます。
2. ケバブ メニュー (3 つの垂直 ドット) を選択します。
3. [**Sort Asc**] または [**Sort Desc**] を選択して、行をそれぞれ 昇順 または 降順 に並べ替えます。

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="See the digits for which the model most confidently guessed '0'." >}}

上の図は、`val_acc` という名前の テーブル 列 の並べ替え オプション を表示する方法を示しています。
   {{% /tab %}}
   {{% tab header="Filter" %}}
ダッシュボード の上にある [**Filter**] ボタン を使用して、 式 で すべての行を フィルター します。

{{< img src="/images/data_vis/filter.png" alt="See only examples which the model gets wrong." >}}

[**Add filter**] を選択して、1 つまたは複数の フィルター を行に追加します。3 つの ドロップダウン メニュー が表示されます。左から右への フィルター タイプ は、 列 名、 オペレーター 、および値に基づいています

|                   | 列 名 | 二項関係    | 値       |
| -----------       | ----------- | ----------- | ----------- |
| 受け入れられる値   | 文字列       |  &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | 整数, float, 文字列, タイムスタンプ , null |

式 エディター には、 列 名のオートコンプリート と論理述語構造を使用して、各 項 の オプション のリストが表示されます。「and」または「or」(および場合によっては 括弧 ) を使用して、複数の論理述語を 1 つの 式 に接続できます。

{{< img src="/images/data_vis/filter_example.png" alt="" >}}
上の図は、`val_loss` 列 に基づく フィルター を示しています。この フィルター は、 検証 損失 が 1 以下 の run を表示します。
   {{% /tab %}}
   {{% tab header="Group" %}}
ダッシュボード の上にある [**Group by**] ボタン を使用して、特定の 列 の値で行を グループ化 します。

{{< img src="/images/data_vis/group.png" alt="The truth distribution shows small errors: 8s and 2s are confused for 7s and 9s for 2s." >}}

デフォルトでは、これにより、他の数値 列 が、その グループ 全体の 列 の値の分布を示す ヒストグラム に変わります。グループ化 は、 データ のより高レベルの パターン を理解するのに役立ちます。
   {{% /tab %}}
{{< /tabpane >}}

### Systemタブ
[**System タブ**] には、CPU 使用率、 システム メモリ 、 ディスク I/O、 ネットワーク トラフィック、GPU 使用率など、特定の run に対して追跡される システム メトリクス が表示されます。

W&B が追跡する システム メトリクス の完全なリストについては、[System メトリクス]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}}) を参照してください。

{{< img src="/images/app_ui/wandb_system_utilization.png" alt="" >}}

システム タブ の例は[こちら](https://wandb.ai/stacey/deep-drive/runs/ki2biuqy/system?workspace=user-carey)をご覧ください。

### Logsタブ
[**Log タブ**] には、 コマンドライン に出力された出力 (標準出力 (`stdout`) や 標準 エラー (`stderr`) など) が表示されます。

右上隅にある [**Download**] ボタン を選択して、 ログ ファイル をダウンロードします。

{{< img src="/images/app_ui/wandb_run_page_log_tab.png" alt="" >}}

ログ タブ の例は[こちら](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/logs)をご覧ください。

### Filesタブ
[**Files タブ**] を使用して、モデル チェックポイント 、 検証 セット の例など、特定の run に関連付けられた ファイル を表示します

{{< img src="/images/app_ui/wandb_run_page_files_tab.png" alt="" >}}

ファイル タブ の例は[こちら](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/files/media/images)をご覧ください。

### Artifactsタブ
[**Artifacts**] タブ には、指定された run の 入力 および 出力 [アーティファクト]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) が一覧表示されます。

{{< img src="/images/app_ui/artifacts_tab.png" alt="" >}}

アーティファクト タブ の例は[こちら](https://wandb.ai/stacey/artifact_july_demo/runs/2cslp2rt/artifacts)をご覧ください。

## run を削除する

W&B App を使用して、 プロジェクト から 1 つまたは複数の run を削除します。

1. 削除する run が含まれている プロジェクト に移動します。
2. プロジェクト サイドバー から [**Runs**] タブ を選択します。
3. 削除する run の横にある チェックボックス をオンにします。
4. テーブル の上にある [**Delete**] ボタン ( ゴミ箱 アイコン ) を選択します。
5. 表示される モーダル から、[**Delete**] を選択します。

{{% alert %}}
特定の ID を持つ run が削除されると、その ID を再度使用できなくなる場合があります。以前に削除された ID で run を開始しようとすると、 エラー が表示され、開始が防止されます。
{{% /alert %}}

{{% alert %}}
多数の run を含む プロジェクト の場合、検索バー を使用して 正規表現 を使用して削除する run を フィルター するか、 フィルター ボタン を使用して、ステータス 、 タグ 、またはその他のプロパティ に基づいて run を フィルター できます。
{{% /alert %}}

## run を整理する

このセクション では、 グループ と ジョブタイプ を使用して run を整理する方法について説明します。run を グループ (たとえば、 実験 名) に割り当て、 ジョブタイプ (たとえば、 前処理 、 トレーニング 、 評価 、 デバッグ ) を指定することで、 ワークフロー を効率化し、モデル の比較を改善できます。

### run に グループ または ジョブタイプ を割り当てる

W&B の各 run は、[**グループ**] と [**ジョブタイプ**] で 分類 できます。

- **グループ**: 実験 の広範な カテゴリ で、run の整理と フィルター に使用されます。
- **ジョブタイプ**: `preprocessing`、`training`、`evaluation` など、run の 機能 。

次の[ワークスペース の例](https://wandb.ai/stacey/model_iterz?workspace=user-stacey) では、Fashion-MNIST データセット から 増え続ける量の データ を使用して ベースライン モデル を トレーニング します。ワークスペース では、使用される データ 量を色で表します。

- **黄色から濃い緑**は、 ベースライン モデル の データ 量が増加していることを示します。
- **水色からバイオレット、マゼンタ**は、追加の パラメータ を持つ、より複雑な「double」モデルの データ 量を示します。

W&B の フィルター オプション と検索バーを使用して、特定の条件に基づいて run を比較します。次に例を示します。
- 同じ データセット での トレーニング 。
- 同じ テストセット での 評価 。

フィルター を適用すると、[**Table**] ビュー が自動的に更新されます。これにより、モデル 間の パフォーマンス の違いを特定できます。たとえば、一方のモデル で他方のモデル よりも大幅に困難な クラス を特定できます。
```