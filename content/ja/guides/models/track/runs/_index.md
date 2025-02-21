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

W&B でログされる単一の計算単位が *run* です。W&B の run は、プロジェクト全体を構成する原子要素と考えることができます。言い換えれば、各 run は、モデルのトレーニングと結果のログ、ハイパーパラメータースイープなど、特定の計算の記録です。

run を開始する一般的なパターンには、以下が含まれますが、これらに限定されません。

* モデルのトレーニング
* ハイパーパラメーターを変更して新しい実験を行う
* 異なるモデルで新しい 機械学習 の実験を行う
* [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) としてデータまたはモデルをログに記録する
* [W&B Artifacts のダウンロード]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ja" >}})

W&B は、作成した run を [*Projects*]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) に保存します。W&B App UI で、run とそのプロパティを run の project ワークスペース内で表示できます。また、[`wandb.Api.Run`]({{< relref path="/ref/python/public-api/run.md" lang="ja" >}}) オブジェクトを使用して、プログラムで run のプロパティにアクセスすることもできます。

`run.log` でログに記録する内容はすべて、その run に記録されます。次のコードスニペットを検討してください。

```python
import wandb

run = wandb.init(entity="nico", project="awesome-project")
run.log({"accuracy": 0.9, "loss": 0.1})
```

最初の行は、W&B Python SDK をインポートします。2 行目は、エンティティ `nico` の下の project `awesome-project` で run を初期化します。3 行目は、モデルの精度と損失をその run に記録します。

ターミナル内では、W&B は以下を返します。

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

ターミナルで W&B が返す URL は、W&B App UI の run のワークスペースにリダイレクトします。ワークスペースで生成されたパネルは、単一のポイントに対応していることに注意してください。

{{< img src="/images/runs/single-run-call.png" alt="" >}}

単一の時点での メトリクス のログ記録は、それほど役に立たない場合があります。判別モデルのトレーニングの場合のより現実的な例は、定期的に メトリクス をログに記録することです。たとえば、次のコードスニペットを検討してください。

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

トレーニングスクリプトは `run.log` を 10 回呼び出します。スクリプトが `run.log` を呼び出すたびに、W&B はそのエポックの精度と損失をログに記録します。W&B が前の出力から出力する URL を選択すると、W&B App UI の run のワークスペースに移動します。

W&B は、`jolly-haze-4` という単一の run 内でシミュレートされたトレーニングループをキャプチャすることに注意してください。これは、スクリプトが `wandb.init` メソッドを 1 回だけ呼び出すためです。

{{< img src="/images/runs/run_log_example_2.png" alt="" >}}

別の例として、[スイープ]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) 中に、W&B は指定したハイパーパラメーター探索空間を探索します。W&B は、スイープが作成する新しいハイパーパラメーターの組み合わせを、一意の run として実装します。

## run を初期化する

[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) で W&B run を初期化します。次のコードスニペットは、W&B Python SDK をインポートして run を初期化する方法を示しています。

山かっこ (`< >`) で囲まれた値は、独自の値に置き換えてください。

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
```

run を初期化すると、W&B は project フィールドに指定した project (`wandb.init(project="<project>"` ) に run を記録します。W&B は、 project がまだ存在しない場合は、新しい project を作成します。project がすでに存在する場合、W&B はその project に run を保存します。

{{% alert %}}
project 名を指定しない場合、W&B は run を `Uncategorized` という project に保存します。
{{% /alert %}}

W&B の各 run には、[*run ID* として知られる一意の識別子]({{< relref path="#unique-run-identifiers" lang="ja" >}}) があります。[一意の ID を指定]({{< relref path="#unique-run-identifiers" lang="ja" >}}) することも、[W&B に ID をランダムに生成させる]({{< relref path="#autogenerated-run-ids" lang="ja" >}}) こともできます。

各 run には、人間が判読できる [ *run 名* として知られる一意でない識別子]({{< relref path="#name-your-run" lang="ja" >}}) もあります。run の名前を指定することも、W&B に名前をランダムに生成させることもできます。

たとえば、次のコードスニペットを検討してください。

```python title="basic.py"
import wandb

run = wandb.init(entity="wandbee", project="awesome-project")
```
コードスニペットは次の出力を生成します。

```bash
🚀 View run exalted-darkness-6 at: 
https://wandb.ai/nico/awesome-project/runs/pgbn9y21
Find logs at: wandb/run-20241106_090747-pgbn9y21/logs
```

上記のコードでは id パラメータの引数が指定されていないため、W&B は一意の run ID を作成します。`nico` は run をログに記録した エンティティ 、`awesome-project` は run がログに記録される project の名前、`exalted-darkness-6` は run の名前、`pgbn9y21` は run ID です。

{{% alert title="ノートブック ユーザー" %}}
run の最後に `run.finish()` を指定して、run が完了したことをマークします。これにより、run が project に適切にログに記録され、バックグラウンドで続行されないようになります。

```python title="notebook.ipynb"
import wandb

run = wandb.init(entity="<entity>", project="<project>")
# Training code, logging, and so forth
run.finish()
```
{{% /alert %}}

各 run には、run の現在のステータスを示す状態があります。可能な run 状態の完全なリストについては、[Run 状態]({{< relref path="#run-states" lang="ja" >}}) を参照してください。

## Run の状態
次の表は、run がとりうる状態について説明しています。

| 状態 | 説明 |
| ----- | ----- |
| 完了 | run が終了し、データが完全に同期されたか、`wandb.finish()` が呼び出されました |
| 失敗 | run がゼロ以外の終了ステータスで終了しました |
| クラッシュ | run が内部プロセスでハートビートの送信を停止しました。これは、マシンがクラッシュした場合に発生する可能性があります |
| 実行中 | run はまだ実行されており、最近ハートビートを送信しました |

## 一意の run 識別子

Run ID は、run の一意の識別子です。デフォルトでは、新しい run を初期化すると、[W&B がランダムで一意の run ID を生成します]({{< relref path="#autogenerated-run-ids" lang="ja" >}})。run を初期化するときに、[自分自身の一意の run ID を指定]({{< relref path="#custom-run-ids" lang="ja" >}}) することもできます。

### 自動生成された run ID

run を初期化するときに run ID を指定しない場合、W&B はランダムな run ID を生成します。run の一意の ID は、W&B App UI で確認できます。

1. [https://wandb.ai/home](https://wandb.ai/home) で W&B App UI に移動します。
2. run を初期化したときに指定した W&B project に移動します。
3. project のワークスペース内で、[**Runs**] タブを選択します。
4. [**Overview**] タブを選択します。

W&B は、[**Run パス**] フィールドに一意の run ID を表示します。run パスは、チームの名前、project の名前、および run ID で構成されます。一意の ID は、run パスの最後の部分です。

たとえば、次の図では、一意の run ID は `9mxi1arc` です。

{{< img src="/images/runs/unique-run-id.png" alt="" >}}

### カスタム run ID
[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) メソッドに `id` パラメーターを渡すことで、独自の run ID を指定できます。

```python 
import wandb

run = wandb.init(entity="<project>", project="<project>", id="<run-id>")
```

run の一意の ID を使用して、W&B App UI で run の概要ページに直接移動できます。次のセルは、特定の run の URL パスを示しています。

```text title="特定の run の W&B App URL"
https://wandb.ai/<entity>/<project>/<run-id>
```

山かっこ (`< >`) で囲まれた値は、エンティティ 、project 、run ID の実際の値のプレースホルダーです。

## run に名前を付ける
run の名前は、人間が判読できる一意でない識別子です。

デフォルトでは、新しい run を初期化すると、W&B はランダムな run 名を生成します。run の名前は、project のワークスペース内と、[run の概要ページ]({{< relref path="#overview-tab" lang="ja" >}}) の上部に表示されます。

{{% alert %}}
run 名を使用して、project ワークスペース内の run をすばやく識別します。
{{% /alert %}}

[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) メソッドに `name` パラメーターを渡すことで、run の名前を指定できます。

```python 
import wandb

run = wandb.init(entity="<project>", project="<project>", name="<run-name>")
```

## run にメモを追加する
特定の run に追加したメモは、[**Overview**] タブの run ページと project ページの run のテーブルに表示されます。

1. W&B project に移動します
2. project サイドバーから [**Workspace**] タブを選択します
3. メモを追加する run を run セレクターから選択します
4. [**Overview**] タブを選択します
5. [**Description**] フィールドの横にある鉛筆アイコンを選択し、メモを追加します

## run を停止する
W&B App またはプログラムで run を停止します。

{{< tabpane text=true >}}
  {{% tab header="プログラムで" %}}
1. run を初期化したターミナルまたはコードエディターに移動します。
2. `Ctrl+D` を押して run を停止します。

たとえば、上記の手順に従うと、ターミナルは次のようになります。

```bash
KeyboardInterrupt
wandb: 🚀 View run legendary-meadow-2 at: https://wandb.ai/nico/history-blaster-4/runs/o8sdbztv
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20241106_095857-o8sdbztv/logs
```

W&B App UI に移動して、run がアクティブでなくなったことを確認します。

1. run がログに記録されていた project に移動します。
2. run の名前を選択します。
  {{% alert %}}
  ターミナルまたはコードエディターの出力から停止する run の名前を見つけることができます。たとえば、上記の例では、run の名前は `legendary-meadow-2` です。
  {{% /alert %}}
3. project サイドバーから [**Overview**] タブを選択します。

[**State**] フィールドの横にある run の状態が `running` から `Killed` に変わります。

{{< img src="/images/runs/stop-run-terminal.png" alt="" >}}  
  {{% /tab %}}
  {{% tab header="W&B App" %}}

1. run がログに記録されている project に移動します。
2. run セレクター内で停止する run を選択します。
3. project サイドバーから [**Overview**] タブを選択します。
4. [**State**] フィールドの横にある上部のボタンを選択します。
{{< img src="/images/runs/stop-run-manual.png" alt="" >}}

[**State**] フィールドの横にある run の状態が `running` から `Killed` に変わります。

{{< img src="/images/runs/stop-run-manual-status.png" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}

可能な run 状態の完全なリストについては、[状態フィールド]({{< relref path="#run-states" lang="ja" >}}) を参照してください。

## ログに記録された run を表示する

run の状態、run にログに記録された Artifacts 、run 中に記録されたログファイルなど、特定の run に関する情報を表示します。

{{< img src="/images/runs/demo-project.gif" alt="" >}}

特定の run を表示するには:

1. [https://wandb.ai/home](https://wandb.ai/home) で W&B App UI に移動します。
2. run を初期化したときに指定した W&B project に移動します。
3. project サイドバー内で、[**Workspace**] タブを選択します。
4. run セレクター内で、表示する run をクリックするか、run 名の一部を入力して一致する run をフィルター処理します。

    デフォルトでは、長い run 名は読みやすくするために中央で切り捨てられます。代わりに run 名を先頭または末尾で切り捨てるには、run のリストの上部にあるアクション `...` メニューをクリックし、[**Run 名のクロップ**] を設定して、末尾、中央、または先頭をクロップします。

特定の run の URL パスには、次の形式があることに注意してください。

```text
https://wandb.ai/<team-name>/<project-name>/runs/<run-id>
```

山かっこ (`< >`) で囲まれた値は、チーム名、project 名、および run ID の実際の値のプレースホルダーです。

### Overview タブ
[**Overview**] タブを使用して、 project 内の特定の run 情報について学習します。次に例を示します。

* **作成者**: run を作成する W&B エンティティ 。
* **コマンド**: run を初期化するコマンド。
* **説明**: 提供された run の説明。run の作成時に説明を指定しない場合、このフィールドは空です。W&B App UI を使用するか、Python SDK を使用してプログラムで run に説明を追加できます。
* **期間**: run がアクティブに計算またはデータをログに記録している時間。一時停止または待機は除きます。
* **Git リポジトリ**: run に関連付けられた git リポジトリ。[Git を有効にする]({{< relref path="/guides/models/app/settings-page/user-settings.md#personal-github-integration" lang="ja" >}}) して、このフィールドを表示する必要があります。
* **ホスト名**: W&B が run を計算する場所。マシンでローカルに run を初期化すると、W&B にマシンの名前が表示されます。
* **名前**: run の名前。
* **OS**: run を初期化するオペレーティングシステム。
* **Python 実行可能ファイル**: run を開始するコマンド。
* **Python バージョン**: run を作成する Python バージョンを指定します。
* **Run パス**: `entity/project/run-ID` の形式で一意の run 識別子を識別します。
* **ランタイム**: run の開始から終了までの合計時間を測定します。これは、run のウォールクロック時間です。ランタイムには、run が一時停止またはリソースを待機している時間が含まれますが、期間には含まれません。
* **開始時間**: run を初期化するタイムスタンプ。
* **状態**: [run の状態]({{< relref path="#run-states" lang="ja" >}})。
* **システムハードウェア**: W&B が run の計算に使用するハードウェア。
* **タグ**: 文字列のリスト。タグは、関連する run をまとめて整理したり、`ベースライン` や `本番環境` などの一時的なラベルを適用したりするのに役立ちます。
* **W&B CLI バージョン**: run コマンドをホストするマシンにインストールされている W&B CLI バージョン。

W&B は、概要セクションの下に次の情報を保存します。

* **Artifacts 出力**: run によって生成された Artifacts 出力。
* **Config**: [`wandb.config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) で保存された config パラメータのリスト。
* **Summary**: [`wandb.log()`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) で保存された summary パラメータのリスト。デフォルトでは、W&B はこの値を最後にログに記録された値に設定します。

{{< img src="/images/app_ui/wandb_run_overview_page.png" alt="W&B ダッシュボード run の概要タブ" >}}

project の概要の例は、[こちら](https://wandb.ai/stacey/deep-drive/overview) を参照してください。

### Workspace タブ
[Workspace] タブを使用して、自動生成されたプロットやカスタムプロット、システム メトリクス など、可視化を表示、検索、グループ化、および配置します。

{{< img src="/images/app_ui/wandb-run-page-workspace-tab.png" alt="" >}}

project ワークスペースの例は、[こちら](https://wandb.ai/stacey/deep-drive/workspace?nw=nwuserstacey) を参照してください。

### Runs タブ

[Runs] タブを使用して、run をフィルター処理、グループ化、および並べ替えます。

{{< img src="/images/runs/run-table-example.png" alt="" >}}

次のタブは、[Runs] タブで実行できる一般的なアクションの一部を示しています。

{{< tabpane text=true >}}
   {{% tab header="列をカスタマイズする" %}}
[Runs] タブには、 project 内の run に関する詳細が表示されます。デフォルトでは、多数の列が表示されます。

- 表示されているすべての列を表示するには、ページを水平方向にスクロールします。
- 列の順序を変更するには、列を左または右にドラッグします。
- 列を固定するには、列名の上にマウスカーソルを置き、表示されるアクションメニュー `...` をクリックして、[**列を固定**] をクリックします。固定された列は、[**名前**] 列の後に、ページの左側の近くに表示されます。固定された列を固定解除するには、[**列を固定解除**] を選択します
- 列を非表示にするには、列名の上にマウスカーソルを置き、表示されるアクションメニュー `...` をクリックして、[**列を非表示にする**] をクリックします。現在非表示になっているすべての列を表示するには、[**列**] をクリックします。
  - 非表示の列の名前をクリックして、表示します。
  - 表示されている列の名前をクリックして、非表示にします。
  - 表示されている列の横にあるピンアイコンをクリックして、固定します。

[Runs] タブをカスタマイズすると、カスタマイズは、[Workspace タブ]({{< relref path="#workspace-tab" lang="ja" >}}) の [**Runs**] セレクターにも反映されます。

   {{% /tab %}}

   {{% tab header="並べ替え" %}}
テーブル内のすべての行を、指定された列の値で並べ替えます。

1. マウスを列タイトルに合わせます。ケバブメニューが表示されます (3 つの垂直ドキュメント)。
2. ケバブメニュー (3 つの垂直ドット) で選択します。
3. [**昇順で並べ替え**] または [**降順で並べ替え**] を選択して、行をそれぞれ昇順または降順で並べ替えます。

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="'0' と最も自信を持って推測したモデルの数字を参照してください。" >}}

上の画像は、`val_acc` という名前のテーブル列の並べ替えオプションを表示する方法を示しています。
   {{% /tab %}}
   {{% tab header="フィルター" %}}
ダッシュボードの上にある [**フィルター**] ボタンを使用して、式で行をすべてフィルター処理します。

{{< img src="/images/data_vis/filter.png" alt="モデルが間違っている例のみを参照してください。" >}}

[**フィルターを追加**] を選択して、行に 1 つ以上のフィルターを追加します。3 つのドロップダウンメニューが表示されます。左から右に、フィルターのタイプは、列名、演算子、および値に基づいています。

|  | 列名 | 二項関係 | 値 |
| ----------- | ----------- | ----------- | ----------- |
| 受け入れられる値 | 文字列 | =, ≠, ≤, ≥, IN, NOT IN, | 整数、浮動小数点数、文字列、タイムスタンプ、null |

式エディターには、列名と論理述語構造のオートコンプリートを使用して、各項のオプションのリストが表示されます。「and」または「or」(および場合によっては括弧) を使用して、複数の論理述語を 1 つの式に接続できます。

{{< img src="/images/data_vis/filter_example.png" alt="" >}}
上の画像は、`val_loss` 列に基づいたフィルターを示しています。フィルターは、検証損失が 1 以下の run を表示します。
   {{% /tab %}}
   {{% tab header="グループ" %}}
ダッシュボードの上にある [**グループ化**] ボタンを使用して、特定の列の値で行をすべてグループ化します。

{{< img src="/images/data_vis/group.png" alt="真実の分布は小さなエラーを示しています: 8 と 2 は 7 と 9、2 は 2 と混同されています。" >}}

デフォルトでは、これにより、他の数値列が、グループ全体のその列の値の分布を示すヒストグラムに変わります。グループ化は、データの高レベルのパターンを理解するのに役立ちます。
   {{% /tab %}}
{{< /tabpane >}}

### System タブ
[**System タブ**] には、CPU 使用率、システムメモリ、ディスク I/O、ネットワークトラフィック、GPU 使用率など、特定の run について追跡されたシステム メトリクス が表示されます。

W&B が追跡するシステム メトリクス の完全なリストについては、[システム メトリクス]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}}) を参照してください。

{{< img src="/images/app_ui/wandb_system_utilization.png" alt="" >}}

[System タブ] の例は、[こちら](https://wandb.ai/stacey/deep-drive/runs/ki2biuqy/system?workspace=user-carey) を参照してください。

### Logs タブ
[**Log タブ**] には、コマンドラインに出力された出力 (標準出力 (`stdout`) や標準エラー (`stderr`) など) が表示されます。

ログファイルをダウンロードするには、右上隅にある [**ダウンロード**] ボタンを選択します。

{{< img src="/images/app_ui/wandb_run_page_log_tab.png" alt="" >}}

[Logs タブ] の例は、[こちら](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/logs) を参照してください。

### Files タブ
[**Files タブ**] を使用して、モデル チェックポイント、検証セットの例など、特定の run に関連付けられたファイルを表示します。

{{< img src="/images/app_ui/wandb_run_page_files_tab.png" alt="" >}}

[Files タブ] の例は、[こちら](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/files/media/images) を参照してください。

### Artifacts タブ
[**Artifacts**] タブには、指定された run の入力および出力 [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) が一覧表示されます。

{{< img src="/images/app_ui/artifacts_tab.png" alt="" >}}

[Artifacts タブ] の例は、[こちら](https://wandb.ai/stacey/artifact_july_demo/runs/2cslp2rt/artifacts) を参照してください。

## run を削除する

W&B App で project から 1 つ以上の run を削除します。

1. 削除する run が含まれている project に移動します。
2. project サイドバーから [**Runs**] タブを選択します。
3. 削除する run の横にあるチェックボックスをオンにします。
4. テーブルの上にある [**削除**] ボタン (ゴミ箱アイコン) を選択します。
5. 表示されるモーダルから [**削除**] を選択します。

{{% alert %}}
多数の run が含まれている project の場合、検索バーを使用して Regex を使用して削除する run をフィルター処理するか、フィルターボタンを使用してステータス、タグ、またはその他のプロパティに基づいて run をフィルター処理できます。
{{% /alert %}}
