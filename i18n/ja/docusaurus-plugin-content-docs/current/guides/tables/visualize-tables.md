---
description: Visualize and analyze W&B Tables.
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# テーブルの視覚化と分析

W&Bテーブルを用いて、データやモデルの予測をログし、可視化します。データを対話的に探索して：

* モデル、エポック、個別の例において、変更内容を正確に比較
* データの高レベルなパターンを理解
* 視覚的なサンプルで洞察をキャプチャして伝える

機械学習モデルのパフォーマンスやデータ解析に関する質問に答えるために、W&Bテーブルをカスタマイズできます。

:::info
W&Bテーブルは以下の振る舞いを持っています:
1. **アーティファクトのコンテキストではステートレス**: アーティファクトのバージョンと共にログされたテーブルは、ブラウザのウィンドウを閉じた後デフォルトの状態に戻ります
2. **ワークスペースやレポートのコンテキストではステートフル**: 単一のrunワークスペース、複数のrunプロジェクトワークスペース、あるいはレポートでテーブルに加えた変更は維持されます。

現在のW&Bテーブルビューを保存する方法については、[ビューを保存する](#save-your-view)を参照してください。
:::

## テーブル操作

W&Bアプリを使って、W&Bテーブルをソート、フィルタ、グループ化します。

<!-- [自分で試してみる →](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json) -->
### ソート

指定された列の値でテーブル内のすべての行をソートします。
1. 列のタイトル上にマウスをホバーさせると、ケバブメニューが表示されます（3つの垂直ドット）。
2. ケバブメニュー（3つの垂直ドット）を選択します。
3. 行を昇順または降順で並べ替えるには、それぞれ**Sort Asc**または**Sort Desc**を選択してください。

![モデルが最も自信を持って「0」と判断した数字を確認してください。](/images/data_vis/data_vis_sort_kebob.png)

上記の画像は、`val_acc`という名前のテーブル列のソートオプションを表示する方法を示しています。

### フィルタ

ダッシュボードの左上にある**Filter**ボタンで、表現式によってすべての行をフィルタリングします。

![モデルが間違っている例のみ表示します。](/images/data_vis/filter.png)

**Add filter**を選択して、1つ以上のフィルタを行に追加します。3つのドロップダウンメニューが表示されます。左から右に、フィルタのタイプは次のとおりです：列名、演算子、値

|                   | 列名         | 二項関係          | 値          |
| -----------       | ----------- | ----------- | ----------- |
| 受け入れ可能な値   | 文字列       |  &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | 整数, 小数, 文字列, タイムスタンプ, null |

式エディタは、列名と論理述語構造のオートコンプリートを使用して、各用語の選択肢一覧を表示します。"and" または "or"（そして時々括弧）を使って複数の論理述語を1つの式に結合できます。

![](/images/data_vis/filter_example.png)
前述の画像は、`val_loss`列に基づくフィルタを示しています。このフィルタは、検証ロスが1以下であるW&BのRunを表示します。
### グループ

特定の列の値によってすべての行をグループ化するには、列ヘッダーの**グループ化**ボタンを使用します。

![真実の分布は、小さな誤差が表示されます。8 と 2 はそれぞれ 7 と 2 のように混同されます。](/images/data_vis/group.png)

デフォルトでは、これにより、他の数値列がグループ全体でその列の値の分布を示すヒストグラムに変わります。グループ化は、データの高次元のパターンを理解するのに役立ちます。

<!-- ## 列の変更
以下のセクションでは、W&Bテーブルの変更方法を示しています。 -->

<!-- ### 列の追加

新しい列をテーブルの左右に挿入することができます。列を追加するには、ケバブ・メニューを選択します。

任意の列のケバブ・メニューから、左右に新しい列を挿入できます。セル式を編集して、既存の列への参照、数学的および論理演算子、行がグループ化された場合の集約関数（平均、合計、最小／最大など）を使用して新しい列を計算します。必要に応じて、式エディタの下にある新しい名前を列に付けます。

![closed\_loop\_score 列は、典型的なループ（0,6,8,9）を持つ数字の信頼スコアを合計します。](/images/data_vis/add_columns.png) -->

<!-- ### 列および表示設定の編集

テーブルは、その列に記録された値のタイプに基づいて、列データを表示します。列名または "列設定"を三点メニューからクリックすることで、次の事ができます。

* **列の内容**を編集する方法：セル式を編集して、別の項目を選択するか、上記で説明したような論理述語式を構築します。また、count() や avg() などの関数を追加して内容に適用します。
* **列のタイプ**：ヒストグラム、値の配列、数値、テキストなどを変換します。W&Bは、データの内容に基づいてタイプを推測します。
* **ページネーション**：グループ化された行で一度に表示するオブジェクトの数を選択します。
* **列ヘッダーの表示名**

### 列の削除 -->
<!-- "Remove"を選択して列を削除します。 -->

## テーブルの比較

上記で説明したすべての操作は、テーブル比較のコンテキストでも機能します。

![左: トレーニングエポック1回後の誤差、右: エポック5回後の誤差](/images/data_vis/table_comparison.png)

### UIから

二つのテーブルを比較するには、アーティファクトと一緒に記録されたテーブルのうち一つを表示してから始めてください。以下の画像では、5回のエポックごとにMNIST検証データでモデルの予測を示しています（[インタラクティブな例 →](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)）。

![「predictions」をクリックしてテーブルを表示](@site/static/images/data_vis/preds_mnist.png)

次に、比較する別のアーティファクトバージョンを選択してください。例えば、5回のトレーニングエポック後に同じモデルによって行われたMNISTの予測結果を比較するために「v4」を選択します。サイドバーで2番目のアーティファクトバージョンにマウスオーバーし、「Compare」をクリックして表示。

![トレーニングエポック1回後（v0、ここに示す）と5回後（v4）のモデル予測を比較するための準備](@site/static/images/data_vis/preds_2.png)

#### マージされたビュー

[ライブ例 →](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)

最初に、両方のテーブルが一緒にマージされた状態が表示されます。選択した最初のテーブルはインデックス0で青いハイライトがあり、2番目のテーブルはインデックス1で黄色いハイライトがあります。

![マージされたビューでは、数値の列はデフォルトでヒストグラムとして表示されます](@site/static/images/data_vis/merged_view.png)

マージされたビューから：

* **結合キーを選択**:  左上のドロップダウンを使用して、二つのテーブルの結合キーとして使用する列を設定します。通常、これはデータセット内の特定の例のファイル名や生成されたサンプルのインクリメントインデックスなど、各行の一意の識別子になります。現在は、どの列でも選択できますが、読み取りづらいテーブルや遅いクエリが発生する場合があります。
* **結合ではなく連結**: このドロップダウンで「すべてのテーブルを連結」を選択することで、二つのテーブルの行を_すべて結合（ユニオン）_し、より大きなテーブルにするのではなく、列をまたいで結合します。
* **それぞれのテーブルを明示的に参照**: フィルタ式で 0、1、および \* を使用して、1つまたは複数のテーブルインスタンス内の列を明示的に指定します
* **ヒストグラムとして詳細な数値の違いを視覚化**: 一目でセル内の値を比較することができます
#### サイドバイサイド表示

2つのテーブルをサイドバイサイドで表示するには、最初のドロップダウンを "Merge Tables: Table" から "List of: Table" に変更し、"Page size" をそれぞれ更新します。最初に選択されたテーブルは左に、2番目のテーブルは右に表示されます。また、"Vertical" チェックボックスをクリックして、これらのテーブルを縦方向にも比較できます。

![サイドバイサイド表示では、テーブルの行が互いに独立しています。](/images/data_vis/side_by_side.png)

* **テーブルを一目で比較する**：両方のテーブルに（並べ替え、フィルター、グループ）を同時に適用して、変更や違いをすぐに見つける。例えば、誤った予測を推測ごとにグループ化して表示する、全体的に最も困難な否定を見る、正しいラベルごとの信頼スコア分布を見るなど。
* **2つのテーブルを独立して探索する**：興味のある側面/行にスクロールしてフォーカスする

### 時間をまたいで比較する

モデルのパフォーマンスをトレーニング時間に関して分析するには、トレーニングの意味のあるステップごとにアーティファクトコンテキストでテーブルをログに記録します：エポックのトレーニングごとに50回、または開発フローに適切な頻度での検証ステップの終了時など。サイドバイサイド表示を使用して、モデル予測の変化を視覚化します。

![各ラベルで、モデルは 5 回のトレーニング後のエポック（右）では 1 回（左）よりも間違えが少なくなります。](/images/data_vis/compare_across_time.png)

トレーニング時間をまたいで予測を視覚化するより詳細なウォークスルーについては、[こちらのレポート](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)や、このインタラクティブな[ノートブック例 →](http://wandb.me/tables-walkthrough)をご覧ください。

### モデルバリアント間の比較

構成（ハイパーパラメーター、基本アーキテクチャーなど）が異なるモデル間の性能を分析するには、同じステップでログされた 2 つの異なるモデルの 2 つのアーティファクトバージョンを比較します。たとえば、`baseline` と新しいモデルバリアント `2x_layers_2x_lr` を比較し、最初の畳み込み層が 32 から 64 に倍増し、2 番目が 128 から 256 に倍増し、学習率が 0.001 から 0.002 に倍増することを比較します。 [このライブの例](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x\_layers\_2x\_lr) から、サイドバイサイド表示を使用し、1回のトレーニングエポック後（左タブ）と 5回のトレーニングエポック後（右タブ）の誤った予測に絞り込みます。

これはモデルの比較のおもちゃの例ですが、テーブルでできる探索分析の簡単さ、柔軟性、深さを示しています。コードを再実行することなく、新しい一度きりのスクリプトを作成したり、新しいチャートを生成したりすることなく。

<Tabs
  defaultValue="one_epoch"
  values={[
    {label: '1回のトレーニングエポック', value: 'one_epoch'},
    {label: '5回のトレーニングエポック', value: 'five_epochs'},
  ]}>
  <TabItem value="one_epoch">
![1エポック後、パフォーマンスはまちまちで、一部のクラスの精度は向上し、他のクラスは低下します。](/images/data_vis/compare_across_variants.png)

  </TabItem>

  <TabItem value="five_epochs">

![5エポック後、"double"バリアントはベースラインに追いつこうとしています。](/images/data_vis/compare_across_variants_after_5_epochs.png)

  </TabItem>

</Tabs>



## あなたのビューを保存する



runワークスペース、プロジェクトワークスペース、またはレポートでインタラクションするテーブルは、自動的にそのビューの状態を保存します。テーブル操作を適用してブラウザを閉じると、次にテーブルにアクセスするときに最後に表示された設定が保持されます。



アーティファクトコンテキストでインタラクションするテーブルは、状態を持たないままです。



特定の状態のワークスペースからテーブルを保存するには、レポートにエクスポートしてください。これは、ワークスペースの可視化パネルの右上隅にある3つのドットメニューから行えます(3つのドット → "Share panel" または "Add to report")。



![Share panelは新しいレポートを作成し、Add to reportは既存のレポートに追加することができます。](/images/data_vis/share_your_view.png)





### 事例集



以下のレポートは、W&Bテーブルの様々なユースケースを示しています。



* [時間経過とともに予測を視覚化する](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)

* [ワークスペースのテーブルを比較する方法](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)

* [画像 & 分類モデル](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)

* [テキスト & 生成言語モデル](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)

* [固有表現認識](https://wandb.ai/stacey/ner\_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)

* [AlphaFoldプロテイン](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)