---
title: データ点の集約
menu:
  default:
    identifier: ja-guides-models-app-features-panels-line-plot-sampling
    parent: line-plot
weight: 20
---

折れ線グラフ内でポイント集約を使うと、データ可視化の精度とパフォーマンスを向上できます。ポイント集約のモードには 2 種類あります: [フルフィデリティ]({{< relref path="#full-fidelity" lang="ja" >}}) と [ランダムサンプリング]({{< relref path="#random-sampling" lang="ja" >}})。W&B はデフォルトで full fidelity モードを使用します。

## Full fidelity

full fidelity モードでは、W&B はデータポイント数に基づいて x 軸を動的にバケットへ分割します。そのうえで、折れ線グラフのポイント集約を描画する際に、各バケット内の最小値・最大値・平均値を計算します。

full fidelity モードでポイント集約を使う主な利点は 3 つあります:

* 極端値やスパイクを保持: データ内の外れ値やスパイクを失わずに保持
* 最小・最大ポイントの描画方法を設定: W&B App で、極値（min/max）をシェード領域として表示するかどうかを対話的に選べます。
* データの忠実度を損なわずに探索: 特定のデータポイントにズームインすると、W&B が x 軸のバケットサイズを再計算します。これにより、精度を保ったままデータを探索できます。キャッシュを使って過去の集約結果を保存し、読み込み時間を短縮します。とくに大規模なデータセットを操作する場合に有効です。

### 最小値と最大値の描画方法を設定する

折れ線グラフの周囲にシェード領域を表示して、最小値と最大値を表示・非表示にできます。

次の画像は青い折れ線グラフを示しています。水色のシェード領域が各バケットの最小値と最大値を表します。

{{< img src="/images/app_ui/shaded-areas.png" alt="シェード付きの信頼領域" >}}

折れ線グラフで最小値と最大値を描画する方法は 3 つあります:

* **Never**: min/max をシェード領域として表示しません。x 軸バケットに沿った集約線のみを表示します。
* **On hover**: チャートにカーソルを合わせたときだけ、min/max のシェード領域が動的に表示されます。視認性を保ちつつ、範囲を対話的に確認できます。
* **Always**: チャート内のすべてのバケットで、min/max のシェード領域を常に表示します。値の全範囲を常時可視化できますが、多数の runs を同時に表示している場合は視覚的ノイズが増えることがあります。

デフォルトでは、最小値と最大値はシェード領域として表示されません。シェード領域のいずれかを表示するには、次の手順に従ってください:

{{< tabpane text=true >}}
{{% tab header="Workspace 内のすべてのチャート" value="all_charts" %}}
1. W&B の Project に移動します
2. 左側のタブで Workspace アイコンを選択します
3. 画面右上、**Add panels** ボタンの左にある歯車アイコンを選択します。
4. 表示される UI スライダーで **Line plots** を選択します
5. **Point aggregation** セクションの **Show min/max values as a shaded area** ドロップダウンから **On over** もしくは **Always** を選択します。
{{% /tab %}}

{{% tab header="Workspace 内の個別チャート" value="single_chart"%}}
1. W&B の Project に移動します
2. 左側のタブで Workspace アイコンを選択します
3. full fidelity モードを有効にしたい折れ線グラフ パネルを選択します
4. 表示されたモーダル内で、**Show min/max values as a shaded area** のドロップダウンから **On hover** もしくは **Always** を選択します。
{{% /tab %}}
{{< /tabpane >}}


### データの忠実度を損なわずにデータを探索する

極端値やスパイクなどの重要なポイントを見逃すことなく、データセットの特定領域を分析できます。折れ線グラフをズームインすると、W&B は各バケットで最小値・最大値・平均値を計算するために使うバケットサイズを調整します。

{{< img src="/images/app_ui/zoom_in.gif" alt="プロットのズーム機能" >}}

W&B はデフォルトで、x 軸を動的に 1000 個のバケットに分割します。各バケットについて、W&B は次の値を計算します:

- **Minimum**: そのバケット内で最も小さい値。
- **Maximum**: そのバケット内で最も大きい値。
- **Average**: そのバケット内の全ポイントの平均値。

W&B は、バケット化してもデータ表現の完全性を保ち、あらゆるプロットで極端値を含める方法で値を描画します。1,000 ポイント以下までズームインすると、full fidelity モードでは追加の集約を行わず、すべてのデータポイントを描画します。

折れ線グラフをズームインするには、次の手順に従ってください:

1. W&B の Project に移動します
2. 左側のタブで Workspace アイコンを選択します
3. 必要に応じて、Workspace に折れ線グラフ パネルを追加するか、既存の折れ線グラフ パネルに移動します。
4. クリックしてドラッグし、ズームインしたい領域を選択します。

{{% alert title="折れ線グラフのグルーピングと式" %}}
Line Plot Grouping を使用する場合、選択したモードに応じて W&B は次を適用します:

- **非ウィンドウ サンプリング（grouping）**: 複数の run 間で x 軸上のポイントを揃えます。同じ x 値を共有するポイントが複数ある場合は平均を取り、そうでなければ離散点として表示されます。
- **ウィンドウ サンプリング（grouping と expressions）**: x 軸を、250 個のバケットまたは最長の線のポイント数（小さい方）に分割します。各バケット内のポイントを平均します。
- **フルフィデリティ（grouping と expressions）**: 非ウィンドウ サンプリングに近い動作ですが、パフォーマンスと詳細のバランスを取るために、run ごとに最大 500 ポイントを取得します。
{{% /alert %}}

 
## Random sampling

Random sampling は、ランダムに抽出した 1500 ポイントで折れ線グラフを描画します。データポイントが非常に多い場合、パフォーマンス面で有効です。

{{% alert color="warning" %}}
Random sampling は非決定的にサンプリングします。つまり、重要な外れ値やスパイクが除外されることがあり、その結果、データの精度が低下する場合があります。
{{% /alert %}}


### ランダムサンプリングを有効にする
デフォルトでは W&B は full fidelity モードを使用します。ランダムサンプリングを有効にするには、次の手順に従ってください:

{{< tabpane text=true >}}
{{% tab header="Workspace 内のすべてのチャート" value="all_charts" %}}
1. W&B の Project に移動します
2. 左側のタブで Workspace アイコンを選択します
3. 画面右上、**Add panels** ボタンの左にある歯車アイコンを選択します。
4. 表示される UI スライダーで **Line plots** を選択します
5. **Point aggregation** セクションで **Random sampling** を選択します
{{% /tab %}}

{{% tab header="Workspace 内の個別チャート" value="single_chart"%}}
1. W&B の Project に移動します
2. 左側のタブで Workspace アイコンを選択します
3. ランダムサンプリングを有効にしたい折れ線グラフ パネルを選択します
4. 表示されたモーダル内の **Point aggregation method** セクションで **Random sampling** を選択します
{{% /tab %}}
{{< /tabpane >}}



### サンプリングしていないデータへのアクセス

[W&B Run API]({{< relref path="/ref/python/public-api/runs.md" lang="ja" >}}) を使うと、run の間にログされたメトリクスの完全な履歴に アクセス できます。次の例は、特定の run から loss の値を取得・処理する方法を示しています:


```python
# W&B API を初期化する
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")

# 'Loss' メトリクスの履歴を取得する
history = run.scan_history(keys=["Loss"])

# 履歴から loss の値を取り出す
losses = [row["Loss"] for row in history]
```