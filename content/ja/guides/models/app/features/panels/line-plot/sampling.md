---
title: ポイント集約
menu:
  default:
    identifier: ja-guides-models-app-features-panels-line-plot-sampling
    parent: line-plot
weight: 20
---

Use point aggregation methods within your line plots for improved data visualization accuracy and performance. There are two types of point aggregation modes: [full fidelity]({{< relref path="#full-fidelity" lang="ja" >}}) and [random sampling]({{< relref path="#random-sampling" lang="ja" >}}). W&B uses full fidelity mode by default.

## Full fidelity

Full fidelity modeを使用すると、W&Bはデータポイントの数に基づいてx軸を動的なバケットに分割します。そして、それぞれのバケット内の最小、最大、平均値を計算し、線プロットのポイント集約をレンダリングします。

フルフィデリティモードを使用する際のポイント集約の3つの主な利点は次のとおりです:

* 極値とスパイクを保持する: データ内の極値とスパイクを保持します。
* 最小値と最大値のレンダリングを設定する: W&Bアプリを使用して、極値（最小/最大）を影付きエリアとして表示するかどうかをインタラクティブに決定できます。
* データの忠実度を失わずにデータを探索する: W&Bは特定のデータポイントにズームインするとx軸のバケットサイズを再計算します。これにより、正確さを失うことなくデータを探索できることを保証します。キャッシュを使用して以前に計算された集計を保存することで、ロード時間を短縮するのに役立ちます。これは、特に大規模なデータセットをナビゲートしている場合に便利です。

### 最小値と最大値のレンダリングの設定

線プロットの周囲に影付きのエリアを使って最小値と最大値を表示または非表示にします。

次の画像は、青い線プロットを示しています。薄い青の影付きエリアは各バケットの最小値と最大値を表しています。

{{< img src="/images/app_ui/shaded-areas.png" alt="" >}}

線プロットで最小値と最大値をレンダリングする方法は3通りあります:

* **Never**: 最小/最大値は影付きエリアとして表示されません。x軸のバケット全体に渡る集約された線だけを表示します。
* **On hover**: グラフにカーソルを合わせると、最小/最大値の影付きエリアが動的に表示されます。このオプションは、ビューをシンプルに保ちながら、範囲をインタラクティブに検査することを可能にします。
* **Always**: 最小/最大の影付きエリアは常にグラフのすべてのバケットで一貫して表示され、常に全範囲の値を視覚化するのに役立ちます。これにより、グラフに多くのrunsが視覚化されている場合、視覚的なノイズが発生する可能性があります。

デフォルトでは、最小値と最大値は影付きエリアとして表示されません。影付きエリアオプションの1つを表示するには、以下の手順に従ってください:

{{< tabpane text=true >}}
{{% tab header="All charts in a workspace" value="all_charts" %}}
1. W&Bプロジェクトに移動します。
2. 左のタブで**Workspace**アイコンを選択します。
3. 画面の右上隅にある歯車アイコンを、**Add panels**ボタンの左側に選択します。
4. 表示されるUIスライダーから**Line plots**を選択します。
5. **Point aggregation**セクション内で、**Show min/max values as a shaded area**ドロップダウンメニューから**On hover**または**Always**を選択します。
{{% /tab %}}

{{% tab header="Individual chart in a workspace" value="single_chart"%}}
1. W&Bプロジェクトに移動します。
2. 左のタブで**Workspace**アイコンを選択します。
3. フルフィデリティモードを有効にしたい線プロットパネルを選択します。
4. 表示されるモーダル内で、**Show min/max values as a shaded area**ドロップダウンメニューから**On hover**または**Always**を選択します。
{{% /tab %}}
{{< /tabpane >}}

### データの忠実度を失わずにデータを探索する

データセットの特定の領域を分析し、極値やスパイクなどの重要なポイントを見逃さないようにします。線プロットをズームインすると、W&Bは各バケット内の最小、最大、および平均値を計算するために使用されるバケットサイズを調整します。

{{< img src="/images/app_ui/zoom_in.gif" alt="" >}}

W&Bはデフォルトでx軸を1000のバケットに動的に分割します。各バケットに対し、W&Bは以下の値を計算します:

- **Minimum**: そのバケット内の最小値。
- **Maximum**: そのバケット内の最大値。
- **Average**: そのバケット内のすべてのポイントの平均値。

W&Bは、すべてのプロットでデータの完全な表現を保持し、極値を含める方法でバケット内の値をプロットします。1,000ポイント以下にズームインすると、フルフィデリティモードは追加の集約なしにすべてのデータポイントをレンダリングします。

線プロットをズームインするには、次の手順に従います:

1. W&Bプロジェクトに移動します。
2. 左のタブで**Workspace**アイコンを選択します。
3. 必要に応じて、ワークスペースに線プロットパネルを追加するか、既存の線プロットパネルに移動します。
4. ズームインしたい特定の領域を選択するためにクリックしてドラッグします。

{{% alert title="Line plot grouping and expressions" %}}
Line Plot Groupingを使用すると、W&Bは選択されたモードに基づいて以下を適用します:

- **Non-windowed sampling (grouping)**: x軸でrunsを超えてポイントを整列させます。複数のポイントが同じx値を共有する場合、平均が取られ、そうでない場合は離散的なポイントとして表示されます。
- **Windowed sampling (grouping and expressions)**: x軸を250のバケットまたは最も長い線のポイント数に分割します（いずれか小さい方）。W&Bは各バケット内のポイントの平均を取ります。
- **Full fidelity (grouping and expressions)**: 非ウィンドウ化サンプリングに似ていますが、パフォーマンスと詳細のバランスを取るためにrunごとに最大500ポイントを取得します。
{{% /alert %}}

## Random sampling

Random samplingはラインプロットをレンダリングするために1,500のランダムにサンプリングされたポイントを使用します。大量のデータポイントがある場合、パフォーマンスの理由でランダムサンプリングが有用です。

{{% alert color="warning" %}}
Random samplingは非決定的にサンプリングします。これは、ランダムサンプリングが時々データ内の重要なアウトライヤーやスパイクを除外し、したがってデータの正確性を低下させることを意味します。
{{% /alert %}}

### ランダムサンプリングを有効にする
デフォルトでは、W&Bはフルフィデリティモードを使用します。ランダムサンプリングを有効にするには、次の手順に従います:

{{< tabpane text=true >}}
{{% tab header="All charts in a workspace" value="all_charts" %}}
1. W&Bプロジェクトに移動します。
2. 左のタブで**Workspace**アイコンを選択します。
3. 画面の右上隅にある歯車アイコンを、**Add panels**ボタンの左側に選択します。
4. 表示されるUIスライダーから**Line plots**を選択します。
5. **Point aggregation**セクションから**Random sampling**を選択します。
{{% /tab %}}

{{% tab header="Individual chart in a workspace" value="single_chart"%}}
1. W&Bプロジェクトに移動します。
2. 左のタブで**Workspace**アイコンを選択します。
3. ランダムサンプリングを有効にしたい線プロットパネルを選択します。
4. 表示されるモーダル内で、**Point aggregation method**セクションから**Random sampling**を選択します。
{{% /tab %}}
{{< /tabpane >}}

### サンプリングされていないデータへのアクセス

[W&B Run API]({{< relref path="/ref/python/public-api/run.md" lang="ja" >}})を使用して、run中にログされたメトリクスの完全な履歴にアクセスできます。次の例は、特定のrunから損失値を取得し処理する方法を示しています:

```python
# W&B APIを初期化
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")

# 'Loss'メトリクスの履歴を取得
history = run.scan_history(keys=["Loss"])

# 履歴から損失値を抽出
losses = [row["Loss"] for row in history]
```