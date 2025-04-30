---
title: PyTorch Geometric
menu:
  default:
    identifier: ja-guides-integrations-pytorch-geometric
    parent: integrations
weight: 310
---

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) または PyG は、最も人気のある幾何学的ディープラーニングのためのライブラリの1つであり、W&B はそれと非常に良く連携し、グラフの可視化と実験の追跡を行うことができます。

PyTorch Geometric をインストールした後、以下の手順に従ってください。

## サインアップとAPI キーの作成

APIキーは、あなたのマシンをW&Bに認証します。APIキーはユーザープロフィールから生成できます。

{{% alert %}}
よりスムーズな方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize)に直接アクセスしてAPIキーを生成することができます。表示されるAPIキーをコピーし、パスワード管理ツールなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
2. **ユーザー設定**を選択し、**API キー**セクションまでスクロールします。
3. **Reveal** をクリックします。表示されたAPIキーをコピーします。APIキーを隠すには、ページをリロードしてください。

## `wandb` ライブラリのインストールとログイン

`wandb` ライブラリをローカルにインストールし、ログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をAPIキーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールし、ログインします。

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## グラフの可視化

入力グラフの詳細（エッジ数、ノード数など）を保存できます。W&B は plotly グラフと HTML パネルのログ記録をサポートしているため、グラフのために作成したあらゆる可視化を W&B にログすることができます。

### PyVis を使用する

以下のスニペットは、PyVis と HTML を使ってそれを行う方法を示しています。

```python
from pyvis.network import Network
import wandb

wandb.init(project=’graph_vis’)
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# PyG グラフから PyVis ネットワークへのエッジを追加
for e in tqdm(g.edge_index.T):
    src = e[0].item()
    dst = e[1].item()

    net.add_node(dst)
    net.add_node(src)
    
    net.add_edge(src, dst, value=0.1)

# PyVisの可視化をHTMLファイルに保存
net.show("graph.html")
wandb.log({"eda/graph": wandb.Html("graph.html")})
wandb.finish()
```

{{< img src="/images/integrations/pyg_graph_wandb.png" alt="この画像は、インタラクティブな HTML 可視化として入力グラフを示しています。" >}}

### Plotly を使用する

Plotly を使用してグラフの可視化を作成するには、まず PyG グラフを networkx オブジェクトに変換する必要があります。その後、ノードとエッジのために Plotly スキャッタープロットを作成する必要があります。このタスクには以下のスニペットが使用できます。

```python
def create_vis(graph):
    G = to_networkx(graph)
    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        line_width=2
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout())

    return fig


wandb.init(project=’visualize_graph’)
wandb.log({‘graph’: wandb.Plotly(create_vis(graph))})
wandb.finish()
```

{{< img src="/images/integrations/pyg_graph_plotly.png" alt="この視覚化結果は、例の関数を使用して作成され、W&B テーブル内に記録されました。" >}}

## メトリクスのログ化

損失関数、精度などのメトリクスを含む実験を追跡するためにW&Bを使用することができます。トレーニングループに次の行を追加してください：

```python
wandb.log({
	‘train/loss’: training_loss,
	‘train/acc’: training_acc,
	‘val/loss’: validation_loss,
	‘val/acc’: validation_acc
})
```

{{< img src="/images/integrations/pyg_metrics.png" alt="W&Bからのプロットが、異なるK値に対するエポックごとのhits@Kメトリクスの変化を示しています。" >}}

## その他のリソース

- [Recommending Amazon Products using Graph Neural Networks in PyTorch Geometric](https://wandb.ai/manan-goel/gnn-recommender/reports/Recommending-Amazon-Products-using-Graph-Neural-Networks-in-PyTorch-Geometric--VmlldzozMTA3MzYw#what-does-the-data-look-like?)
- [Point Cloud Classification using PyTorch Geometric](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)
- [Point Cloud Segmentation using PyTorch Geometric](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)