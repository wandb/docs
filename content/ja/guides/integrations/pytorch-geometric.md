---
title: PyTorch Geometric
menu:
  default:
    identifier: ja-guides-integrations-pytorch-geometric
    parent: integrations
weight: 310
---

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) 、または PyG は、幾何学的ディープラーニングで最も人気のあるライブラリの 1 つであり、W&B はグラフの可視化と Experiments の追跡において非常にうまく機能します。

Pytorch Geometric をインストールしたら、以下の手順に従って開始してください。

## サインアップして API キーを作成する

API キー は、お使いのマシンを W&B に対して認証します。API キーは、 ユーザー プロフィールから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にある ユーザー プロフィール アイコンをクリックします。
2. **ユーザー 設定** を選択し、**API キー** セクションまでスクロールします。
3. **表示** をクリックします。表示された API キーをコピーします。API キーを非表示にするには、ページをリロードしてください。

## `wandb` ライブラリをインストールしてログインする

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [ 環境 変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を API キー に設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリをインストールしてログインします。

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

## グラフを可視化する

エッジ数、ノード数など、入力グラフに関する詳細を保存できます。W&B は Plotly チャートと HTML パネル の ログ をサポートしているため、グラフ用に作成した 可視化 は W&B にも ログ できます。

### PyVis を使用する

次のスニペットは、PyVis と HTML でそれを行う方法を示しています。

```python
from pyvis.network import Network
Import wandb

wandb.init(project=’graph_vis’)
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# Add the edges from the PyG graph to the PyVis network
for e in tqdm(g.edge_index.T):
    src = e[0].item()
    dst = e[1].item()

    net.add_node(dst)
    net.add_node(src)
    
    net.add_edge(src, dst, value=0.1)

# Save the PyVis visualisation to a HTML file
net.show("graph.html")
wandb.log({"eda/graph": wandb.Html("graph.html")})
wandb.finish()
```

{{< img src="/images/integrations/pyg_graph_wandb.png" alt="This image shows the input graph as an interactive HTML visualization." >}}

### Plotly を使用する

Plotly を使用してグラフの 可視化 を作成するには、まず PyG グラフを networkx オブジェクトに変換する必要があります。次に、ノードとエッジの両方に対して Plotly 散布図を作成する必要があります。次のスニペットをこのタスクに使用できます。

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

{{< img src="/images/integrations/pyg_graph_plotly.png" alt="A visualization created using the example function and logged inside a W&B Table." >}}

## メトリクス を ログ する

W&B を使用して、Experiments や、損失関数、精度などの関連 メトリクス を追跡できます。次の行を トレーニング ループに追加します。

```python
wandb.log({
	‘train/loss’: training_loss,
	‘train/acc’: training_acc,
	‘val/loss’: validation_loss,
	‘val/acc’: validation_acc
})
```

{{< img src="/images/integrations/pyg_metrics.png" alt="Plots from W&B showing how the hits@K metric changes over epochs for different values of K." >}}

## その他のリソース

- [PyTorch Geometric でグラフ ニューラルネットワーク を使用して Amazon 製品を推奨する](https://wandb.ai/manan-goel/gnn-recommender/reports/Recommending-Amazon-Products-using-Graph-Neural-Networks-in-PyTorch-Geometric--VmlldzozMTA3MzYw#what-does-the-data-look-like?)
- [PyTorch Geometric を使用した ポイント クラウド分類](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)
- [PyTorch Geometric を使用した ポイント クラウドセグメンテーション](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)
