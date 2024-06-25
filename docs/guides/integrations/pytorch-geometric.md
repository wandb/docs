---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# PyTorch Geometric

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) または PyG は、幾何学的ディープラーニングのための最も人気のあるライブラリの一つであり、W&B はグラフの可視化と実験の追跡に非常に効果的です。

## はじめに

pytorch geometric をインストールした後、wandb ライブラリをインストールしてログインします。

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```python
pip install wandb
wandb login
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

## グラフの可視化

入力グラフの詳細（エッジの数、ノードの数など）を保存できます。W&BはplotlyチャートやHTMLパネルのログに対応しているため、グラフ用に作成した可視化をW&Bにログとして保存することができます。

### PyVisの使用

以下のスニペットは、PyVisおよびHTMLを使用してその方法を示しています。

```python
from pyvis.network import Network
Import wandb

wandb.init(project=’graph_vis’)
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# PyGグラフのエッジをPyVisネットワークに追加します
for e in tqdm(g.edge_index.T):
    src = e[0].item()
    dst = e[1].item()

    net.add_node(dst)
    net.add_node(src)
    
    net.add_edge(src, dst, value=0.1)

# PyVis可視化をHTMLファイルに保存します
net.show("graph.html")
wandb.log({"eda/graph": wandb.Html("graph.html")})
wandb.finish()
```

| ![この画像は、入力グラフを対話型HTML可視化として表示しています。](@site/static/images/integrations/pyg_graph_wandb.png) | 
|:--:| 
| **この画像は、入力グラフを対話型HTML可視化として表示しています。** |

### Plotlyの使用

Plotlyを使用してグラフの可視化を作成するには、まずPyGグラフをnetworkxオブジェクトに変換する必要があります。その後、ノードとエッジのためにPlotly散布図を作成する必要があります。以下のスニペットは、この作業のためのものです。

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

| ![この可視化は、上記スニペットで示された関数を使用して作成され、W&Bテーブルの中にログされています。](@site/static/images/integrations/pyg_graph_plotly.png) | 
|:--:| 
| **この可視化は、上記スニペットで示された関数を使用して作成され、W&Bテーブルの中にログされています。** |

## メトリクスのログ

W&Bを使用して、ロス関数や精度などのメトリクスとともに全ての実験を追跡することができます。以下の行をトレーニングループに追加するだけで、すぐに使用できます！

```python
wandb.log({
	‘train/loss’: training_loss,
	‘train/acc’: training_acc,
	‘val/loss’: validation_loss,
	‘val/acc’: validation_acc
})
```

| ![W&Bからのプロットは、異なるKの値に対するヒット@Kメトリクスがエポックごとにどのように変化するかを示しています。](@site/static/images/integrations/pyg_metrics.png) | 
|:--:| 
| **W&Bからのプロットは、異なるKの値に対するヒット@Kメトリクスがエポックごとにどのように変化するかを示しています。** |

## その他のリソース

- [Recommending Amazon Products using Graph Neural Networks in PyTorch Geometric](https://wandb.ai/manan-goel/gnn-recommender/reports/Recommending-Amazon-Products-using-Graph-Neural-Networks-in-PyTorch-Geometric--VmlldzozMTA3MzYw#what-does-the-data-look-like?)
- [Point Cloud Classification using PyTorch Geometric](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)
- [Point Cloud Segmentation using PyTorch Geometric](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)