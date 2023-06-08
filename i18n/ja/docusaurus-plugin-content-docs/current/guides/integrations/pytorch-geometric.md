---
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PyTorch Geometric

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)またはPyGは、幾何学的ディープラーニングにおいて最も人気のあるライブラリの1つであり、W&Bはグラフの可視化や実験のトラッキングにこれと非常にうまく機能します。

## はじめに

PyTorch Geometricをインストールした後、wandbライブラリをインストールし、ログインしてください。

<Tabs
  defaultValue="script"
  values={[
    {label: 'コマンドライン', value: 'script'},
    {label: 'ノートブック', value: 'notebook'},
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

入力グラフの詳細（エッジ数、ノード数など）を保存できます。W&Bは、PlotlyチャートとHTMLパネルのログをサポートしているため、グラフに作成した可視化もW&Bにログとして記録できます。

### PyVisを使う

以下のスニペットは、PyVisとHTMLを使ってこれがどのように行われるかを示しています。

```python
from pyvis.network import Network
import wandb

wandb.init(project='graph_vis')
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# PyGグラフからエッジをPyVisネットワークに追加する
for e in tqdm(g.edge_index.T):
    src = e[0].item()
    dst = e[1].item()

    net.add_node(dst)
    net.add_node(src)
    
    net.add_edge(src, dst, value=0.1)

# PyVisの可視化をHTMLファイルに保存する
net.show("graph.html")
wandb.log({"eda/graph": wandb.Html("graph.html")})
wandb.finish()
```
| ![この画像は、インタラクティブなHTML可視化としての入力グラフを示しています。](@site/static/images/integrations/pyg_graph_wandb.png) |
|:--:|
| **この画像は、インタラクティブなHTML可視化としての入力グラフを示しています。** |

### Plotlyの使用法

グラフの可視化を作成するためにplotlyを使用するには、まずPyGグラフをnetworkxオブジェクトに変換する必要があります。その後、ノードとエッジのためのPlotly散布図を作成する必要があります。以下のスニペットがこのタスクに使用できます。

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


wandb.init(project='visualize_graph')
wandb.log({'graph': wandb.Plotly(create_vis(graph))})
wandb.finish()
```

| ![この可視化は、上記のスニペットに示されている関数を使用して作成され、W&Bテーブルの中に記録されました。](@site/static/images/integrations/pyg_graph_plotly.png) | 
|:--:| 
| **この可視化は、上記のスニペットに示されている関数を使用して作成され、W&Bテーブルの中に記録されました。** |

## メトリクスの記録
W&Bを使って、損失関数や精度などのメトリクスとともに、すべての実験をトラッキングすることができます。トレーニングループに以下の行を追加するだけで、準備完了です!



```python

wandb.log({

	‘train/loss’: training_loss,

	‘train/acc’: training_acc,

	‘val/loss’: validation_loss,

	‘val/acc’: validation_acc

})

```



| ![W＆Bのプロットによる試行回数による異なるK値に対するhits@K メトリクスの変化の可視化](@site/static/images/integrations/pyg_metrics.png) | 

|:--:| 

| **W＆Bのプロットによる試行回数による異なるK値に対するhits@K メトリクスの変化の可視化** |



## さらなるリソース



- [PyTorch Geometricを使ったAmazon商品のグラフニューラルネットワークによる推薦](https://wandb.ai/manan-goel/gnn-recommender/reports/Recommending-Amazon-Products-using-Graph-Neural-Networks-in-PyTorch-Geometric--VmlldzozMTA3MzYw#what-does-the-data-look-like?)

- [PyTorch Geometricを使ったポイントクラウド分類](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)

- [PyTorch Geometricを使ったポイントクラウドセグメンテーション](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)