---
title: PyTorch Geometric
menu:
  default:
    identifier: ja-guides-integrations-pytorch-geometric
    parent: integrations
weight: 310
---

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) または PyG は、幾何学的ディープラーニングで最も人気のあるライブラリの一つであり、W&B はグラフの可視化と実験管理の追跡で非常に優れた働きをします。

Pytorch Geometric をインストールした後、次の手順に従って始めてください。

## サインアップと API キーの作成

API キーは、マシンを W&B に認証します。ユーザープロフィールから API キーを生成できます。

{{% alert %}}
より簡略化された方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キーを生成することができます。表示された API キーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリック。
1. **User Settings** を選び、**API Keys** セクションまでスクロール。
1. **Reveal** をクリック。表示された API キーをコピー。API キーを隠すには、ページを再読み込み。

## `wandb` ライブラリのインストールとログイン

ローカルに `wandb` ライブラリをインストールしてログインするには：

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を API キーに設定。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールしてログイン。

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

入力グラフの詳細（エッジ数、ノード数など）を保存できます。W&B は plotly チャートや HTML パネルのログをサポートしているので、作成したグラフの可視化も W&B にログを残せます。

### PyVis を使用

PyVis と HTML を使用するときの例は以下の通りです。

```python
from pyvis.network import Network
import wandb

wandb.init(project=’graph_vis’)
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# PyG グラフから PyVis ネットワークへのエッジ追加
for e in tqdm(g.edge_index.T):
    src = e[0].item()
    dst = e[1].item()

    net.add_node(dst)
    net.add_node(src)
    
    net.add_edge(src, dst, value=0.1)

# PyVis 可視化を HTML ファイルに保存
net.show("graph.html")
wandb.log({"eda/graph": wandb.Html("graph.html")})
wandb.finish()
```

{{< img src="/images/integrations/pyg_graph_wandb.png" alt="この画像は入力グラフをインタラクティブ HTML 可視化として表示しています。" >}}

### Plotly を使用

plotly を使用してグラフ可視化を行うには、まず PyG グラフを networkx オブジェクトに変換する必要があります。その後、ノードとエッジのための Plotly スキャッタープロットを作成する必要があります。以下のコードスニペットを使用してこのタスクを行うことができます。

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

{{< img src="/images/integrations/pyg_graph_plotly.png" alt="例示関数を使用して作成され、W&B Table内にログされた可視化。" >}}

## メトリクスをログ

W&B を使用して、損失関数や精度などの実験と関連メトリクスを追跡できます。トレーニングループに次の行を追加します：

```python
wandb.log({
	‘train/loss’: training_loss,
	‘train/acc’: training_acc,
	‘val/loss’: validation_loss,
	‘val/acc’: validation_acc
})
```

{{< img src="/images/integrations/pyg_metrics.png" alt="W&Bからのプロットが、異なるKの値に対して epochsで hits@Kメトリックがどのように変化するかを示しています。" >}}

## 追加のリソース

- [Recommending Amazon Products using Graph Neural Networks in PyTorch Geometric](https://wandb.ai/manan-goel/gnn-recommender/reports/Recommending-Amazon-Products-using-Graph-Neural-Networks-in-PyTorch-Geometric--VmlldzozMTA3MzYw#what-does-the-data-look-like?)
- [Point Cloud Classification using PyTorch Geometric](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)
- [Point Cloud Segmentation using PyTorch Geometric](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)