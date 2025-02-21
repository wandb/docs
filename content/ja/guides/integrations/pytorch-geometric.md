---
title: PyTorch Geometric
menu:
  default:
    identifier: ja-guides-integrations-pytorch-geometric
    parent: integrations
weight: 310
---

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)、または PyG は、幾何学的 ディープラーニング で最も人気のある ライブラリの 1 つであり、W&B は、グラフの 可視化 と Experiments の 追跡において非常にうまく連携します。

Pytorch Geometric をインストールしたら、次の手順に従って開始してください。

## サインアップして APIキー を作成する

APIキー は、お使いのマシンを W&B に対して認証します。APIキー は、 ユーザー プロフィールから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にある ユーザー プロフィール アイコンをクリックします。
2. 「**User Settings**（ユーザー 設定）」を選択し、「**API Keys**（APIキー ）」セクションまでスクロールします。
3. 「**Reveal**（表示）」をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページをリロードします。

## `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を APIキー に設定します。

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

## グラフを 可視化 する

エッジ数、ノード数など、入力グラフに関する詳細を保存できます。W&B は plotly チャートと HTML パネルの ログ記録 をサポートしているため、グラフ用に作成した 可視化 は W&B にも ログ記録 できます。

### PyVis の使用

次のスニペットは、PyVis と HTML でそれを行う方法を示しています。

```python
from pyvis.network import Network
Import wandb

wandb.init(project=’graph_vis’)
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# Add the edges from the PyG graph to the PyVis network
# PyG グラフのエッジを PyVis ネットワークに追加します
for e in tqdm(g.edge_index.T):
    src = e[0].item()
    dst = e[1].item()

    net.add_node(dst)
    net.add_node(src)
    
    net.add_edge(src, dst, value=0.1)

# Save the PyVis visualisation to a HTML file
# PyVis の 可視化 を HTML ファイルに保存します
net.show("graph.html")
wandb.log({"eda/graph": wandb.Html("graph.html")})
wandb.finish()
```

{{< img src="/images/integrations/pyg_graph_wandb.png" alt="This image shows the input graph as an interactive HTML visualization." >}}

### Plotly の使用

plotly を使用してグラフの 可視化 を作成するには、まず PyG グラフを networkx オブジェクトに変換する必要があります。次に、ノードとエッジの両方に対して Plotly 散布図を作成する必要があります。次のスニペットは、このタスクに使用できます。

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

## メトリクス の ログ記録

W&B を使用して、Experiments や、損失関数、精度などの関連 メトリクス を 追跡 できます。次の行を トレーニング ループに追加します。

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
- [PyTorch Geometric を使用した Point Cloud Classification](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)
- [PyTorch Geometric を使用した Point Cloud Segmentation](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)
