---
title: PyTorch Geometric
menu:
  default:
    identifier: ja-guides-integrations-pytorch-geometric
    parent: integrations
weight: 310
---

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) または PyGは、幾何学的ディープラーニング向けの最も人気なライブラリの一つであり、W&B との相性が非常に良く、グラフの可視化や実験管理に活用できます。

PyTorch Geometric のインストール後、次の手順で始めましょう。

## サインアップとAPIキーの作成

APIキーは、あなたのマシンを W&B に認証するためのものです。APIキーはユーザープロフィールから発行できます。

{{% alert %}}
より簡単な方法として、[W&B認証ページ](https://wandb.ai/authorize) に直接アクセスしてAPIキーを発行できます。表示されたAPIキーをコピーし、パスワードマネージャ等の安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
2. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
3. **Reveal** をクリックして表示されたAPIキーをコピーします。APIキーを非表示にしたい場合は、ページを再読み込みしてください。

## `wandb`ライブラリのインストールとログイン

ローカル環境に`wandb`ライブラリをインストールし、ログインする手順です。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) にAPIキーを設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb`ライブラリをインストールし、ログインします。

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

入力グラフのエッジ数やノード数などの詳細情報も保存できます。W&B では plotly チャートやHTMLパネルのログが可能なので、作成したグラフの可視化もW&Bに記録できます。

### PyVis の利用

以下のスニペットは、PyVis とHTMLを使ってグラフを可視化する方法の例です。

```python
from pyvis.network import Network
import wandb

with wandb.init(project=’graph_vis’) as run:
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

    # PyGグラフのエッジをPyVisネットワークに追加
    for e in tqdm(g.edge_index.T):
        src = e[0].item()
        dst = e[1].item()

        net.add_node(dst)
        net.add_node(src)
        
        net.add_edge(src, dst, value=0.1)

    # PyVisの可視化をHTMLファイルとして保存
    net.show("graph.html")
    run.log({"eda/graph": wandb.Html("graph.html")})
```

{{< img src="/images/integrations/pyg_graph_wandb.png" alt="インタラクティブなグラフ可視化" >}}

### Plotly の利用

plotly でグラフ可視化を行うには、まずPyGのグラフをnetworkxオブジェクトに変換します。その後、ノードとエッジそれぞれ用にPlotlyの散布図を作成します。下記のスニペットはその一例です。

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


with wandb.init(project=’visualize_graph’) as run:
    run.log({‘graph’: wandb.Plotly(create_vis(graph))})
```

{{< img src="/images/integrations/pyg_graph_plotly.png" alt="サンプル関数で作成し、W&B Tableに記録した可視化例" >}}

## メトリクスのログ

W&Bを利用すれば、損失関数、精度などのメトリクスや、Experimentsのトラッキングも可能です。以下の行をトレーニングループに追加しましょう。

```python
with wandb.init(project="my_project", entity="my_entity") as run:
    run.log({
        'train/loss': training_loss,
        'train/acc': training_acc,
        'val/loss': validation_loss,
        'val/acc': validation_acc
        })
```

{{< img src="/images/integrations/pyg_metrics.png" alt="エポック毎のhits@Kメトリクス" >}}

## その他リソース

- [Recommending Amazon Products using Graph Neural Networks in PyTorch Geometric](https://wandb.ai/manan-goel/gnn-recommender/reports/Recommending-Amazon-Products-using-Graph-Neural-Networks-in-PyTorch-Geometric--VmlldzozMTA3MzYw#what-does-the-data-look-like?)
- [Point Cloud Classification using PyTorch Geometric](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)
- [Point Cloud Segmentation using PyTorch Geometric](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)