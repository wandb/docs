---
title: PyTorch Geometric
menu:
  default:
    identifier: pytorch-geometric
    parent: integrations
weight: 310
---

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)、または PyG は、幾何学的ディープラーニングのための最も人気のあるライブラリのひとつであり、W&B はグラフの可視化や実験管理の追跡に非常に優れた連携を発揮します。

PyTorch Geometric をインストールしたら、次のステップで始めましょう。

## サインアップと API キーの作成

API キーは、あなたのマシンを W&B に認証します。API キーはユーザープロフィールから生成できます。

{{% alert %}}
よりスムーズな方法として、[W&B 認証ページ](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザーアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックして API キーを表示します。表示された API キーをコピーしてください。API キーを非表示にするにはページを再読み込みしてください。

## `wandb` ライブラリのインストールとログイン

ローカルに `wandb` ライブラリをインストールし、ログインするには：

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) をご自身の API キーに設定します。

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

{{% tab header="Python ノートブック" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## グラフを可視化する

入力グラフの辺数やノード数など、グラフの詳細を保存できます。W&B は plotly チャートや HTML パネルのログにも対応しているため、グラフの可視化を作成すれば W&B に記録することが可能です。

### PyVis を使う

次のスニペットは、PyVis と HTML を使った可視化方法の例です。

```python
from pyvis.network import Network
import wandb

with wandb.init(project=’graph_vis’) as run:
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

    # PyG のグラフから PyVis ネットワークへ辺を追加
    for e in tqdm(g.edge_index.T):
        src = e[0].item()
        dst = e[1].item()

        net.add_node(dst)
        net.add_node(src)
        
        net.add_edge(src, dst, value=0.1)

    # PyVis で可視化したものを HTML ファイルとして保存
    net.show("graph.html")
    run.log({"eda/graph": wandb.Html("graph.html")})
```

{{< img src="/images/integrations/pyg_graph_wandb.png" alt="インタラクティブなグラフの可視化" >}}

### Plotly を使う

plotly でグラフ可視化をするには、まず PyG グラフを networkx オブジェクトに変換します。その後、ノードとエッジの両方について Plotly の scatter プロットを作成する必要があります。以下のスニペットはこの処理の一例です。

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

{{< img src="/images/integrations/pyg_graph_plotly.png" alt="サンプル関数で作成し、W&B Table に保存した可視化例" >}}

## メトリクスを記録する

W&B を使えば実験や関連メトリクス（損失関数や正解率など）を追跡できます。トレーニングループに下記のコードを追加してください。

```python
with wandb.init(project="my_project", entity="my_entity") as run:
    run.log({
        'train/loss': training_loss,
        'train/acc': training_acc,
        'val/loss': validation_loss,
        'val/acc': validation_acc
        })
```

{{< img src="/images/integrations/pyg_metrics.png" alt="エポックごとの hits@K メトリクス" >}}

## その他のリソース

- [PyTorch Geometric でグラフニューラルネットワークを使った Amazon 製品レコメンデーション](https://wandb.ai/manan-goel/gnn-recommender/reports/Recommending-Amazon-Products-using-Graph-Neural-Networks-in-PyTorch-Geometric--VmlldzozMTA3MzYw#what-does-the-data-look-like?)
- [PyTorch Geometric での Point Cloud 分類](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)
- [PyTorch Geometric を用いた Point Cloud セグメンテーション](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)