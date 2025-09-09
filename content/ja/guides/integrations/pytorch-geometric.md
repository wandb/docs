---
title: PyTorch Geometric
menu:
  default:
    identifier: ja-guides-integrations-pytorch-geometric
    parent: integrations
weight: 310
---

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)（PyG）は、幾何学的ディープラーニングのための最も人気のあるライブラリのひとつで、グラフの可視化や実験の追跡において W&B と非常に相性良く動作します。

PyTorch Geometric をインストールしたら、次の手順で始めましょう。

## サインアップして API キーを作成

API キーは、お使いのマシンを W&B に認証するためのものです。API キーはユーザー プロファイルから生成できます。

{{% alert %}}
よりスムーズに進めるには、[W&B authorization page](https://wandb.ai/authorize) から直接 API キーを生成できます。表示された API キーをコピーし、パスワード マネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザー プロファイル アイコンをクリックします。
1. 「**User Settings**」を選び、「**API Keys**」セクションまでスクロールします。
1. 「**Reveal**」をクリックします。表示された API キーをコピーします。API キーを隠すには、ページを再読み込みします。

## `wandb` ライブラリをインストールしてログイン

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` をあなたの API キーに設定する[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})にします。

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

入力グラフについて、エッジ数、ノード数などの詳細を保存できます。W&B は Plotly チャートや HTML パネルのログに対応しているため、作成したグラフの可視化も W&B に記録できます。

### PyVis を使う

以下のスニペットは、PyVis と HTML でそれを行う方法の一例です。

```python
from pyvis.network import Network
import wandb

with wandb.init(project=’graph_vis’) as run:
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

    # PyG のグラフから PyVis のネットワークへエッジを追加
    for e in tqdm(g.edge_index.T):
        src = e[0].item()
        dst = e[1].item()

        net.add_node(dst)
        net.add_node(src)
        
        net.add_edge(src, dst, value=0.1)

    # PyVis の可視化を HTML ファイルに保存
    net.show("graph.html")
    run.log({"eda/graph": wandb.Html("graph.html")})
```

{{< img src="/images/integrations/pyg_graph_wandb.png" alt="インタラクティブなグラフ可視化" >}}

### Plotly を使う

Plotly でグラフを可視化するには、まず PyG のグラフを networkx のオブジェクトに変換する必要があります。続いて、ノードとエッジの両方に対して Plotly の散布図を作成します。以下のスニペットはそのために使えます。

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

{{< img src="/images/integrations/pyg_graph_plotly.png" alt="サンプル関数で作成し、W&B の Table にログした可視化。" >}}

## メトリクスをログする

W&B を使って、実験や関連するメトリクス（損失、精度など）を追跡できます。トレーニング ループに次の行を追加してください:

```python
with wandb.init(project="my_project", entity="my_entity") as run:
    run.log({
        'train/loss': training_loss,
        'train/acc': training_acc,
        'val/loss': validation_loss,
        'val/acc': validation_acc
        })
```

{{< img src="/images/integrations/pyg_metrics.png" alt="エポックに対する hits@K メトリクス" >}}

## 参考資料

- [PyTorch Geometric でグラフ ニューラルネットワークを用いた Amazon 製品のレコメンデーション](https://wandb.ai/manan-goel/gnn-recommender/reports/Recommending-Amazon-Products-using-Graph-Neural-Networks-in-PyTorch-Geometric--VmlldzozMTA3MzYw#what-does-the-data-look-like?)
- [PyTorch Geometric を用いた点群分類](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)
- [PyTorch Geometric と Dynamic Graph CNN を用いた点群セグメンテーション](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)