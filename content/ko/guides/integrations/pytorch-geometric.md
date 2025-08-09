---
title: PyTorch Geometric
menu:
  default:
    identifier: ko-guides-integrations-pytorch-geometric
    parent: integrations
weight: 310
---

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) 또는 PyG는 지오메트릭 딥러닝 분야에서 가장 인기 있는 라이브러리 중 하나이며, W&B는 그래프 시각화 및 실험 추적을 위해 PyG와 매우 잘 호환됩니다.

PyTorch Geometric을 설치한 후, 아래의 단계를 따라 시작해보세요.

## 가입 및 API 키 생성

API 키는 사용자의 머신을 W&B에 인증해줍니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
좀 더 간단한 방법으로는, [W&B 인증 페이지](https://wandb.ai/authorize)에 바로 접속해 API 키를 생성할 수 있습니다. 표시되는 API 키를 복사해 비밀번호 관리자 등 안전한 장소에 보관하세요.
{{% /alert %}}

1. 우측 상단의 사용자 프로필 아이콘을 클릭합니다.
1. **User Settings**를 선택하고, **API Keys** 섹션까지 스크롤합니다.
1. **Reveal**을 클릭하여 API 키를 확인하고 복사합니다. 키를 다시 숨기려면 페이지를 새로고침하세요.

## `wandb` 라이브러리 설치 및 로그인

로컬 환경에 `wandb` 라이브러리를 설치하고 로그인하려면:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 본인의 API 키로 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치하고 로그인합니다.

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

## 그래프 시각화하기

입력 그래프의 엣지 수, 노드 수 등 다양한 정보를 남길 수 있습니다. W&B는 plotly 차트와 HTML 패널에 대한 로깅을 지원하므로, 그래프에 대한 어떤 시각화든 W&B에 함께 기록할 수 있습니다.

### PyVis 사용하기

아래 예시는 PyVis와 HTML을 이용해 시각화를 기록하는 방법을 보여줍니다.

```python
from pyvis.network import Network
import wandb

with wandb.init(project=’graph_vis’) as run:
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

    # PyG 그래프의 엣지를 PyVis 네트워크에 추가
    for e in tqdm(g.edge_index.T):
        src = e[0].item()
        dst = e[1].item()

        net.add_node(dst)
        net.add_node(src)
        
        net.add_edge(src, dst, value=0.1)

    # PyVis 시각화 결과를 HTML 파일로 저장
    net.show("graph.html")
    run.log({"eda/graph": wandb.Html("graph.html")})
```

{{< img src="/images/integrations/pyg_graph_wandb.png" alt="Interactive graph visualization" >}}

### Plotly 사용하기

plotly로 그래프를 시각화하려면 먼저 PyG 그래프를 networkx 오브젝트로 변환해야 합니다. 이후, 노드와 엣지에 대한 plotly scatter plot을 각각 작성해야 합니다. 아래의 코드를 참고하세요.

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

{{< img src="/images/integrations/pyg_graph_plotly.png" alt="A visualization created using the example function and logged inside a W&B Table." >}}

## 메트릭 로깅

W&B를 이용해 실험 및 관련 메트릭(예: loss 함수, accuracy 등)을 추적할 수 있습니다. 트레이닝 루프에 아래 코드를 추가하세요.

```python
with wandb.init(project="my_project", entity="my_entity") as run:
    run.log({
        'train/loss': training_loss,
        'train/acc': training_acc,
        'val/loss': validation_loss,
        'val/acc': validation_acc
        })
```

{{< img src="/images/integrations/pyg_metrics.png" alt="hits@K metrics over epochs" >}}

## 추가 자료

- [Recommending Amazon Products using Graph Neural Networks in PyTorch Geometric](https://wandb.ai/manan-goel/gnn-recommender/reports/Recommending-Amazon-Products-using-Graph-Neural-Networks-in-PyTorch-Geometric--VmlldzozMTA3MzYw#what-does-the-data-look-like?)
- [Point Cloud Classification using PyTorch Geometric](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)
- [Point Cloud Segmentation using PyTorch Geometric](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)