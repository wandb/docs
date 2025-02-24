---
title: PyTorch Geometric
menu:
  default:
    identifier: ko-guides-integrations-pytorch-geometric
    parent: integrations
weight: 310
---

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) 또는 PyG는 기하 딥러닝을 위한 가장 인기 있는 라이브러리 중 하나이며 W&B는 그래프를 시각화하고 Experiments를 추적하는 데 매우 효과적입니다.

Pytorch Geometric을 설치한 후 다음 단계에 따라 시작하세요.

## 가입하고 API 키를 생성하세요

API 키는 사용자의 장치를 W&B에 인증합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
보다 간소화된 접근 방식을 위해 [https://wandb.ai/authorize](https://wandb.ai/authorize)로 직접 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장하세요.
{{% /alert %}}

1. 오른쪽 상단 모서리에 있는 사용자 프로필 아이콘을 클릭합니다.
2. **User Settings**를 선택한 다음 **API Keys** 섹션으로 스크롤합니다.
3. **Reveal**을 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고칩니다.

## `wandb` 라이브러리를 설치하고 로그인합니다

로컬에 `wandb` 라이브러리를 설치하고 로그인하려면 다음을 수행하세요.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. API 키에 [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}}) `WANDB_API_KEY`를 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` 라이브러리를 설치하고 로그인합니다.

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

## 그래프 시각화

엣지 수, 노드 수 등을 포함한 입력 그래프에 대한 세부 정보를 저장할 수 있습니다. W&B는 Plotly 차트와 HTML 패널 로깅을 지원하므로 그래프에 대해 생성하는 모든 시각화를 W&B에 로깅할 수도 있습니다.

### PyVis 사용

다음 스니펫은 PyVis 및 HTML을 사용하여 이를 수행하는 방법을 보여줍니다.

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

### Plotly 사용

Plotly를 사용하여 그래프 시각화를 생성하려면 먼저 PyG 그래프를 networkx 오브젝트로 변환해야 합니다. 그런 다음 노드와 엣지 모두에 대해 Plotly 산점도를 만들어야 합니다. 아래 스니펫을 이 작업에 사용할 수 있습니다.

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

## 메트릭 로그

W&B를 사용하여 Experiments 및 관련 메트릭(예: 손실 함수, 정확도 등)을 추적할 수 있습니다. 다음 줄을 트레이닝 루프에 추가합니다.

```python
wandb.log({
	‘train/loss’: training_loss,
	‘train/acc’: training_acc,
	‘val/loss’: validation_loss,
	‘val/acc’: validation_acc
})
```

{{< img src="/images/integrations/pyg_metrics.png" alt="Plots from W&B showing how the hits@K metric changes over epochs for different values of K." >}}

## 추가 자료

- [PyTorch Geometric에서 그래프 신경망을 사용하여 아마존 제품 추천](https://wandb.ai/manan-goel/gnn-recommender/reports/Recommending-Amazon-Products-using-Graph-Neural-Networks-in-PyTorch-Geometric--VmlldzozMTA3MzYw#what-does-the-data-look-like?)
- [PyTorch Geometric을 사용한 포인트 클라우드 분류](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)
- [PyTorch Geometric을 사용한 포인트 클라우드 분할](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)
