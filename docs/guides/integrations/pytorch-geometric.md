---
title: PyTorch Geometric
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) 혹은 PyG는 기하 딥러닝에 있어 가장 인기 있는 라이브러리 중 하나이며, W&B와 함께 그래프를 시각화하고 Experiments를 추적하는 데 매우 적합합니다.

## 시작하기

pytorch geometric을 설치한 후, wandb 라이브러리를 설치하고 로그인하세요.

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

## 그래프 시각화

입력 그래프에 대한 세부 사항, 예를 들어 엣지와 노드의 수 등을 저장할 수 있습니다. W&B는 plotly 차트 및 HTML 패널을 로그하는 것을 지원하므로 그래프에 대한 시각화를 생성하면 이를 W&B에 로그할 수 있습니다.

### PyVis 사용하기

다음 코드는 PyVis와 HTML을 사용하는 방법을 보여줍니다.

```python
from pyvis.network import Network
Import wandb

wandb.init(project=’graph_vis’)
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# PyG 그래프에서 PyVis 네트워크로 엣지를 추가합니다.
for e in tqdm(g.edge_index.T):
    src = e[0].item()
    dst = e[1].item()

    net.add_node(dst)
    net.add_node(src)
    
    net.add_edge(src, dst, value=0.1)

# PyVis 시각화를 HTML 파일로 저장합니다.
net.show("graph.html")
wandb.log({"eda/graph": wandb.Html("graph.html")})
wandb.finish()
```

| ![이 이미지는 입력 그래프를 인터랙티브한 HTML 시각화로 보여줍니다.](/images/integrations/pyg_graph_wandb.png) | 
|:--:| 
| **이 이미지는 입력 그래프를 인터랙티브한 HTML 시각화로 보여줍니다.** |

### Plotly 사용하기

plotly를 사용하여 그래프 시각화를 생성하려면, 먼저 PyG 그래프를 networkx 오브젝트로 변환해야 합니다. 그 후, Plotly의 산점도를 노드와 엣지에 대해 생성해야 합니다. 아래 코드는 이 작업을 수행할 수 있습니다.

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

| ![이 시각화는 위 코드 스니펫에서 보여진 함수를 사용하여 생성되었으며 W&B Table에 로그되었습니다.](/images/integrations/pyg_graph_plotly.png) | 
|:--:| 
| **이 시각화는 위 코드 스니펫에서 보여진 함수를 사용하여 생성되었으며 W&B Table에 로그되었습니다.** |

## 메트릭 로그하기

W&B를 사용하여 모든 Experiments와 손실 함수, 정확도 등의 메트릭들을 추적할 수 있습니다. 트레이닝 루프에 다음 줄을 추가하면 됩니다.

```python
wandb.log({
	‘train/loss’: training_loss,
	‘train/acc’: training_acc,
	‘val/loss’: validation_loss,
	‘val/acc’: validation_acc
})
```

| ![W&B의 플롯은 서로 다른 K 값에 대해 에포크 동안 hits@K 메트릭이 어떻게 변하는지를 보여줍니다.](/images/integrations/pyg_metrics.png) | 
|:--:| 
| **W&B의 플롯은 서로 다른 K 값에 대해 에포크 동안 hits@K 메트릭이 어떻게 변하는지를 보여줍니다.** |

## 추가 자료

- [Recommending Amazon Products using Graph Neural Networks in PyTorch Geometric](https://wandb.ai/manan-goel/gnn-recommender/reports/Recommending-Amazon-Products-using-Graph-Neural-Networks-in-PyTorch-Geometric--VmlldzozMTA3MzYw#what-does-the-data-look-like?)
- [Point Cloud Classification using PyTorch Geometric](https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-using-PyTorch-Geometric--VmlldzozMTExMTE3)
- [Point Cloud Segmentation using PyTorch Geometric](https://wandb.ai/wandb/point-cloud-segmentation/reports/Point-Cloud-Segmentation-using-Dynamic-Graph-CNN--VmlldzozMTk5MDcy)