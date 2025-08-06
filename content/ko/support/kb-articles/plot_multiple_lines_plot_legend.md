---
title: 여러 개의 선을 범례와 함께 그래프에 그리려면 어떻게 해야 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-plot_multiple_lines_plot_legend
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.plot.line_series()`를 사용하여 여러 줄이 있는 커스텀 차트를 생성할 수 있습니다. [프로젝트 페이지]({{< relref path="/guides/models/track/project-page.md" lang="ko" >}})에서 해당 라인 차트를 확인할 수 있습니다. 범례를 추가하려면 `wandb.plot.line_series()`에 `keys` 인수를 포함하세요. 예를 들면 다음과 같습니다:

```python

with wandb.init(project="my_project") as run:

    run.log(
        {
            "my_plot": wandb.plot.line_series(
                xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
            )
        }
    )
```

여러 줄이 있는 플롯(multi-line plot)에 대한 추가 설명은 **여러 줄(Multi-line)** 탭의 [여기]({{< relref path="/guides/models/track/log/plots.md#basic-charts" lang="ko" >}})에서 확인할 수 있습니다.