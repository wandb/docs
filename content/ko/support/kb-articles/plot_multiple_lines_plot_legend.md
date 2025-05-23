---
title: How do I plot multiple lines on a plot with a legend?
menu:
  support:
    identifier: ko-support-kb-articles-plot_multiple_lines_plot_legend
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
---

`wandb.plot.line_series()`를 사용하여 여러 줄로 된 사용자 정의 차트를 만드세요. 라인 차트를 보려면 [프로젝트 페이지]({{< relref path="/guides/models/track/project-page.md" lang="ko" >}})로 이동하세요. 범례를 추가하려면 `wandb.plot.line_series()`에 `keys` 인수를 포함하세요. 예:

```python
wandb.log(
    {
        "my_plot": wandb.plot.line_series(
            xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
        )
    }
)
```

**여러 줄** 탭 아래의 [여기]({{< relref path="/guides/models/track/log/plots.md#basic-charts" lang="ko" >}})에서 여러 줄 플롯에 대한 추가 세부 정보를 참조하세요.
