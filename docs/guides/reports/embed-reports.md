---
title: Embed a report
description: Notion에 W&B Reports를 직접 임베드하거나 HTML IFrame 요소를 사용하세요.
displayed_sidebar: default
---

### HTML iframe 요소

리포트 내 오른쪽 상단 모서리에 있는 **Share** 버튼을 선택하세요. 모달 창이 나타납니다. 모달 창에서 **Copy embed code**를 선택하세요. 복사된 코드는 HTML의 인라인 프레임 (IFrame) 요소 내에 표시됩니다. 원하는 iframe HTML 요소에 복사된 코드를 붙여넣으세요.

_참고: **공개** 리포트만 현재 임베드 시에 볼 수 있습니다._

__

![](/images/reports/get_embed_url.gif)

### Confluence

아래의 애니메이션은 Confluence의 IFrame 셀에 리포트의 직접 링크를 삽입하는 방법을 보여줍니다.

![](//images/reports/embed_iframe_confluence.gif)

### Notion

아래의 애니메이션은 Notion 문서에서 Embed 블록과 리포트의 임베드 코드를 사용하여 리포트를 삽입하는 방법을 보여줍니다.

![](//images/reports/embed_iframe_notion.gif)

### Gradio

`gr.HTML` 요소를 사용하여 W&B Reports를 Gradio Apps 내에 임베드하고 Hugging Face Spaces 내에서 사용할 수 있습니다.

```python
import gradio as gr


def wandb_report(url):
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe)


with gr.Blocks() as demo:
    report = wandb_report(
        "https://wandb.ai/_scott/pytorch-sweeps-demo/reports/loss-22-10-07-16-00-17---VmlldzoyNzU2NzAx"
    )
demo.launch()
```