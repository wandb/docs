---
description: Embed W&B reports directly into Notion or with an HTML IFrame element.
displayed_sidebar: default
---

# 리포트 내장하기

<head>
  <title>인기 있는 애플리케이션에 리포트 내장하기.</title>
</head>

### HTML iframe 요소

리포트 내에서 오른쪽 상단에 있는 **공유** 버튼을 선택합니다. 모달 창이 나타납니다. 모달 창 내에서 **내장 코드 복사**를 선택합니다. 복사된 코드는 Inline Frame (IFrame) HTML 요소 내에서 렌더링됩니다. 복사된 코드를 원하는 iframe HTML 요소에 붙여넣습니다.

_참고: 현재 내장된 리포트는 **공개**된 리포트만 볼 수 있습니다._

__

![](/images/reports/get_embed_url.gif)

### Confluence

다음 애니메이션은 Confluence 내 IFrame 셀에 리포트의 직접 링크를 삽입하는 방법을 보여줍니다.

![](//images/reports/embed_iframe_confluence.gif)

### Notion

다음 애니메이션은 Notion 문서에 리포트를 Notion의 Embed 블록과 리포트의 내장 코드를 사용하여 삽입하는 방법을 보여줍니다.

![](//images/reports/embed_iframe_notion.gif)

### Gradio

`gr.HTML` 요소를 사용하여 Gradio Apps 내에 W&B 리포트를 내장하고 Hugging Face Spaces 내에서 사용할 수 있습니다.

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

##