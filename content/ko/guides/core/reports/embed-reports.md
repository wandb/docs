---
title: Embed a report
description: W&B 리포트 를 Notion에 직접 임베드하거나 HTML IFrame 엘리먼트를 사용할 수 있습니다.
menu:
  default:
    identifier: ko-guides-core-reports-embed-reports
    parent: reports
weight: 50
---

## HTML iframe 엘리먼트

리포트 내 우측 상단 모서리에 있는 **Share** 버튼을 선택하세요. 모달 창이 나타납니다. 모달 창에서 **Copy embed code**를 선택합니다. 복사된 코드는 인라인 프레임(IFrame) HTML 엘리먼트 내에서 렌더링됩니다. 복사된 코드를 원하는 iframe HTML 엘리먼트에 붙여넣으세요.

{{% alert %}}
**Public** 리포트만 임베드할 때 볼 수 있습니다.
{{% /alert %}}

{{< img src="/images/reports/get_embed_url.gif" alt="" >}}

## Confluence

다음 애니메이션은 Confluence의 IFrame 셀 내에 리포트에 대한 직접 링크를 삽입하는 방법을 보여줍니다.

{{< img src="//images/reports/embed_iframe_confluence.gif" alt="" >}}

## Notion

다음 애니메이션은 Notion의 Embed 블록과 리포트의 임베디드 코드를 사용하여 리포트를 Notion 문서에 삽입하는 방법을 보여줍니다.

{{< img src="//images/reports/embed_iframe_notion.gif" alt="" >}}

## Gradio

`gr.HTML` 엘리먼트를 사용하여 Gradio 앱 내에 W&B Reports 를 임베드하고 Hugging Face Spaces 내에서 사용할 수 있습니다.

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
