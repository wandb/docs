---
title: 리포트 임베드하기
description: W&B Reports를 Notion에 직접 임베드하거나 HTML IFrame 요소를 사용해 임베드할 수 있습니다.
menu:
  default:
    identifier: ko-guides-core-reports-embed-reports
    parent: reports
weight: 50
---

## HTML iframe 요소

리포트 내 오른쪽 상단에 있는 **Share** 버튼을 선택하세요. 모달 창이 나타납니다. 모달 창에서 **Copy embed code** 를 선택하세요. 복사된 코드는 Inline Frame (IFrame) HTML 요소 내에서 렌더링됩니다. 복사한 코드를 원하는 iframe HTML 요소에 붙여넣으세요.

{{% alert %}}
임베드로 볼 수 있는 리포트는 **public** 리포트만 가능합니다.
{{% /alert %}}

{{< img src="/images/reports/get_embed_url.gif" alt="Getting embed code" >}}

## Confluence

아래의 애니메이션은 Confluence의 IFrame 셀에 리포트의 직접 링크를 삽입하는 방법을 보여줍니다.

{{< img src="//images/reports/embed_iframe_confluence.gif" alt="Embedding in Confluence" >}}

## Notion

아래의 애니메이션은 Notion에서 Embed 블록과 리포트의 임베드 코드를 사용해 리포트를 문서에 삽입하는 방법을 보여줍니다.

{{< img src="//images/reports/embed_iframe_notion.gif" alt="Embedding in Notion" >}}

## Gradio

`gr.HTML` 요소를 사용해 W&B Reports 를 Gradio 앱에 임베드할 수 있으며, 이를 Hugging Face Spaces 내에서 사용할 수 있습니다.

```python
import gradio as gr


def wandb_report(url):
    # iframe 코드 생성
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe)


with gr.Blocks() as demo:
    report = wandb_report(
        "https://wandb.ai/_scott/pytorch-sweeps-demo/reports/loss-22-10-07-16-00-17---VmlldzoyNzU2NzAx"
    )
demo.launch()
```

##