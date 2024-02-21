---
description: Embed W&B reports directly into Notion or with an HTML IFrame element.
displayed_sidebar: default
---

# 리포트 임베드하기

<head>
  <title>인기 있는 애플리케이션에 리포트 임베드하기.</title>
</head>

### HTML iframe 요소

리포트 내의 오른쪽 상단에 있는 **공유** 버튼을 선택합니다. 모달 창이 나타납니다. 모달 창 내에서 **임베드 코드 복사**를 선택합니다. 복사된 코드는 Inline Frame (IFrame) HTML 요소 내에서 렌더링됩니다. 복사한 코드를 원하는 iframe HTML 요소에 붙여넣으세요.

_참고: 현재로서는 **공개**된 리포트만 임베드했을 때 볼 수 있습니다._

__

![](/images/reports/get_embed_url.gif)

### 컨플루언스

다음 애니메이션은 컨플루언스에서 IFrame 셀 내에 리포트의 직접 링크를 삽입하는 방법을 보여줍니다.

![](//images/reports/embed_iframe_confluence.gif)

### 노션

다음 애니메이션은 노션 문서에 리포트를 임베드 블록을 사용하여 리포트의 임베드 코드와 함께 삽입하는 방법을 보여줍니다.

![](//images/reports/embed_iframe_notion.gif)

### 그레이디오

`gr.HTML` 요소를 사용하여 그레이디오 앱 내에 W&B 리포트를 임베드하고 Hugging Face Spaces 내에서 사용할 수 있습니다.

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