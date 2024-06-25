---
description: W&B Reports を直接 Notion に埋め込むか、HTML の IFrame 要素を使用してください。
displayed_sidebar: default
---


# Embed reports

<head>
  <title>レポートを人気のアプリケーションに埋め込む</title>
</head>

### HTML iframe 要素

レポート内の右上にある **Share** ボタンを選択します。モーダルウィンドウが表示されます。モーダルウィンドウ内で **Copy embed code** を選択します。コピーしたコードは Inline Frame (IFrame) HTML 要素内にレンダリングされます。コピーしたコードをお好みの iframe HTML 要素に貼り付けてください。

_注: 現在、埋め込まれた場合に表示できるのは **public** レポートのみです。_

![](/images/reports/get_embed_url.gif)

### Confluence

以下のアニメーションは、IFrame セル内にレポートの直接リンクを挿入する方法を Confluence で示しています。

![](//images/reports/embed_iframe_confluence.gif)

### Notion

以下のアニメーションは、Notion の Embed ブロックを使用してレポートを Notion ドキュメントに挿入する方法を示しています。

![](//images/reports/embed_iframe_notion.gif)

### Gradio

`gr.HTML` 要素を使用して W&B Reports を Gradio アプリに埋め込み、Hugging Face Spaces 内で使用することができます。

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