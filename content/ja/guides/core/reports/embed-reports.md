---
title: レポートを埋め込む
description: W&B レポートを Notion に直接埋め込んだり、HTML の IFrame 要素で表示したりできます。
menu:
  default:
    identifier: embed-reports
    parent: reports
weight: 50
---

## HTML iframe 要素

レポートの右上隅にある **Share** ボタンを選択してください。モーダルウィンドウが表示されます。そこで **Copy embed code** を選択します。コピーされたコードは、Inline Frame (IFrame) の HTML 要素内でレンダリングされます。お好きな iframe HTML 要素にコピーしたコードを貼り付けてください。

{{% alert %}}
**Public** レポートのみ埋め込み時に表示できます。
{{% /alert %}}

{{< img src="/images/reports/get_embed_url.gif" alt="埋め込みコードを取得する" >}}

## Confluence

次のアニメーションでは、Confluence の IFrame セルにレポートの直接リンクを挿入する方法を示しています。

{{< img src="//images/reports/embed_iframe_confluence.gif" alt="Confluence への埋め込み" >}}

## Notion

次のアニメーションでは、Notion ドキュメントで Embed ブロックとレポートの埋め込みコードを使ってレポートを挿入する方法を示しています。

{{< img src="//images/reports/embed_iframe_notion.gif" alt="Notion への埋め込み" >}}

## Gradio

`gr.HTML` 要素を使用することで、W&B Reports を Gradio アプリ内に埋め込んだり、Hugging Face Spaces で利用したりできます。

```python
import gradio as gr

# レポートのURLを受け取り、iframeタグで埋め込む関数
def wandb_report(url):
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe)

# Gradio Blocks アプリとしてデモを起動
with gr.Blocks() as demo:
    report = wandb_report(
        "https://wandb.ai/_scott/pytorch-sweeps-demo/reports/loss-22-10-07-16-00-17---VmlldzoyNzU2NzAx"
    )
demo.launch()
```

##