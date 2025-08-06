---
title: wandb 가 내 터미널이나 Jupyter 노트북 출력에 기록하지 않도록 하려면 어떻게 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-stop_wandb_writing_terminal_jupyter_notebook_output
support:
- 환경 변수
toc_hide: true
type: docs
url: /support/:filename
---

환경 변수 [`WANDB_SILENT`]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 `true`로 설정하세요.

{{< tabpane text=true langEqualsHeader=true >}}
  {{% tab header="Python" %}}
```python
os.environ["WANDB_SILENT"] = "true"
```
  {{% /tab %}}
  {{% tab "Notebook" %}}
```python
%env WANDB_SILENT=true
```
  {{% /tab %}}
  {{% tab "Command-Line" %}}
```shell
WANDB_SILENT=true
```
  {{% /tab %}}
{{< /tabpane >}}