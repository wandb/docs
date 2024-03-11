---
description: Create a W&B Report with the App UI or programmatically with the Weights
  & Biases SDK.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 리포트 생성하기

<head>
  <title>W&B 리포트 생성하기</title>
</head>

W&B App UI를 사용하여 상호작용적으로 리포트를 생성하거나 W&B Python SDK를 사용하여 프로그래매틱하게 생성하세요.

:::info
Python SDK를 사용하여 프로그래매틱하게 리포트를 생성하는 기능은 베타 버전이며 활발히 개발 중입니다. 예시를 보려면 이 [Google Colab을 확인하세요](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb).
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: '리포트 탭', value: 'reporttab'},
    {label: 'Python SDK (Beta)', value: 'sdk'},
  ]}>
  <TabItem value="app">

1. W&B App에서 프로젝트 워크스페이스로 이동합니다.
2. 워크스페이스의 오른쪽 상단에 있는 **리포트 생성**을 클릭하세요.

![](/images/reports/create_a_report_button.png)

3. 모달이 나타납니다. 시작할 차트를 선택하세요. 이후 리포트 인터페이스에서 차트를 추가하거나 삭제할 수 있습니다.

![](/images/reports/create_a_report_modal.png)

4. 새로운 run이 리포트에 추가되는 것을 방지하려면 **필터 run 세트** 옵션을 선택하세요. 이 옵션을 켜거나 끌 수 있습니다. **리포트 생성**을 클릭하면 리포트 탭에서 계속 작업할 수 있는 초안 리포트가 생성됩니다.


  </TabItem>
  <TabItem value="reporttab">

1. W&B App에서 프로젝트 워크스페이스로 이동합니다.
2. 프로젝트에서 **리포트** 탭(클립보드 이미지)을 선택하세요.
3. 리포트 페이지에서 **리포트 생성** 버튼을 선택하세요. 

![](/images/reports/create_report_button.png)
  </TabItem>
  <TabItem value="sdk">

`wandb` 라이브러리를 사용하여 프로그래매틱하게 리포트를 생성하세요.

```python
import wandb
import wandb.apis.reports as wr
```

Report Class Public API ([`wandb.apis.reports`](https://docs.wandb.ai/ref/python/public-api/api#reports))를 사용하여 리포트 인스턴스를 생성하세요. 프로젝트의 이름을 지정하세요.

```python
report = wr.Report(project="report_standard")
```

리포트는 `.save()` 메소드를 호출할 때까지 W&B 서버에 업로드되지 않습니다:

```python
report.save()
```

App UI나 프로그래매틱하게 리포트를 편집하는 방법에 대한 정보는 [리포트 편집하기](https://docs.wandb.ai/guides/reports/edit-a-report)를 참조하세요.
  </TabItem>
</Tabs>