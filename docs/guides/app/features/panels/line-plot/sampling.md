---
title: Use point aggregation
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

포인트 집계 메소드를 사용하여 선 그래프의 데이터 시각화 정확성과 성능을 향상하세요. 포인트 집계 모드는 두 가지 유형이 있습니다: [full fidelity](#full-fidelity)와 [random sampling](#random-sampling). W&B는 기본적으로 full fidelity 모드를 사용합니다.

## Full fidelity

full fidelity 모드를 사용할 때, W&B는 데이터 포인트 수에 따라 x축을 동적으로 버킷으로 나눕니다. 그런 다음 각 버킷의 최소값, 최대값, 평균값을 계산하여 선 그래프를 위한 포인트 집계를 렌더링합니다.

full fidelity 모드를 사용한 포인트 집계의 주요 장점 세 가지는 다음과 같습니다:

* 극단값 및 스파이크 보존: 데이터에서 극단값 및 스파이크를 유지합니다.
* 최소 및 최대 포인트 렌더링 구성: W&B 앱을 사용하여 극단값(최소/최대값)을 그늘진 영역으로 표시할지 여부를 대화형으로 결정합니다.
* 데이터 정확성을 잃지 않고 데이터를 탐색: 특정 데이터 포인트로 줌 인할 때 W&B는 x축 버킷 크기를 재계산합니다. 이를 통해 데이터 탐색 시 정확성을 유지할 수 있습니다. 이전에 계산된 집계를 저장하여 로드 시간을 줄이는 캐싱이 사용되며, 특히 대용량 데이터셋을 탐색할 때 유용합니다.

### 최소 및 최대 포인트 렌더링 구성

선 그래프 주위의 그늘진 영역으로 최소 및 최대 값을 표시하거나 숨길 수 있습니다.

다음 이미지는 파란색 선 그래프를 보여줍니다. 밝은 파란색 그늘진 영역은 각 버킷의 최소값 및 최대값을 나타냅니다.

![](/images/app_ui/shaded-areas.png)

선 그래프에서 최소값 및 최대값을 렌더링하는 방법은 세 가지가 있습니다:

* **Never**: 최소/최대 값이 그늘진 영역으로 표시되지 않습니다. x축 버킷에 대해 집계된 선만 표시됩니다.
* **On hover**: 그래프 위로 마우스를 가져가면 최소/최대 값의 그늘진 영역이 동적으로 나타납니다. 이 옵션은 뷰를 깔끔하게 유지하면서 범위를 대화형으로 검사할 수 있게 해줍니다.
* **Always**: 최소/최대 값의 그늘진 영역이 그래프의 모든 버킷에 대해 지속적으로 표시되어 항상 전체 값 범위를 시각화할 수 있게 해줍니다. 여러 run이 그래프에 시각화된 경우 시각적 노이즈가 생길 수 있습니다.

기본적으로 최소 및 최대 값은 그늘진 영역으로 표시되지 않습니다. 그늘진 영역 옵션 중 하나를 보려면 다음 단계를 따르세요:

<Tabs
  defaultValue="all_charts"
  values={[
    {label: 'All charts in  a workspace', value: 'all_charts'},
    {label: 'Individual chart in a workspace', value: 'single_chart'},
  ]}>
  <TabItem value="all_charts">

1. W&B 프로젝트로 이동
2. 왼쪽 탭에서 **Workspace** 아이콘 클릭
3. 화면 오른쪽 상단, **Add panels** 버튼 왼쪽의 톱니바퀴 아이콘 선택
4. 나타나는 UI 슬라이더에서 **Line plots** 선택
5. **Point aggregation** 섹션 내에서 **Show min/max values as a shaded area** 드롭다운 메뉴에서 **On over** 또는 **Always** 선택

  </TabItem>
  <TabItem value="single_chart">

1. W&B 프로젝트로 이동
2. 왼쪽 탭에서 **Workspace** 아이콘 클릭
3. full fidelity 모드를 활성화할 선 그래프 패널 선택
4. 나타나는 모달 내에서 **Show min/max values as a shaded area** 드롭다운 메뉴에서 **On hover** 또는 **Always** 선택

  </TabItem>
</Tabs>

### 데이터 정확성을 잃지 않고 데이터를 탐색

극단값이나 스파이크 같은 중요한 포인트를 놓치지 않고 데이터셋의 특정 영역을 분석하세요. 선 그래프를 확대할 때, W&B는 각 버킷 내 최소값, 최대값, 평균값을 계산하는 데 사용되는 버킷 크기를 조정합니다.

![](/images/app_ui/zoom_in.gif)

W&B는 기본적으로 x축을 동적으로 1000개의 버킷으로 나눕니다. 각 버킷에 대해 W&B는 다음 값을 계산합니다:

- **Minimum**: 해당 버킷 내 최저값.
- **Maximum**: 해당 버킷 내 최고값.
- **Average**: 해당 버킷 내 모든 포인트의 평균값.

W&B는 전체 데이터 표현을 유지하고 각 그래프에 극단값을 포함하는 방식으로 버킷 내 값을 플롯합니다. 1,000 포인트 이하로 줌인하면, full fidelity 모드는 추가 집계 없이 모든 데이터를 렌더링합니다.

선 그래프를 줌인하려면 다음 단계를 따르세요:

1. W&B 프로젝트로 이동
2. 왼쪽 탭에서 **Workspace** 아이콘 클릭
3. 워크스페이스에 선 그래프 패널을 추가하거나 기존의 선 그래프 패널로 이동 (선택 사항)
4. 확대할 특정 영역을 클릭하고 드래그하여 선택

:::info 선 그래프 그룹화 및 표현식
선 그래프 그룹화를 사용할 때, 선택된 모드에 따라 W&B는 다음을 적용합니다:

- **비창 기반 샘플링 (그룹화)**: x축에서 run 간 포인트를 정렬합니다. 여러 포인트가 동일한 x값을 공유하면 평균을 취하고, 그렇지 않으면 개별 포인트로 나타납니다.
- **창 기반 샘플링 (그룹화 및 표현식)**: x축을 250개의 버킷 또는 가장 긴 선의 포인트 수(작은 쪽)로 나눕니다. W&B는 각 버킷 내 포인트의 평균을 취합니다.
- **full fidelity (그룹화 및 표현식)**: 비창 기반 샘플링과 유사하지만, 성능과 세부사항을 균형 있게 유지하기 위해 run당 최대 500 포인트를 가져옵니다.
:::

## Random sampling

Random sampling은 1500개의 무작위 샘플링 포인트를 사용하여 선 그래프를 렌더링합니다. 많은 수의 데이터 포인트가 있을 때, 성능상의 이유로 Random sampling이 유용합니다.

:::warning
Random sampling은 비결정적으로 샘플링합니다. 이는 random sampling이 때로 중요한 외곽값이나 스파이크를 제외하여 데이터 정확성을 감소시킬 수 있음을 의미합니다.
:::

### Random sampling 활성화
기본적으로 W&B는 full fidelity 모드를 사용합니다. Random sampling을 활성화하려면 다음 단계를 따르세요:

<Tabs
  defaultValue="all_charts"
  values={[
    {label: 'All charts in a workspace', value: 'all_charts'},
    {label: 'Individual chart in a workspace', value: 'single_chart'},
  ]}>
  <TabItem value="all_charts">

1. W&B 프로젝트로 이동
2. 왼쪽 탭에서 **Workspace** 아이콘 클릭
3. 화면 오른쪽 상단, **Add panels** 버튼 왼쪽의 톱니바퀴 아이콘 선택
4. 나타나는 UI 슬라이더에서 **Line plots** 선택
5. **Point aggregation** 섹션에서 **Random sampling** 선택

  </TabItem>
  <TabItem value="single_chart">

1. W&B 프로젝트로 이동
2. 왼쪽 탭에서 **Workspace** 아이콘 클릭
3. Random sampling을 활성화할 선 그래프 패널 선택
4. 나타나는 모달 내 **Point aggregation method** 섹션에서 **Random sampling** 선택

  </TabItem>
</Tabs>

### 비샘플링 데이터 엑세스

[W&B Run API](../../../../../ref/python/public-api/run.md)를 사용하여 run 동안 기록된 메트릭의 전체 이력에 엑세스할 수 있습니다. 다음 예시는 특정 run에서 손실 값을 가져오고 처리하는 방법을 보여줍니다:

```python
# W&B API 초기화
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")

# 'Loss' 메트릭 이력 가져오기
history = run.scan_history(keys=["Loss"])

# 이력에서 손실 값 추출하기
losses = [row["Loss"] for row in history]
```