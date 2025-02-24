---
title: Point aggregation
menu:
  default:
    identifier: ko-guides-models-app-features-panels-line-plot-sampling
    parent: line-plot
weight: 20
---

데이터 시각화 정확도와 성능을 향상시키기 위해 라인 플롯 내에서 포인트 집계 방식을 사용하세요. 포인트 집계 모드에는 [full fidelity]({{< relref path="#full-fidelity" lang="ko" >}}) 모드와 [random sampling]({{< relref path="#random-sampling" lang="ko" >}}) 모드의 두 가지 유형이 있습니다. W&B는 기본적으로 full fidelity 모드를 사용합니다.

## Full fidelity

full fidelity 모드를 사용하면 W&B는 데이터 포인트 수에 따라 x축을 동적 버킷으로 나눕니다. 그런 다음 각 버킷 내에서 최소값, 최대값, 평균값을 계산하여 라인 플롯에 대한 포인트 집계를 렌더링합니다.

full fidelity 모드를 포인트 집계에 사용할 때의 주요 이점은 다음과 같습니다.

* 극단값과 스파이크 유지: 데이터에서 극단값과 스파이크를 유지합니다.
* 최소점과 최대점 렌더링 방식 구성: W&B App을 사용하여 극단값(최소/최대)을 음영 영역으로 표시할지 여부를 대화식으로 결정합니다.
* 데이터 정확도 손실 없이 데이터 탐색: 특정 데이터 포인트를 확대하면 W&B가 x축 버킷 크기를 다시 계산합니다. 이를 통해 정확도를 잃지 않고 데이터를 탐색할 수 있습니다. 이전에 계산된 집계를 저장하는 데 캐싱이 사용되므로 로딩 시간을 줄일 수 있으며, 이는 대규모 데이터셋을 탐색할 때 특히 유용합니다.

### 최소점 및 최대점 렌더링 방식 구성

라인 플롯 주변에 음영 영역을 사용하여 최소값과 최대값을 표시하거나 숨깁니다.

다음 이미지는 파란색 라인 플롯을 보여줍니다. 밝은 파란색 음영 영역은 각 버킷의 최소값과 최대값을 나타냅니다.

{{< img src="/images/app_ui/shaded-areas.png" alt="" >}}

라인 플롯에서 최소값과 최대값을 렌더링하는 세 가지 방법이 있습니다.

* **Never**: 최소/최대값이 음영 영역으로 표시되지 않습니다. x축 버킷에서 집계된 라인만 표시합니다.
* **On hover**: 차트 위로 마우스를 가져가면 최소/최대값에 대한 음영 영역이 동적으로 나타납니다. 이 옵션은 보기를 깔끔하게 유지하면서 범위를 대화식으로 검사할 수 있도록 합니다.
* **Always**: 최소/최대 음영 영역이 차트의 모든 버킷에 대해 일관되게 표시되므로 항상 전체 값 범위를 시각화할 수 있습니다. 차트에 시각화된 Run이 많으면 시각적 노이즈가 발생할 수 있습니다.

기본적으로 최소값과 최대값은 음영 영역으로 표시되지 않습니다. 음영 영역 옵션 중 하나를 보려면 다음 단계를 따르세요.

{{< tabpane text=true >}}
{{% tab header="워크스페이스의 모든 차트" value="all_charts" %}}
1. W&B 프로젝트로 이동합니다.
2. 왼쪽 탭에서 **Workspace** 아이콘을 선택합니다.
3. 화면 오른쪽 상단의 톱니바퀴 아이콘을 **Add panels** 버튼 바로 왼쪽에 있는 것을 선택합니다.
4. 나타나는 UI 슬라이더에서 **Line plots**를 선택합니다.
5. **Point aggregation** 섹션 내에서 **Show min/max values as a shaded area** 드롭다운 메뉴에서 **On over** 또는 **Always**를 선택합니다.
{{% /tab %}}

{{% tab header="워크스페이스의 개별 차트" value="single_chart"%}}
1. W&B 프로젝트로 이동합니다.
2. 왼쪽 탭에서 **Workspace** 아이콘을 선택합니다.
3. full fidelity 모드를 활성화할 라인 플롯 패널을 선택합니다.
4. 나타나는 모달 내에서 **Show min/max values as a shaded area** 드롭다운 메뉴에서 **On hover** 또는 **Always**를 선택합니다.
{{% /tab %}}
{{< /tabpane >}}

### 데이터 정확도 손실 없이 데이터 탐색

극단값 또는 스파이크와 같은 중요한 포인트를 놓치지 않고 데이터셋의 특정 영역을 분석합니다. 라인 플롯을 확대하면 W&B는 각 버킷 내에서 최소값, 최대값 및 평균값을 계산하는 데 사용되는 버킷 크기를 조정합니다.

{{< img src="/images/app_ui/zoom_in.gif" alt="" >}}

W&B는 x축을 기본적으로 1000개의 버킷으로 동적으로 나눕니다. 각 버킷에 대해 W&B는 다음 값을 계산합니다.

- **Minimum**: 해당 버킷의 최저값입니다.
- **Maximum**: 해당 버킷의 최고값입니다.
- **Average**: 해당 버킷의 모든 포인트의 평균값입니다.

W&B는 전체 데이터 표현을 유지하고 모든 플롯에 극단값을 포함하는 방식으로 버킷의 값을 플롯합니다. 1,000개 이하의 포인트로 확대하면 full fidelity 모드는 추가 집계 없이 모든 데이터 포인트를 렌더링합니다.

라인 플롯을 확대하려면 다음 단계를 따르세요.

1. W&B 프로젝트로 이동합니다.
2. 왼쪽 탭에서 **Workspace** 아이콘을 선택합니다.
3. 필요에 따라 워크스페이스에 라인 플롯 패널을 추가하거나 기존 라인 플롯 패널로 이동합니다.
4. 클릭하고 드래그하여 확대할 특정 영역을 선택합니다.

{{% alert title="라인 플롯 그룹화 및 표현식" %}}
라인 플롯 그룹화를 사용하는 경우 W&B는 선택한 모드에 따라 다음을 적용합니다.

- **Non-windowed sampling (grouping)**: x축에서 Run 간에 포인트를 정렬합니다. 여러 포인트가 동일한 x값을 공유하는 경우 평균이 계산됩니다. 그렇지 않으면 개별 포인트로 표시됩니다.
- **Windowed sampling (grouping and expressions)**: x축을 250개의 버킷 또는 가장 긴 라인의 포인트 수(둘 중 더 작은 값)로 나눕니다. W&B는 각 버킷 내의 포인트 평균을 취합니다.
- **Full fidelity (grouping and expressions)**: non-windowed sampling과 유사하지만 성능과 세부 정보의 균형을 맞추기 위해 Run당 최대 500개의 포인트를 가져옵니다.
{{% /alert %}}

## Random sampling

Random sampling은 1500개의 임의로 샘플링된 포인트를 사용하여 라인 플롯을 렌더링합니다. Random sampling은 데이터 포인트 수가 많을 때 성능상의 이유로 유용합니다.

{{% alert color="warning" %}}
Random sampling은 비결정적으로 샘플링합니다. 즉, random sampling은 때때로 데이터에서 중요한 이상값이나 스파이크를 제외하므로 데이터 정확도가 떨어집니다.
{{% /alert %}}

### Random sampling 활성화
기본적으로 W&B는 full fidelity 모드를 사용합니다. Random sampling을 활성화하려면 다음 단계를 따르세요.

{{< tabpane text=true >}}
{{% tab header="워크스페이스의 모든 차트" value="all_charts" %}}
1. W&B 프로젝트로 이동합니다.
2. 왼쪽 탭에서 **Workspace** 아이콘을 선택합니다.
3. 화면 오른쪽 상단의 톱니바퀴 아이콘을 **Add panels** 버튼 바로 왼쪽에 있는 것을 선택합니다.
4. 나타나는 UI 슬라이더에서 **Line plots**를 선택합니다.
5. **Point aggregation** 섹션에서 **Random sampling**을 선택합니다.
{{% /tab %}}

{{% tab header="워크스페이스의 개별 차트" value="single_chart"%}}
1. W&B 프로젝트로 이동합니다.
2. 왼쪽 탭에서 **Workspace** 아이콘을 선택합니다.
3. random sampling을 활성화할 라인 플롯 패널을 선택합니다.
4. 나타나는 모달 내에서 **Point aggregation method** 섹션에서 **Random sampling**을 선택합니다.
{{% /tab %}}
{{< /tabpane >}}

### 샘플링되지 않은 데이터에 액세스

[W&B Run API]({{< relref path="/ref/python/public-api/run.md" lang="ko" >}})를 사용하여 Run 중에 기록된 메트릭의 전체 기록에 액세스할 수 있습니다. 다음 예제에서는 특정 Run에서 손실 값을 검색하고 처리하는 방법을 보여줍니다.

```python
# W&B API를 초기화합니다.
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")

# 'Loss' 메트릭의 기록을 검색합니다.
history = run.scan_history(keys=["Loss"])

# 기록에서 손실 값을 추출합니다.
losses = [row["Loss"] for row in history]
```
