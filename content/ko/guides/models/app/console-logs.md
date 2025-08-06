---
title: 콘솔 로그
menu:
  default:
    identifier: ko-guides-models-app-console-logs
url: guides/app/console-logs
---

실험을 실행하면, 콘솔에 다양한 메시지가 출력되는 것을 볼 수 있습니다. W&B는 콘솔 로그를 캡처하여 W&B App에 표시합니다. 이 메시지들을 활용해 실험의 동작을 디버깅하고 모니터링할 수 있습니다.

## 콘솔 로그 보기

W&B App에서 run의 콘솔 로그를 확인하는 방법:

1. W&B App에서 자신의 프로젝트로 이동합니다.
2. **Runs** 테이블에서 원하는 run을 선택합니다.
3. 프로젝트 사이드바에서 **Logs** 탭을 클릭합니다.

{{% alert %}}
스토리지 제한으로 인해 최대 10,000줄의 로그만 표시됩니다.
{{% /alert %}}


## 콘솔 로그의 종류

W&B는 정보 메시지, 경고, 에러 등 다양한 유형의 콘솔 로그를 캡처하며, 각 로그의 심각도는 접두사로 구분됩니다.

### 정보 메시지
정보 메시지는 run의 진행 상황과 상태에 대한 업데이트를 제공합니다. 일반적으로 `wandb:`로 시작합니다.

```text
wandb: Starting Run: abc123
wandb: Run data is saved locally in ./wandb/run-20240125_120000-abc123
```

### 경고 메시지
실행을 멈추지는 않지만 잠재적인 이슈를 알리는 경고는 `WARNING:`으로 표시됩니다.

```text
WARNING Found .wandb file, not streaming tensorboard metrics.
WARNING These runs were logged with a previous version of wandb.
```

### 에러 메시지
심각한 문제를 알리는 에러 메시지는 `ERROR:`로 시작합니다. 이는 run이 정상적으로 완료되지 않을 수 있음을 의미합니다.

```text
ERROR Unable to save notebook session history.
ERROR Failed to save notebook.
```

## 콘솔 로그 설정

코드 내에서 `wandb.init()`에 `wandb.Settings` 오브젝트를 전달하여 W&B의 콘솔 로그 동작을 설정할 수 있습니다. `wandb.Settings` 내에서 아래 파라미터들을 지정하여 로그 동작을 제어할 수 있습니다:

- `show_errors`: `True`로 설정 시, 에러 메시지가 W&B App에 표시됩니다. `False`로 설정하면 표시되지 않습니다.
- `silent`: `True`로 설정 시, 모든 W&B 콘솔 출력이 숨겨집니다. 프로덕션 환경 등에서 콘솔 노이즈를 최소화하고 싶을 때 유용합니다.
- `show_warnings`: `True`로 설정 시, 경고 메시지가 W&B App에 표시됩니다. `False`로 하면 표시되지 않습니다.
- `show_info`: `True`로 설정 시, 정보 메시지가 W&B App에 표시됩니다. `False`로 하면 표시되지 않습니다.

아래는 이러한 설정을 적용하는 예시입니다:

```python
import wandb

settings = wandb.Settings(
    show_errors=True,  # 에러 메시지를 W&B App에 표시
    silent=False,      # 모든 W&B 콘솔 출력을 비활성화하지 않음
    show_warnings=True # 경고 메시지를 W&B App에 표시
)

with wandb.init(settings=settings) as run:
    # 여기에 트레이닝 코드를 작성하세요
    run.log({"accuracy": 0.95})
```

## 커스텀 로깅

W&B는 애플리케이션의 콘솔 로그를 캡처하지만, 사용자의 별도 로깅 설정에는 간섭하지 않습니다. Python의 내장 `print()` 함수나 `logging` 모듈을 자유롭게 사용할 수 있습니다.

```python
import wandb

with wandb.init(project="my-project") as run:
    for i in range(100, 1000, 100):
        # 이 코드는 W&B에도 log 되고 콘솔에도 출력됩니다.
        run.log({"epoch": i, "loss": 0.1 * i})
        print(f"epoch: {i} loss: {0.1 * i}")
```

콘솔 로그는 아래와 유사하게 출력됩니다:

```text
1 epoch:  100 loss: 1.3191105127334595
2 epoch:  200 loss: 0.8664389848709106
3 epoch:  300 loss: 0.6157898902893066
4 epoch:  400 loss: 0.4961796700954437
5 epoch:  500 loss: 0.42592573165893555
6 epoch:  600 loss: 0.3771176040172577
7 epoch:  700 loss: 0.3393910825252533
8 epoch:  800 loss: 0.3082585036754608
9 epoch:  900 loss: 0.28154927492141724
```

## 타임스탬프

모든 콘솔 로그에는 자동으로 타임스탬프가 추가됩니다. 이를 통해 각 로그 메시지가 생성된 시점을 확인할 수 있습니다.

콘솔 로그에서 타임스탬프 표시를 켜거나 끌 수 있습니다. 콘솔 페이지 상단 왼쪽의 **Timestamp visible** 드롭다운에서 표시 여부를 선택하세요.

## 콘솔 로그 검색

콘솔 로그 페이지 상단의 검색창을 이용해 키워드로 로그를 필터링할 수 있습니다. 특정 용어나 레이블, 에러 메시지를 검색할 수 있습니다.

## 커스텀 레이블로 필터링

{{% alert color="secondary"  %}}
`x_`로 시작하는 파라미터(예: `x_label`)는 공개 프리뷰 단계입니다. 의견이 있으시면 [W&B GitHub 저장소에 이슈를 남겨주세요](https://github.com/wandb/wandb).
{{% /alert %}}

콘솔 로그 페이지 상단의 UI 검색창에서 `wandb.Settings`의 `x_label` 인수로 전달한 레이블을 기반으로 로그를 필터링할 수 있습니다.

```python
import wandb

# 주요 노드에서 run을 초기화
run = wandb.init(
    entity="entity",
    project="project",
	settings=wandb.Settings(
        x_label="custom_label"  # (옵션) 로그 필터링을 위한 커스텀 레이블
        )
)
```

## 콘솔 로그 다운로드

W&B App에서 run의 콘솔 로그를 다운로드하는 방법:

1. W&B App에서 자신의 프로젝트로 이동합니다.
2. **Runs** 테이블에서 원하는 run을 선택합니다.
3. 프로젝트 사이드바에서 **Logs** 탭을 클릭합니다.
4. 콘솔 로그 페이지 오른쪽 상단의 다운로드 버튼을 클릭합니다.

## 콘솔 로그 복사

W&B App에서 run의 콘솔 로그를 복사하는 방법:

1. W&B App에서 자신의 프로젝트로 이동합니다.
2. **Runs** 테이블에서 원하는 run을 선택합니다.
3. 프로젝트 사이드바에서 **Logs** 탭을 클릭합니다.
4. 콘솔 로그 페이지 오른쪽 상단의 복사 버튼을 클릭합니다.