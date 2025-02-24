---
title: Save and diff code
menu:
  default:
    identifier: ko-guides-models-app-features-panels-code
    parent: panels
weight: 50
---

기본적으로 W&B는 가장 최신 git 커밋 해시만 저장합니다. 더 많은 코드 기능을 켜서 UI에서 실험 간의 코드를 동적으로 비교할 수 있습니다.

`wandb` 버전 0.8.28부터 W&B는 `wandb.init()`을 호출하는 기본 트레이닝 파일의 코드를 저장할 수 있습니다.

## 라이브러리 코드 저장

코드 저장을 활성화하면 W&B는 `wandb.init()`을 호출한 파일의 코드를 저장합니다. 추가 라이브러리 코드를 저장하려면 다음 세 가지 옵션이 있습니다.

### `wandb.init()` 호출 후 `wandb.run.log_code(".")` 호출

```python
import wandb

wandb.init()
wandb.run.log_code(".")
```

### `code_dir`이 설정된 settings 오브젝트를 `wandb.init`에 전달

```python
import wandb

wandb.init(settings=wandb.Settings(code_dir="."))
```

이것은 현재 디렉토리와 모든 하위 디렉토리의 모든 Python 소스 코드 파일을 [artifact]({{< relref path="/ref/python/artifact.md" lang="ko" >}})로 캡처합니다. 저장되는 소스 코드 파일의 유형 및 위치를 더 세밀하게 제어하려면 [참조 문서]({{< relref path="/ref/python/run.md#log_code" lang="ko" >}})를 참조하십시오.

### UI에서 코드 저장 설정

프로그래밍 방식으로 코드 저장을 설정하는 것 외에도 W&B 계정 설정에서 이 기능을 전환할 수도 있습니다. 이는 계정과 연결된 모든 팀에 대해 코드 저장을 활성화합니다.

> 기본적으로 W&B는 모든 팀에 대해 코드 저장을 비활성화합니다.

1. W&B 계정에 로그인합니다.
2. **Settings** > **Privacy**로 이동합니다.
3. **Project and content security**에서 **Disable default code saving**을 켭니다.

## 코드 비교기
서로 다른 W&B Runs에서 사용된 코드를 비교합니다.

1. 페이지 오른쪽 상단에서 **Add panels** 버튼을 선택합니다.
2. **TEXT AND CODE** 드롭다운을 확장하고 **Code**를 선택합니다.

{{< img src="/images/app_ui/code_comparer.png" alt="" >}}

## Jupyter 세션 기록

W&B는 Jupyter 노트북 세션에서 실행된 코드 기록을 저장합니다. Jupyter 내에서 **wandb.init()**를 호출하면 W&B는 훅을 추가하여 현재 세션에서 실행된 코드 기록이 포함된 Jupyter 노트북을 자동으로 저장합니다.

1. 코드가 포함된 프로젝트 워크스페이스로 이동합니다.
2. 왼쪽 네비게이션 바에서 **Artifacts** 탭을 선택합니다.
3. **code** 아티팩트를 확장합니다.
4. **Files** 탭을 선택합니다.

{{< img src="/images/app_ui/jupyter_session_history.gif" alt="" >}}

이렇게 하면 iPython의 디스플레이 메소드를 호출하여 생성된 출력과 함께 세션에서 실행된 셀이 표시됩니다. 이를 통해 주어진 Run 내에서 Jupyter 내에서 실행된 코드를 정확히 볼 수 있습니다. 가능한 경우 W&B는 코드 디렉토리에서도 찾을 수 있는 노트북의 최신 버전도 저장합니다.

{{< img src="/images/app_ui/jupyter_session_history_display.png" alt="" >}}
