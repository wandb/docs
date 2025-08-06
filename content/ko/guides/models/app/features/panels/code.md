---
title: 코드 저장 및 변경점 비교
menu:
  default:
    identifier: ko-guides-models-app-features-panels-code
    parent: panels
weight: 50
---

기본적으로 W&B는 최신 git 커밋 해시만 저장합니다. UI에서 실험 간 코드 비교를 동적으로 하려면 추가 코드 기능을 활성화할 수 있습니다.

`wandb` 버전 0.8.28부터, W&B는 `wandb.init()`이 호출된 메인 트레이닝 파일의 코드를 저장할 수 있습니다.

## 라이브러리 코드 저장하기

코드 저장을 활성화하면 W&B는 `wandb.init()`을 호출한 파일의 코드를 저장합니다. 추가적인 라이브러리 코드를 저장하려면 세 가지 방법이 있습니다:

### `wandb.init()` 이후에 `wandb.Run.log_code(".")` 호출

```python
import wandb

with wandb.init() as run:
  run.log_code(".")
```

### `code_dir`가 설정된 settings 오브젝트를 `wandb.init()`에 전달

```python
import wandb

wandb.init(settings=wandb.Settings(code_dir="."))
```

이렇게 하면 현재 디렉토리와 모든 하위 디렉토리 내의 모든 python 소스 코드 파일을 [artifact]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}})로 저장합니다. 저장되는 소스 코드 파일의 종류와 위치를 더 세밀하게 제어하려면 [참고 문서]({{< relref path="/ref/python/sdk/classes/run.md#log_code" lang="ko" >}})를 확인하세요.

### UI에서 코드 저장하기

코드 저장을 코드로 설정하는 것 외에도, W&B 계정의 Settings에서 이 기능을 토글할 수 있습니다. 이 설정은 계정에 연결된 모든 팀에 적용됩니다.

> 기본적으로 W&B는 모든 팀에 대해 코드 저장 기능을 비활성화합니다.

1. W&B 계정에 로그인합니다.
2. **Settings** > **Privacy**로 이동합니다.
3. **Project and content security** 항목 아래에서 **Disable default code saving**을 켭니다.

## 코드 비교 기능

다른 W&B run에서 사용한 코드를 비교할 수 있습니다:

1. 페이지 오른쪽 상단의 **Add panels** 버튼을 선택합니다.
2. **TEXT AND CODE** 드롭다운을 확장한 후 **Code**를 선택합니다.

{{< img src="/images/app_ui/code_comparer.png" alt="Code comparer panel" >}}

## Jupyter 세션 히스토리

W&B는 Jupyter 노트북 세션에서 실행된 코드의 히스토리를 저장합니다. Jupyter 안에서 **wandb.init()**을 호출하면, W&B가 훅을 추가하여 현재 세션에서 실행된 코드의 히스토리를 담은 Jupyter 노트북을 자동으로 저장합니다.

1. 코드가 있는 프로젝트 워크스페이스로 이동합니다.
2. 왼쪽 네비게이션 바에서 **Artifacts** 탭을 선택합니다.
3. **code** 아티팩트를 확장합니다.
4. **Files** 탭을 선택합니다.

{{< img src="/images/app_ui/jupyter_session_history.gif" alt="Jupyter session history" >}}

이 화면에는 세션에서 실행된 셀과 iPython의 display 메소드 호출로 생성된 출력이 함께 표시됩니다. 이를 통해 특정 run 내에서 Jupyter에서 실행된 코드를 정확하게 확인할 수 있습니다. 가능하다면, W&B는 가장 최근 버전의 노트북도 코드 디렉토리에서 함께 저장합니다.

{{< img src="/images/app_ui/jupyter_session_history_display.png" alt="Jupyter session output" >}}