---
title: Anonymous mode
description: 데이터를 로그하고 시각화하기, W&B 계정 없이도 가능합니다
displayed_sidebar: default
---

코드를 쉽게 실행할 수 있도록 게시하시겠습니까? 익명 모드를 사용하여 다른 사람이 W&B 계정을 먼저 생성하지 않고도 코드를 실행하고, W&B 대시보드를 보고, 결과를 시각화할 수 있습니다.

익명 모드에서 결과가 로그가 되도록 허용합니다:

```python
import wandb

wandb.init(anonymous="allow")
```

예를 들어, 다음 코드조각은 W&B를 사용하여 아티팩트를 생성하고 로그를 작성하는 방법을 보여줍니다:

```python
import wandb

wandb.require("core")

run = wandb.init(anonymous="allow")

artifact = wandb.Artifact(name="art1", type="foo")
artifact.add_file(local_path="path/to/file")
run.log_artifact(artifact)

run.finish()
```

[예제 노트북을 시도해보세요](http://bit.ly/anon-mode) 익명 모드가 어떻게 작동하는지 확인할 수 있습니다.

### 계정 없이 결과를 볼 수 있는 방법은?

누군가가 스크립트를 실행하고 `anonymous="allow"`를 설정한 경우에는:

1. **임시 계정 자동 생성:** W&B는 이미 로그인된 계정을 확인합니다. 계정이 없는 경우 W&B는 자동으로 새 익명 계정을 생성하고 해당 세션에 대해 API 키를 저장합니다.
2. **빠른 결과 로그:** 사용자는 스크립트를 실행하거나 다시 실행하여 자동으로 W&B 대시보드 UI에서 결과를 볼 수 있습니다. 이러한 미인지된 익명 run은 7일 동안 사용할 수 있습니다.
3. **데이터가 유용할 때 클레임:** 사용자가 W&B에서 가치 있는 결과를 찾으면 페이지 상단의 배너에 있는 버튼을 클릭하여 run 데이터를 실제 계정으로 쉽게 저장할 수 있습니다. run을 클레임하지 않으면 7일 후에 삭제됩니다.

:::caution
**익명 run 링크는 민감합니다**. 이러한 링크는 누구나 실험의 결과를 7일 동안 볼 수 있도록 하므로 신뢰할 수 있는 사람에게만 링크를 공유하십시오. 실행 결과를 공개적으로 공유하면서 작성자의 신원을 숨기려면 support@wandb.com에 문의하여 유스 케이스에 대해 더 알아보세요.
:::

### 기존 계정이 있는 사용자에게는 어떤 일이 발생합니까?

스크립트에서 `anonymous="allow"`를 설정하면 W&B는 익명 계정을 생성하기 전에 먼저 기존 계정이 없는지 확인합니다. 즉, W&B 사용자가 스크립트를 찾고 실행하면 결과가 정상적인 run처럼 자신의 계정에 정확하게 로그가 됩니다.

### 익명 사용자에게 사용 불가능한 기능은 무엇인가요?

*   **영구 데이터 없음:** run은 익명 계정에서 7일 동안만 저장됩니다. 실제 계정에 저장하여 익명 run 데이터를 클레임할 수 있습니다.

![](/images/app_ui/anon_mode_no_data.png)

*   **아티팩트 로그 불가능:** 익명 run에 아티팩트를 로그할 수 없다는 경고를 커맨드라인에 출력합니다:
    ```bash
    wandb: WARNING Artifacts logged anonymously cannot be claimed and expire after 7 days.
    ```

* **프로필 및 설정 페이지 없음:** 특정 페이지는 실제 계정에만 유용하므로 UI에서 사용할 수 없습니다.