---
title: 'setup()

  '
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-functions-setup
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_setup.py >}}




### <kbd>function</kbd> `setup`

```python
setup(settings: 'Settings | None' = None) → _WandbSetup
```

현재 프로세스와 자식 프로세스에서 사용할 W&B 를 준비합니다.

보통은 `wandb.init()` 이 내부적으로 호출하기 때문에 따로 신경 쓰지 않아도 됩니다.

여러 개의 프로세스에서 wandb 를 사용할 때, 부모 프로세스에서 자식 프로세스를 시작하기 전에 `wandb.setup()` 을 호출하면 성능과 리소스 활용이 향상될 수 있습니다.

`wandb.setup()` 은 `os.environ` 을 변경하므로, 자식 프로세스들이 수정된 환경 변수 값을 상속받는 것이 중요합니다.

`wandb.teardown()` 도 참고하세요.



**Args:**
 
 - `settings`:  전체적으로 적용할 설정 값입니다. 이후에 호출되는 `wandb.init()` 에서 덮어쓸 수 있습니다.



**Example:**
 ```python
import multiprocessing

import wandb


def run_experiment(params):
    with wandb.init(config=params):
         # 실험을 실행합니다
         pass


if __name__ == "__main__":
    # 백엔드를 시작하고 전역 설정을 지정합니다
    wandb.setup(settings={"project": "my_project"})

    # 실험 파라미터를 정의합니다
    experiment_params = [
         {"learning_rate": 0.01, "epochs": 10},
         {"learning_rate": 0.001, "epochs": 20},
    ]

    # 여러 프로세스를 시작하여 각각 별도의 실험을 실행합니다
    processes = []
    for params in experiment_params:
         p = multiprocessing.Process(target=run_experiment, args=(params,))
         p.start()
         processes.append(p)

    # 모든 프로세스가 끝날 때까지 기다립니다
    for p in processes:
         p.join()

    # 선택 사항: 백엔드를 명시적으로 종료합니다
    wandb.teardown()
```