---
title: save
menu:
  reference:
    identifier: ko-ref-python-save
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L1875-L1979 >}}

하나 이상의 파일을 W&B에 동기화합니다.

```python
save(
    glob_str: (str | os.PathLike | None) = None,
    base_path: (str | os.PathLike | None) = None,
    policy: PolicyName = "live"
) -> (bool | list[str])
```

상대 경로는 현재 작업 디렉토리를 기준으로 합니다.

"myfiles/*"와 같은 Unix glob은 `policy`와 관계없이 `save`가 호출될 때 확장됩니다. 특히 새 파일은 자동으로 선택되지 않습니다.

`base_path`는 업로드된 파일의 디렉토리 구조를 제어하기 위해 제공될 수 있습니다. 이는 `glob_str`의 접두사여야 하며 그 아래의 디렉토리 구조는 보존됩니다. 이는 다음 예제를 통해 가장 잘 이해할 수 있습니다.

```
wandb.save("these/are/myfiles/*")
# => run에서 "these/are/myfiles/" 폴더에 파일을 저장합니다.

wandb.save("these/are/myfiles/*", base_path="these")
# => run에서 "are/myfiles/" 폴더에 파일을 저장합니다.

wandb.save("/User/username/Documents/run123/*.txt")
# => run에서 "run123/" 폴더에 파일을 저장합니다. 아래 참고 사항을 참조하십시오.

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => run에서 "username/Documents/run123/" 폴더에 파일을 저장합니다.

wandb.save("files/*/saveme.txt")
# => 각 "saveme.txt" 파일을 "files/"의 적절한 하위 디렉토리에 저장합니다.
```

참고: 절대 경로 또는 glob이 제공되고 `base_path`가 없는 경우 위의 예와 같이 하나의 디렉토리 수준이 보존됩니다.

| Args |  |
| :--- | :--- |
|  `glob_str` | 상대 또는 절대 경로 또는 Unix glob입니다. |
|  `base_path` | 디렉토리 구조를 추론하는 데 사용할 경로입니다. 예제를 참조하십시오. |
|  `policy` | `live`, `now` 또는 `end` 중 하나입니다. * live: 파일이 변경됨에 따라 파일을 업로드하고 이전 버전을 덮어씁니다. * now: 파일을 지금 한 번 업로드합니다. * end: run이 종료될 때 파일을 업로드합니다. |

| Returns |  |
| :--- | :--- |
| 일치하는 파일에 대해 생성된 심볼릭 링크의 경로입니다. 기록상의 이유로 인해 레거시 코드에서 부울을 반환할 수 있습니다. |