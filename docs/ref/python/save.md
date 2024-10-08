# save

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L1935-L2041' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

하나 이상의 파일을 W&B에 동기화합니다.

```python
save(
    glob_str: Optional[Union[str, os.PathLike]] = None,
    base_path: Optional[Union[str, os.PathLike]] = None,
    policy: PolicyName = "live"
) -> Union[bool, List[str]]
```

상대 경로는 현재 작업 중인 디렉토리를 기준으로 합니다.

"myfiles/*"와 같은 Unix glob은 `save`가 호출될 때 확장되며, `policy`와 무관하게 작동합니다. 특히, 새로운 파일은 자동으로 감지되지 않습니다.

`base_path`를 제공하여 업로드되는 파일의 디렉토리 구조를 제어할 수 있습니다. 이는 `glob_str`의 접두사가 되어야 하며, 그 아래의 디렉토리 구조는 유지됩니다. 예제를 통해 이해할 수 있습니다:

```
wandb.save("these/are/myfiles/*")
# => files를 "these/are/myfiles/" 폴더에 저장합니다.

wandb.save("these/are/myfiles/*", base_path="these")
# => files를 "are/myfiles/" 폴더에 저장합니다.

wandb.save("/User/username/Documents/run123/*.txt")
# => run123/ 폴더에 files를 저장합니다. 아래 참고하세요.

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => username/Documents/run123/ 폴더에 files를 저장합니다.

wandb.save("files/*/saveme.txt")
# => 각 "saveme.txt" 파일을 "files/"의 적절한 하위 디렉토리에 저장합니다.
```

참고: 절대 경로나 glob을 제공하고 `base_path`가 없는 경우, 위 예제와 같이 하나의 디렉토리 레벨이 유지됩니다.

| 인수 |  |
| :--- | :--- |
|  `glob_str` |  상대 경로나 절대 경로 또는 Unix glob입니다. |
|  `base_path` |  디렉토리 구조를 유추하기 위한 경로입니다; 예제를 참조하세요. |
|  `policy` |  `live`, `now`, 또는 `end` 중 하나입니다. * live: 파일이 변경될 때 업로드하며, 이전 버전을 덮어씁니다 * now: 지금 바로 파일을 한 번 업로드합니다 * end: run이 끝날 때 파일을 업로드합니다 |

| 반환값 |  |
| :--- | :--- |
|  일치하는 파일에 대해 생성된 심볼릭 링크의 경로입니다. 과거 코드에서는 역사적인 이유로 boolean 값을 반환할 수 있습니다. |