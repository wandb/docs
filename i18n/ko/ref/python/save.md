
# save

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L1830-L1860' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


`glob_str`과 일치하는 모든 파일이 지정된 정책으로 wandb에 동기화되도록 합니다.

```python
save(
    glob_str: Optional[str] = None,
    base_path: Optional[str] = None,
    policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```

| 인수 |  |
| :--- | :--- |
|  `glob_str` |  (string) 유닉스 글로브 또는 일반 경로로의 상대 또는 절대 경로입니다. 이것이 지정되지 않으면 메소드는 아무것도 하지 않습니다. |
|  `base_path` |  (string) 글로브를 상대적으로 실행할 기본 경로 |
|  `policy` |  (string) `live`, `now`, `end` 중 하나 - live: 파일이 변경될 때마다 이전 버전을 덮어쓰면서 업로드 - now: 지금 파일을 한 번 업로드 - end: run이 끝날 때만 파일 업로드 |