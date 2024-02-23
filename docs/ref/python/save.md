
# save

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L1830-L1860' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


`glob_str`과 일치하는 모든 파일이 지정된 정책에 따라 wandb에 동기화되도록 합니다.

```python
save(
    glob_str: Optional[str] = None,
    base_path: Optional[str] = None,
    policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```

| 인수 |  |
| :--- | :--- |
|  `glob_str` |  (문자열) 유닉스 글로브 또는 정규 경로에 대한 상대적 또는 절대적 경로입니다. 이것이 지정되지 않으면 메서드는 아무 작업도 하지 않습니다. |
|  `base_path` |  (문자열) 글로브를 실행하기 위한 기본 경로 |
|  `policy` |  (문자열) `live`, `now`, `end` 중 하나 - live: 파일이 변경될 때마다 이전 버전을 덮어쓰며 업로드 - now: 지금 단 한 번 파일을 업로드 - end: 실행이 끝날 때만 파일 업로드 |