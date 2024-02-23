
# 완료

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/wandb_run.py#L4108-L4119' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

실행을 완료로 표시하고 모든 데이터 업로드를 마칩니다.

```python
finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None
```

이것은 같은 프로세스에서 여러 실행을 생성할 때 사용됩니다.
스크립트가 종료될 때 이 메서드를 자동으로 호출합니다.

| 인수 |  |
| :--- | :--- |
|  `exit_code` |  실행을 실패로 표시하려면 0이 아닌 것으로 설정 |
|  `quiet` |  로그 출력을 최소화하려면 true로 설정 |