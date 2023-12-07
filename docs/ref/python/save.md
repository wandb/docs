# save

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.1/wandb/sdk/wandb_run.py#L1822-L1852' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Ensure all files matching `glob_str` are synced to wandb with the policy specified.

```python
save(
    glob_str: Optional[str] = None,
    base_path: Optional[str] = None,
    policy: "PolicyName" = "live"
) -> Union[bool, List[str]]
```

| Arguments |  |
| :--- | :--- |
|  `glob_str` |  (string) a relative or absolute path to a unix glob or regular path. If this isn't specified the method is a noop. |
|  `base_path` |  (string) the base path to run the glob relative to |
|  `policy` |  (string) one of `live`, `now`, or `end` - live: upload the file as it changes, overwriting the previous version - now: upload the file once now - end: only upload file when the run ends |
