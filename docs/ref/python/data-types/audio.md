# Audio



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1042-L1186)



Wandb class for audio clips.

```python
Audio(
 data_or_path, sample_rate=None, caption=None
)
```





| Arguments | |
| :--- | :--- |
| `data_or_path` | (string or numpy array) A path to an audio file or a numpy array of audio data. |
| `sample_rate` | (int) Sample rate, required when passing in raw numpy array of audio data. |
| `caption` | (string) Caption to display with audio. |



## Methods

### `durations`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1144-L1146)

```python
@classmethod
durations(
 audio_list
)
```




### `resolve_ref`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1160-L1172)

```python
resolve_ref()
```




### `sample_rates`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1148-L1150)

```python
@classmethod
sample_rates(
 audio_list
)
```






