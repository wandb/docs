import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# setup

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_setup.py'/>




### <kbd>function</kbd> `setup`

```python
setup(
    settings: Optional[ForwardRef('wandb.sdk.wandb_settings.Settings'), Dict[str, Any]] = None
) → Optional[ForwardRef('_WandbSetup')]
```

Prepares W&B for use in the current process and its children. 

You can usually ignore this as it is implicitly called by `wandb.init()`. 

When using wandb in multiple processes, calling `wandb.setup()` in the parent process before starting child processes may improve performance and resource utilization. 

Note that `wandb.setup()` modifies `os.environ`, and it is important that child processes inherit the modified environment variables. 

See also `wandb.teardown()`. 



**Args:**
 
 - `settings` (Optional[Union[Dict[str, Any], wandb.Settings]]):  Configuration settings  to apply globally. These can be overridden by subsequent `wandb.init()` calls. 



**Example:**
 ```python
    import multiprocessing

    import wandb


    def run_experiment(params):
         with wandb.init(config=params):
             # Run experiment
             pass


    if __name__ == "__main__":
         # Start backend and set global config
         wandb.setup(settings={"project": "my_project"})

         # Define experiment parameters
         experiment_params = [
             {"learning_rate": 0.01, "epochs": 10},
             {"learning_rate": 0.001, "epochs": 20},
         ]

         # Start multiple processes, each running a separate experiment
         processes = []
         for params in experiment_params:
             p = multiprocessing.Process(target=run_experiment, args=(params,))
             p.start()
             processes.append(p)

         # Wait for all processes to complete
         for p in processes:
             p.join()

         # Optional: Explicitly shut down the backend
         wandb.teardown()
    ```