with wandb.init(project="<project>", entity="<entity>") as run:
    for i in range(3):
        run.name = f"run{i+1}"
        # Your training code here
        pass
