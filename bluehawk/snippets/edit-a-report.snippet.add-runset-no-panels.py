report = wr.Report(
    project = "<project>",
    title="An amazing title",
    description="A descriptive description.",
)

blocks = wr.PanelGrid(
    runsets=[
        wr.RunSet(project="<project>", entity="<entity>")
    ]
)

report.blocks = [blocks]
report.save()
