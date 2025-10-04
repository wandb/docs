report = wr.Report(
  entity="<entity>",
  project="<project>",
)

report.blocks = [
  wr.PanelGrid(
      runsets=[runset]
  )
]

report.save()
