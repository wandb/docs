runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="SummaryMetric('accuracy') > 0.9"
)

runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Metric('state') in ['finished'] and SummaryMetric('train/train_loss') < 0.5"
)
