import wandb
import wandb_workspaces.reports.v2 as wr


## Psuedocode for testing various report features ##
# if testing is set to True:
# replace <entity> and <project> with test values
##

# :snippet-start: add-plots
report = wr.Report(
    project = "<project>",
    title="<title>",
    description="A descriptive description.",
)

blocks = [
    wr.PanelGrid(
        panels=[
            wr.LinePlot(x="time", y="velocity"),
            wr.ScatterPlot(x="time", y="acceleration"),
        ]
    )
]

report.blocks = blocks
report.save()
# :snippet-end: add-plots



# :snippet-start: add-runset-no-panels
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
# :snippet-end: add-runset-no-panels

# :snippet-start: add-runsets-and-panels
report = wr.Report(
    project = "<project>",
    title="An amazing title",
    description="A descriptive description.",
)

blocks = wr.PanelGrid(
    runsets=[
        wr.RunSet(project="<project>", entity="<entity>")
    ],
    panels=[
        wr.LinePlot(
            title="line title",
            x="x",
            y=["y"],
            range_x=[0, 100],
            range_y=[0, 100],
            log_x=True,
            log_y=True,
            title_x="x axis title",
            title_y="y axis title",
            ignore_outliers=True,
            groupby="hyperparam1",
            groupby_aggfunc="mean",
            groupby_rangefunc="minmax",
            smoothing_factor=0.5,
            smoothing_type="gaussian",
            smoothing_show_original=True,
            max_runs_to_show=10,
            plot_type="stacked-area",
            font_size="large",
            legend_position="west",
        ),
        wr.ScatterPlot(
            title="scatter title",
            x="y",
            y="y",
            # z='x',
            range_x=[0, 0.0005],
            range_y=[0, 0.0005],
            # range_z=[0,1],
            log_x=False,
            log_y=False,
            # log_z=True,
            running_ymin=True,
            running_ymean=True,
            running_ymax=True,
            font_size="small",
            regression=True,
        ),
    ],
    
)

report.blocks = [blocks]
report.save()
# :snippet-end: add-runsets-and-panels


entity = "<entity>"
project = "<project>"

for group in ["control", "experiment_a", "experiment_b"]:
    for i in range(3):
        with wandb.init(entity=entity, project=project, group=group, config={"group": group, "run": i}, name=f"{group}_run_{i}") as run:
            # Simulate some training
            for step in range(100):
                run.log({
                    "acc": 0.5 + (step / 100) * 0.3 + (i * 0.05),
                    "loss": 1.0 - (step / 100) * 0.5
                })

# :snippet-start: group-runs-config
# Create a report that groups runs by a config value
report = wr.Report(
  entity=entity,
  project=project,
  title="Grouped Runs Example",
)

# Create a runset that groups runs by the "group" config value
runset = wr.Runset(
  project=project,
  entity=entity,
  groupby=["config.group"] 
)
# Add the runset to a panel grid in the report
report.blocks = [
  wr.PanelGrid(
      runsets=[runset],
          )
      ]
# Save the report
report.save()
# :snippet-end: group-runs-config


##### Group runs by run metadata #####
entity = "<entity>"
project = "<project>"

# :snippet-start: group-runs-metadata
# Create a report that groups runs by their metadata (e.g., run name)
report = wr.Report(
  entity=entity,
  project=project,
  title="Grouped Runs by Metadata Example",
)

# Create a runset that groups runs by their name (metadata)
runset = wr.Runset(
  project=project,
  entity=entity,
  groupby=["Name"]  # Group by run names
)

# Add the runset to a panel grid in the report
report.blocks = [
  wr.PanelGrid(
      runsets=[runset],
          )
      ]
# Save the report
report.save()
# :snippet-end: group-runs-metadata
##### END #####


##### Group runs by summary metrics #####
entity = "<entity>"
project = "<project>"

for group in ["control", "experiment_a", "experiment_b"]:
    for i in range(3):
        with wandb.init(entity=entity, project=project, group=group, config={"group": group, "run": i}, name=f"{group}_run_{i}") as run:
            # Simulate some training
            for step in range(100):
                run.log({
                    "acc": 0.5 + (step / 100) * 0.3 + (i * 0.05),
                    "loss": 1.0 - (step / 100) * 0.5
                })

# :snippet-start: group-runs-summary-metrics
# Create a report that groups runs by a summary metric
report = wr.Report(
  entity=entity,
  project=project,
  title="Grouped Runs by Summary Metrics Example",
)

# Create a runset that groups runs by the "summary.acc" summary metric
runset = wr.Runset(
  project=project,
  entity=entity,
  groupby=["summary.acc"]  # Group by summary values 
)

# Add the runset to a panel grid in the report
report.blocks = [
  wr.PanelGrid(
      runsets=[runset],
          )
      ]
# Save the report
report.save()
# :snippet-end: group-runs-summary-metrics
##### END #####


##### Config filters #####
# :snippet-start: config-filters-0
config = {
    "learning_rate": 0.01,
    "batch_size": 32,
}

with wandb.init(project="<project>", entity="<entity>", config=config) as run:
    # Your training code here
    pass

# :snippet-end: config-filters-0

# :snippet-start: config-filters-1
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Config('learning_rate') > 0.01"
)
# :snippet-end: config-filters-1

# :snippet-start: config-filters-2
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Config('learning_rate') > 0.01 and Config('batch_size') == 32"
)
# :snippet-end: config-filters-2

# :snippet-start: config-filters-3
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
# :snippet-end: config-filters-3
##### END #####


##### Metric filters #####
# :snippet-start: metric-filters-0
with wandb.init(project="<project>", entity="<entity>") as run:
    for i in range(3):
        run.name = f"run{i+1}"
        # Your training code here
        pass
# :snippet-end: metric-filters-0

# :snippet-start: metric-filters-1
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Metric('displayName') in ['run1', 'run2', 'run3']"
)
# :snippet-end: metric-filters-1

# :snippet-start: metric-filters-2
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Metric('state') in ['finished']"
)
# :snippet-end: metric-filters-2

# :snippet-start: metric-filters-3
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Metric('state') not in ['crashed']"
)
# :snippet-end: metric-filters-3
##### END #####

##### Summary metric filters #####
# :snippet-start: summary-metric-filters
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
# :snippet-end: summary-metric-filters
##### END #####

# :snippet-start: tag-filters
runset = wr.Runset(
  entity="<entity>",
  project="<project>",
  filters="Tags('training') == 'training'"
)
# :snippet-end: tag-filters



###### Adding different block types to a report ######

# :snippet-start: add-code-blocks
report = wr.Report(project = "<project>")

report.blocks = [
    wr.CodeBlock(
        code=["this:", "- is", "- a", "cool:", "- yaml", "- file"], language="yaml"
    ),
    wr.CodeBlock(code=["Hello, World!"], language="python")
]

report.save()
# :snippet-end: add-code-blocks

# :snippet-start: add-markdown
report = wr.Report(project = "<project>")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
report.save()
# :snippet-end: add-markdown

# :snippet-start: add-html
report = wr.Report(project = "<project>")

report.blocks = [
    wr.H1(text="How Programmatic Reports work"),
    wr.H2(text="Heading 2"),
    wr.UnorderedList(items=["Bullet 1", "Bullet 2"]),
]

report.save()
# :snippet-end: add-html


# :snippet-start: embed-rich-media
report = wr.Report(project = "<project>")

report.blocks = [
    wr.Video(url="https://www.youtube.com/embed/6riDJMI-Y8U"),
    wr.Twitter(
        embed_html='<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The voice of an angel, truly. <a href="https://twitter.com/hashtag/MassEffect?src=hash&amp;ref_src=twsrc^tfw">#MassEffect</a> <a href="https://t.co/nMev97Uw7F">pic.twitter.com/nMev97Uw7F</a></p>&mdash; Mass Effect (@masseffect) <a href="https://twitter.com/masseffect/status/1428748886655569924?ref_src=twsrc^tfw">August 20, 2021</a></blockquote>\n'
    ),
]
report.save()
# :snippet-end: embed-rich-media

