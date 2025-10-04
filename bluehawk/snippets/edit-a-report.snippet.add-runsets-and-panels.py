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
