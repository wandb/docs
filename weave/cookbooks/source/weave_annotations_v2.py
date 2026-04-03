import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Weave Annotation Queue Report
    Kitchen-sink visual analytics examples for a single annotation queue.

    **Prerequisites:**
    ```
    pip install altair matplotlib numpy pandas requests vl-convert-python wandb weave wordcloud
    ```
    """)
    return


@app.cell
def _():
    # ─── Configuration ──────────────────────────────────────────────────────────
    WANDB_API_KEY = ""
    ENTITY_PROJECT = ""
    QUEUE_ID = ""
    DEDUP_STRATEGY = ""  # "latest" = keep last annotation per (call, scorer, annotator)
                               # "all"    = keep every annotation (duplicates counted separately)
                               # "mean"   = average duplicates into one value per (call, scorer, annotator)
    # ────────────────────────────────────────────────────────────────────────────
    return DEDUP_STRATEGY, ENTITY_PROJECT, QUEUE_ID, WANDB_API_KEY


@app.cell
def _():
    import os
    import wandb
    import weave
    import requests
    import pandas as pd
    import numpy as np
    import altair as alt
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from datetime import datetime, timezone
    from IPython.display import display, Markdown, HTML
    from weave.trace_server.trace_server_interface import (
        AnnotationQueuesQueryReq,
        AnnotationQueueItemsQueryReq,
        AnnotationQueuesStatsReq,
        AnnotationQueueReadReq,
        CallsFilter,
    )
    from weave.trace_server.interface.query import Query

    CALL_URL_TEMPLATE = "https://wandb.ai/{entity}/{project}/weave/calls/{call_id}"

    alt.data_transformers.disable_max_rows()
    alt.renderers.enable('png')
    return (
        AnnotationQueueItemsQueryReq,
        AnnotationQueueReadReq,
        AnnotationQueuesStatsReq,
        CALL_URL_TEMPLATE,
        CallsFilter,
        Markdown,
        Query,
        WordCloud,
        alt,
        display,
        pd,
        plt,
        requests,
        wandb,
        weave,
    )


@app.cell
def _(
    AnnotationQueueItemsQueryReq,
    AnnotationQueueReadReq,
    AnnotationQueuesStatsReq,
    CallsFilter,
    DEDUP_STRATEGY,
    ENTITY_PROJECT,
    QUEUE_ID,
    Query,
    WANDB_API_KEY,
    pd,
    requests,
    wandb,
    weave,
):
    def resolve_wb_user_ids(api_key, user_ids):
        """Resolve base64-encoded W&B user IDs to usernames."""
        query = 'query($id: ID) { user(id: $id) { username name email } }'
        mapping = {}
        for uid in set((uid for uid in user_ids if uid)):
            try:
                resp = requests.post('https://api.wandb.ai/graphql', auth=('api', api_key), json={'query': query, 'variables': {'id': uid}}, timeout=10)
                resp.raise_for_status()
                body = resp.json()
                if 'errors' in body:
                    mapping[uid] = uid
                    continue
                user_info = body.get('data', {}).get('user') or {}
                mapping[uid] = user_info.get('username') or user_info.get('name') or uid
            except Exception:
                mapping[uid] = uid
        return mapping
    wandb.login(key=WANDB_API_KEY)
    client = weave.init(ENTITY_PROJECT)
    project_id = f'{client.entity}/{client.project}'
    print('Loading queue data...')
    queue = client.server.annotation_queue_read(AnnotationQueueReadReq(project_id=project_id, queue_id=QUEUE_ID)).queue
    stats = client.server.annotation_queues_stats(AnnotationQueuesStatsReq(project_id=project_id, queue_ids=[queue.id])).stats[0]
    scorers = {}
    for _ref_uri in queue.scorer_refs:
        scorers[_ref_uri] = weave.ref(_ref_uri).get()
    items_res = client.server.annotation_queue_items_query(AnnotationQueueItemsQueryReq(project_id=project_id, queue_id=queue.id, limit=10000))
    items_df = pd.DataFrame([{'item_id': i.id, 'call_id': i.call_id, 'op_name': i.call_op_name, 'trace_id': i.call_trace_id, 'state': i.annotation_state, 'annotator': i.annotator_user_id, 'added_at': i.created_at, 'updated_at': i.updated_at} for i in items_res.items])
    feedback_raw = list(client.get_feedback(query=Query(**{'$expr': {'$eq': [{'$getField': 'queue_id'}, {'$literal': queue.id}]}}), limit=10000))
    _feedback_columns = ['feedback_id', 'call_id', 'annotation_ref', 'scorer_name', 'scorer_type', 'value', 'annotator', 'created_at']
    if feedback_raw:
        feedback_df = pd.DataFrame([{'feedback_id': f.id, 'call_id': f.weave_ref.split('/')[-1] if f.weave_ref else None, 'annotation_ref': f.annotation_ref, 'scorer_name': scorers[f.annotation_ref].name if f.annotation_ref and f.annotation_ref in scorers else None, 'scorer_type': (scorers[f.annotation_ref].field_schema or {}).get('type') if f.annotation_ref and f.annotation_ref in scorers else None, 'value': f.payload.get('value') if f.payload else None, 'annotator': f.wb_user_id, 'created_at': pd.to_datetime(f.created_at, utc=True)} for f in feedback_raw])
    else:
        feedback_df = pd.DataFrame(columns=_feedback_columns)
    all_user_ids = set()
    if not items_df.empty:
        all_user_ids.update(items_df['annotator'].dropna().unique())
    if not feedback_df.empty:
        all_user_ids.update(feedback_df['annotator'].dropna().unique())
    USER_MAP = {}
    if all_user_ids:
        USER_MAP = resolve_wb_user_ids(WANDB_API_KEY, all_user_ids)
        if not items_df.empty:
            items_df['annotator'] = items_df['annotator'].map(lambda x: USER_MAP.get(x, x))
        if not feedback_df.empty:
            feedback_df['annotator'] = feedback_df['annotator'].map(lambda x: USER_MAP.get(x, x))
    if DEDUP_STRATEGY == 'latest' and (not feedback_df.empty):
        feedback_df = feedback_df.sort_values('created_at').groupby(['call_id', 'annotation_ref', 'annotator'], as_index=False).last()
    elif DEDUP_STRATEGY == 'mean' and (not feedback_df.empty):

        def _agg_value(vals):
            numeric = pd.to_numeric(_vals, errors='coerce')
            if numeric.notna().all():
                return numeric.mean()
            return _vals.iloc[-1]
        feedback_df = feedback_df.sort_values('created_at').groupby(['call_id', 'annotation_ref', 'annotator'], as_index=False).agg({'feedback_id': 'last', 'scorer_name': 'first', 'scorer_type': 'first', 'value': _agg_value, 'created_at': 'max'})
    call_ids = items_df['call_id'].unique().tolist() if not items_df.empty else []
    call_model_map = {}
    call_op_map = {}
    if call_ids:
        calls = client.get_calls(filter=CallsFilter(call_ids=call_ids), columns=['id', 'inputs', 'output', 'summary', 'op_name'], include_costs=True, limit=len(call_ids))
        for call in calls:
            model = None
            if hasattr(call, 'inputs') and isinstance(call.inputs, dict):
                model = call.inputs.get('model') or call.inputs.get('model_name') or call.inputs.get('model_id')
            if model is None and hasattr(call, 'output') and isinstance(call.output, dict):
                model = call.output.get('model')
            if model is None and hasattr(call, 'summary') and isinstance(call.summary, dict):
                usage = call.summary.get('usage') or {}
                if isinstance(usage, dict) and len(usage) == 1:
                    model = list(usage.keys())[0]
                weave_info = call.summary.get('weave') or {}
                costs = weave_info.get('costs') or {}
                if isinstance(costs, dict) and len(costs) == 1 and (model is None):
                    model = list(costs.keys())[0]
            call_model_map[call.id] = model or '(unknown)'
            op = getattr(call, 'op_name', None) or '(unknown)'
            call_op_map[call.id] = op
    model_df = pd.DataFrame([{'call_id': cid, 'model': m} for cid, m in call_model_map.items()])
    op_df = pd.DataFrame([{'call_id': cid, 'op_name_full': op} for cid, op in call_op_map.items()])
    # Resolve annotator IDs
    # Deduplication
    # Fetch call data (model + op) upfront
    print(f'  {len(items_df)} items, {len(feedback_df)} feedback entries, {len(scorers)} scorer(s), {len(call_model_map)} calls fetched')
    return (
        call_model_map,
        call_op_map,
        client,
        feedback_df,
        items_df,
        model_df,
        op_df,
        queue,
        scorers,
        stats,
    )


@app.cell
def _(CALL_URL_TEMPLATE, client, feedback_df, model_df, op_df, scorers):
    def scorer_type(ref_uri):
        """Classify a scorer as boolean, integer, number, string_enum, or string_free."""
        _spec = scorers.get(_ref_uri)
        if _spec is None:
            return 'unknown'
        schema = _spec.field_schema or {}
        _stype = schema.get('type', 'string')
        if _stype == 'boolean':
            return 'boolean'
        if _stype == 'integer':
            return 'integer'
        if _stype == 'number':
            return 'number'
        if 'enum' in schema:
            return 'string_enum'
        if _stype == 'string':
            _vals = feedback_df.loc[feedback_df['annotation_ref'] == _ref_uri, 'value'].dropna().unique()
            if len(_vals) <= 10:
                return 'string_enum'
            return 'string_free'
        return 'unknown'

    def call_url(call_id):
        return CALL_URL_TEMPLATE.format(entity=client.entity, project=client.project, call_id=call_id)

    def short_op(op_name):
        if not op_name:
            return '(unknown)'
        return str(op_name).split('/')[-1].split(':')[-1]
    CHART_WIDTH = 600
    CHART_HEIGHT = 350
    STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'because', 'about', 'its', 'it', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'if', 'while', 'up', 'don', 'doesn', 'didn', 'isn', 'aren', 'wasn', 'weren', 'won', 'wouldn', 'couldn', 'shouldn'}

    def _top_words(texts, n=10):
        """Count word frequencies from an iterable of strings, excluding stopwords."""
        freq = {}
        for t in texts:
            for _w in str(t).lower().split():
                _w = _w.strip('.,!?;:"\'()[]')
                if len(_w) > 2 and _w not in STOPWORDS:
                    freq[_w] = freq.get(_w, 0) + 1
        return sorted(freq.items(), key=lambda x: -x[1])[:n]
    scorer_ref_to_type = {ref: scorer_type(ref) for ref in scorers}
    enriched_df = feedback_df.copy()
    if not enriched_df.empty:
        if not model_df.empty:
            enriched_df = enriched_df.merge(model_df, on='call_id', how='left')
            enriched_df['model'] = enriched_df['model'].fillna('(unknown)')
        else:
            enriched_df['model'] = '(unknown)'
        if not op_df.empty:
            enriched_df = enriched_df.merge(op_df, on='call_id', how='left')
            enriched_df['op'] = enriched_df['op_name_full'].apply(short_op)
        else:
    # Build an enriched feedback DataFrame with op + model joined in
            enriched_df['op'] = '(unknown)'
    return (
        CHART_HEIGHT,
        CHART_WIDTH,
        call_url,
        enriched_df,
        scorer_ref_to_type,
        short_op,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Queue
    Queue metadata and scoring configuration at a glance.
    """)
    return


@app.cell
def _(display, pd, queue, scorers, stats):
    pct_complete = stats.completed_items / stats.total_items * 100 if stats.total_items else 0
    queue_info = pd.DataFrame([
        {"Field": "Name", "Value": queue.name},
        {"Field": "Description", "Value": queue.description or "(none)"},
        {"Field": "Created", "Value": str(queue.created_at)},
        {"Field": "Total Items", "Value": str(stats.total_items)},
        {"Field": "Completed", "Value": f"{stats.completed_items} ({pct_complete:.1f}%)"},
    ] + [
        {"Field": f"Scorer: {s.name}", "Value": (s.field_schema or {}).get("type", "?")}
        for s in scorers.values()
    ])
    display(queue_info)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Item Completion
    Shows what fraction of queue items have received at least one annotation across any scorer.
    """)
    return


@app.cell
def _(
    CHART_HEIGHT,
    CHART_WIDTH,
    Markdown,
    alt,
    display,
    feedback_df,
    items_df,
    pd,
):
    total_items = len(items_df)
    annotated_call_ids = set(feedback_df['call_id'].unique()) if not feedback_df.empty else set()
    annotated_count = len(set(items_df['call_id']) & annotated_call_ids) if not items_df.empty else 0
    unannotated_count = total_items - annotated_count
    _pct = annotated_count / total_items * 100 if total_items else 0
    display(Markdown(f'**{annotated_count}** of **{total_items}** items have been annotated (**{_pct:.1f}%**)'))
    completion_data = pd.DataFrame([{'Status': 'Annotated', 'Count': annotated_count}, {'Status': 'Unannotated', 'Count': unannotated_count}])
    _pie = alt.Chart(completion_data).mark_arc(innerRadius=50).encode(theta=alt.Theta('Count:Q'), color=alt.Color('Status:N'), tooltip=['Status', 'Count']).properties(title='Annotation Completion', width=CHART_WIDTH, height=CHART_HEIGHT)
    display(_pie)
    return (total_items,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Participation
    Annotator activity summary, daily annotation volume, and per-scorer coverage.
    The summary table shows each annotator's total annotations, unique calls reviewed,
    active days, and throughput. The line chart tracks daily annotation volume,
    and the bar chart shows what percentage of queue items each scorer has covered.
    """)
    return


@app.cell
def _(
    CHART_HEIGHT,
    CHART_WIDTH,
    Markdown,
    alt,
    display,
    feedback_df,
    pd,
    scorers,
    total_items,
):
    if not feedback_df.empty:
        _rows = []  # Annotator summary table
        for annotator, grp in feedback_df.groupby('annotator'):
            dates = grp['created_at'].dt.date
            active_days = dates.nunique()
            _rows.append({'Annotator': annotator or '(unknown)', 'Annotations': len(grp), 'Unique Calls': grp['call_id'].nunique(), 'Active Days': active_days, 'Ann / Active Day': round(len(grp) / max(active_days, 1), 1), 'First': grp['created_at'].min().strftime('%Y-%m-%d'), 'Last': grp['created_at'].max().strftime('%Y-%m-%d')})
        summary_table = pd.DataFrame(_rows).sort_values('Annotations', ascending=False)
        display(Markdown('### Annotator Summary'))
        display(summary_table)
        fb_daily = feedback_df.copy()
        fb_daily['date'] = fb_daily['created_at'].dt.date
        daily_counts = fb_daily.groupby('date').size().reset_index(name='Annotations')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        _line = alt.Chart(daily_counts).mark_line(point=True).encode(x=alt.X('date:T', title='Date'), y=alt.Y('Annotations:Q', title='Annotations'), tooltip=['date:T', 'Annotations:Q']).properties(title='Annotations Per Day', width=CHART_WIDTH, height=CHART_HEIGHT)
        display(_line)
        coverage_rows = []
        for _ref_uri, _spec in scorers.items():
            scorer_fb = feedback_df[feedback_df['annotation_ref'] == _ref_uri]
            covered = scorer_fb['call_id'].nunique()
            _pct = covered / total_items * 100 if total_items else 0  # Annotations per day
            coverage_rows.append({'Scorer': _spec.name, 'Coverage %': round(_pct, 1)})
        coverage_df = pd.DataFrame(coverage_rows)
        _bar = alt.Chart(coverage_df).mark_bar().encode(x=alt.X('Scorer:N', title='Scorer', sort='-y'), y=alt.Y('Coverage %:Q', title='Coverage %', scale=alt.Scale(domain=[0, 100])), color=alt.Color('Scorer:N', legend=None), tooltip=['Scorer', 'Coverage %']).properties(title='Annotation Coverage Per Scorer', width=CHART_WIDTH, height=CHART_HEIGHT)
        display(_bar)
    else:
        display(Markdown('*No feedback data yet.*'))  # Coverage per scorer
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scores
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Score Distributions
    Distribution of values for each scorer. Boolean scorers are shown as pie charts,
    numeric (integer/number) as histograms, string enums (<10 unique values) as bar charts,
    and free-text strings (>10 unique values) as word clouds.
    """)
    return


@app.cell
def _(
    CHART_HEIGHT,
    CHART_WIDTH,
    Markdown,
    WordCloud,
    alt,
    display,
    feedback_df,
    pd,
    plt,
    scorer_ref_to_type,
    scorers,
):
    if not feedback_df.empty:
        for _ref_uri, _spec in scorers.items():
            _stype = scorer_ref_to_type[_ref_uri]
            _group = feedback_df[feedback_df['annotation_ref'] == _ref_uri]
            _values = _group['value'].dropna()
            if _values.empty:
                continue
            display(Markdown(f'### {_spec.name} (`{_stype}`, n={len(_values)})'))
            if _stype == 'boolean':
                bool_counts = _values.value_counts().reset_index()
                bool_counts.columns = ['Value', 'Count']
                bool_counts['Value'] = bool_counts['Value'].astype(str)
                _pie = alt.Chart(bool_counts).mark_arc(innerRadius=50).encode(theta=alt.Theta('Count:Q'), color=alt.Color('Value:N'), tooltip=['Value', 'Count']).properties(title=f'{_spec.name} Distribution', width=CHART_WIDTH, height=CHART_HEIGHT)
                display(_pie)
            elif _stype in ('integer', 'number'):
                nums = pd.to_numeric(_values, errors='coerce').dropna()
                if nums.empty:
                    continue
                hist_df = nums.reset_index(drop=True).to_frame(name='Score')
                if _stype == 'integer':
                    _bar = alt.Chart(hist_df).mark_bar().encode(x=alt.X('Score:O', title='Score'), y=alt.Y('count():Q', title='Count'), tooltip=['Score:O', 'count():Q']).properties(title=f'{_spec.name} Distribution', width=CHART_WIDTH, height=CHART_HEIGHT)
                else:
                    _bar = alt.Chart(hist_df).mark_bar().encode(x=alt.X('Score:Q', bin=alt.Bin(maxbins=20), title='Score'), y=alt.Y('count():Q', title='Count'), tooltip=['count():Q']).properties(title=f'{_spec.name} Distribution', width=CHART_WIDTH, height=CHART_HEIGHT)
                display(_bar)
            elif _stype == 'string_enum':
                val_counts = _values.value_counts().reset_index()
                val_counts.columns = ['Value', 'Count']
                _bar = alt.Chart(val_counts).mark_bar().encode(x=alt.X('Value:N', title='Value', sort='-y'), y=alt.Y('Count:Q', title='Count'), color=alt.Color('Value:N', legend=None), tooltip=['Value', 'Count']).properties(title=f'{_spec.name} Distribution', width=CHART_WIDTH, height=CHART_HEIGHT)
                display(_bar)
            elif _stype == 'string_free':
                _text = ' '.join((str(v) for v in _values))
                _wc = WordCloud(width=800, height=400, background_color='white').generate(_text)
                _fig, _ax = plt.subplots(figsize=(10, 5))
                _ax.imshow(_wc, interpolation='bilinear')
                _ax.axis('off')
                _ax.set_title(f'{_spec.name} Word Cloud')
                plt.tight_layout()
                display(_fig)
                plt.close(_fig)
                top = _top_words(_values)
                if top:
                    display(Markdown(f'**Top 10 words:**'))
                    tw_df = pd.DataFrame(top, columns=['Word', 'Count'])
                    tw_df.insert(0, '#', range(1, len(tw_df) + 1))
                    display(tw_df)
    else:
        display(Markdown('*No feedback data yet.*'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scores Analysis
    Detailed score analysis per scorer. For numeric scorers, a scatterplot shows
    each annotation colored by annotator, plus tables of top-5 and bottom-5 calls
    (requiring >= 2 annotations). For boolean/enum scorers, top-5 calls per bucket
    are shown. All tables include a URL linking back to the call in W&B.
    """)
    return


@app.cell
def _(
    CHART_HEIGHT,
    CHART_WIDTH,
    Markdown,
    alt,
    call_model_map,
    call_op_map,
    call_url,
    display,
    enriched_df,
    pd,
    scorer_ref_to_type,
    scorers,
    short_op,
):
    def _make_call_table(call_ids_list):
        """Build a DataFrame with Call ID, Op, Model, URL for a list of call IDs."""
        _rows = []
        for cid in call_ids_list:
            _rows.append({'Call ID': cid, 'Op': short_op(call_op_map.get(cid, '(unknown)')), 'Model': call_model_map.get(cid, '(unknown)'), 'URL': call_url(cid)})
        return pd.DataFrame(_rows)
    if not enriched_df.empty:
        for _ref_uri, _spec in scorers.items():
            _stype = scorer_ref_to_type[_ref_uri]
            _group = enriched_df[enriched_df['annotation_ref'] == _ref_uri].copy()
            if _group.empty:
                continue
            display(Markdown(f'### {_spec.name}'))
            if _stype in ('integer', 'number'):
                _group['num'] = pd.to_numeric(_group['value'], errors='coerce')
                _scored = _group.dropna(subset=['num'])
                if _scored.empty:
                    continue
                y_enc = alt.Y('num:Q', title='Score')
                if _stype == 'integer':
                    schema = _spec.field_schema or {}
                    y_min = schema.get('minimum', int(_scored['num'].min()))
                    y_max = schema.get('maximum', int(_scored['num'].max()))
                    y_enc = alt.Y('num:Q', title='Score', scale=alt.Scale(domain=[y_min, y_max]), axis=alt.Axis(tickMinStep=1))
                scatter = alt.Chart(_scored.reset_index()).mark_circle(size=60).encode(x=alt.X('index:Q', title='Annotation Index'), y=y_enc, color=alt.Color('annotator:N', title='Annotator'), tooltip=['call_id', 'annotator', 'num']).properties(title=f'{_spec.name} — Scores by Annotator', width=CHART_WIDTH, height=CHART_HEIGHT)
                display(scatter)
                _per_call = _scored.groupby('call_id').agg(avg_score=('num', 'mean'), num_annotations=('num', 'count')).reset_index()
                multi_ann = _per_call[_per_call['num_annotations'] >= 2]
                if not multi_ann.empty:
                    top5 = multi_ann.nlargest(5, 'avg_score')
                    display(Markdown(f'**Top 5 (avg score, >= 2 annotations):**'))
                    display(_make_call_table(top5['call_id'].tolist()))
                    bottom5 = multi_ann.nsmallest(5, 'avg_score')
                    display(Markdown(f'**Bottom 5 (avg score, >= 2 annotations):**'))
                    display(_make_call_table(bottom5['call_id'].tolist()))
                else:
                    display(Markdown('*Not enough calls with >= 2 annotations for top/bottom ranking.*'))
            elif _stype in ('boolean', 'string_enum'):
                _values = _group['value'].dropna()
                unique_vals = _values.unique()
                for val in unique_vals:
                    val_group = _group[_group['value'] == val]
                    top_calls = val_group.groupby('call_id').size().reset_index(name='count').nlargest(5, 'count')
                    display(Markdown(f'**Top 5 calls — {_spec.name} = `{val}`:**'))
                    display(_make_call_table(top_calls['call_id'].tolist()))
    else:
        display(Markdown('*No feedback data yet.*'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scores: Trend Over Time
    For numeric scorers, the daily mean score is shown as a bar chart with
    standard deviation error bars. For string enum scorers, a multi-line chart
    tracks the daily count of each value over time.
    """)
    return


@app.cell
def _(
    CHART_HEIGHT,
    CHART_WIDTH,
    Markdown,
    alt,
    display,
    feedback_df,
    pd,
    scorer_ref_to_type,
    scorers,
):
    if not feedback_df.empty:
        for _ref_uri, _spec in scorers.items():
            _stype = scorer_ref_to_type[_ref_uri]
            _group = feedback_df[feedback_df['annotation_ref'] == _ref_uri].copy()
            if _group.empty:
                continue
            _group['date'] = _group['created_at'].dt.floor('D')
            if _stype in ('integer', 'number'):
                _group['num'] = pd.to_numeric(_group['value'], errors='coerce')
                _scored = _group.dropna(subset=['num'])
                if _scored.empty:
                    continue
                daily = _scored.groupby('date')['num'].agg(['mean', 'std', 'count']).reset_index()
                daily.columns = ['date', 'mean', 'std', 'count']
                daily['std'] = daily['std'].fillna(0)
                daily['upper'] = daily['mean'] + daily['std']
                daily['lower'] = daily['mean'] - daily['std']
                _bars = alt.Chart(daily).mark_bar().encode(x=alt.X('date:T', title='Date'), y=alt.Y('mean:Q', title='Mean Score'), tooltip=['date:T', 'mean:Q', 'std:Q', 'count:Q'])
                _errorbars = alt.Chart(daily).mark_errorbar().encode(x=alt.X('date:T'), y=alt.Y('lower:Q', title='Mean Score'), y2=alt.Y2('upper:Q'))
                _chart = (_bars + _errorbars).properties(title=f'{_spec.name} — Daily Mean (± Std Dev)', width=CHART_WIDTH, height=CHART_HEIGHT)
                display(_chart)
            elif _stype == 'string_enum':
                daily_val = _group.groupby(['date', 'value']).size().reset_index(name='count')
                _line = alt.Chart(daily_val).mark_line(point=True).encode(x=alt.X('date:T', title='Date'), y=alt.Y('count:Q', title='Count'), color=alt.Color('value:N', title='Value'), tooltip=['date:T', 'value:N', 'count:Q']).properties(title=f'{_spec.name} — Daily Value Counts', width=CHART_WIDTH, height=CHART_HEIGHT)
                display(_line)
    else:
        display(Markdown('*No feedback data yet.*'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scores By Op
    Breaks down scores by the op (function) that generated each call.
    Boolean scorers use grouped bar charts (True vs False), numeric scorers show
    mean with std dev error bars, string enums use grouped bars per value,
    and free-text strings use a word cloud colored by op.

    *Note: we only have one op in this example.*
    """)
    return


@app.cell
def _(
    CHART_HEIGHT,
    CHART_WIDTH,
    Markdown,
    WordCloud,
    alt,
    display,
    enriched_df,
    pd,
    plt,
    scorer_ref_to_type,
    scorers,
):
    if not enriched_df.empty:
        for _ref_uri, _spec in scorers.items():
            _stype = scorer_ref_to_type[_ref_uri]
            _group = enriched_df[enriched_df['annotation_ref'] == _ref_uri].copy()
            if _group.empty:
                continue
            display(Markdown(f'### {_spec.name}'))
            if _stype == 'boolean':
                _group['value_str'] = _group['value'].astype(str)
                _agg = _group.groupby(['op', 'value_str']).size().reset_index(name='Count')
                _bar = alt.Chart(_agg).mark_bar().encode(x=alt.X('op:N', title='Op'), y=alt.Y('Count:Q', title='Count'), color=alt.Color('value_str:N', title='Value'), xOffset='value_str:N', tooltip=['op', 'value_str', 'Count']).properties(title=f'{_spec.name} by Op', width=CHART_WIDTH, height=CHART_HEIGHT)
                display(_bar)
            elif _stype in ('integer', 'number'):
                _group['num'] = pd.to_numeric(_group['value'], errors='coerce')
                _scored = _group.dropna(subset=['num'])
                if _scored.empty:
                    continue
                op_stats = _scored.groupby('op')['num'].agg(['mean', 'std', 'count']).reset_index()
                op_stats['std'] = op_stats['std'].fillna(0)
                op_stats['upper'] = op_stats['mean'] + op_stats['std']
                op_stats['lower'] = op_stats['mean'] - op_stats['std']
                _bars = alt.Chart(op_stats).mark_bar().encode(x=alt.X('op:N', title='Op', sort='-y'), y=alt.Y('mean:Q', title='Mean Score'), color=alt.Color('op:N', legend=None), tooltip=['op', 'mean', 'std', 'count'])
                _errorbars = alt.Chart(op_stats).mark_errorbar().encode(x=alt.X('op:N'), y=alt.Y('lower:Q', title='Mean Score'), y2=alt.Y2('upper:Q'))
                _chart = (_bars + _errorbars).properties(title=f'{_spec.name} by Op (Mean ± Std Dev)', width=CHART_WIDTH, height=CHART_HEIGHT)
                display(_chart)
                _best = op_stats.loc[op_stats['mean'].idxmax()]
                _worst = op_stats.loc[op_stats['mean'].idxmin()]
                _delta = _best['mean'] - _worst['mean']
                display(Markdown(f'**Best:** {_best['op']} ({_best['mean']:.2f}) · **Worst:** {_worst['op']} ({_worst['mean']:.2f}) · **Delta:** {_delta:.2f}'))
            elif _stype == 'string_enum':
                _agg = _group.groupby(['op', 'value']).size().reset_index(name='Count')
                _bar = alt.Chart(_agg).mark_bar().encode(x=alt.X('op:N', title='Op'), y=alt.Y('Count:Q', title='Count'), color=alt.Color('value:N', title='Value'), xOffset='value:N', tooltip=['op', 'value', 'Count']).properties(title=f'{_spec.name} by Op', width=CHART_WIDTH, height=CHART_HEIGHT)
                display(_bar)
            elif _stype == 'string_free':
                op_groups = _group.groupby('op')['value'].apply(list)
                _all_vals = [v for _vals in op_groups for v in _vals if pd.notna(v)]
                _wc_freq = dict(_top_words(_all_vals, n=200))
                if _wc_freq:
                    _wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(_wc_freq)
                    _fig, _ax = plt.subplots(figsize=(10, 5))
                    _ax.imshow(_wc, interpolation='bilinear')
                    _ax.axis('off')
                    _ax.set_title(f'{_spec.name} by Op — Word Cloud')
                    plt.tight_layout()
                    display(_fig)
                    plt.close(_fig)
                display(Markdown(f'**Top 10 words by Op:**'))
                _rows = []
                for op_name, _vals in op_groups.items():
                    for _rank, (_w, _c) in enumerate(_top_words(_vals), 1):
                        _rows.append({'Op': op_name, '#': _rank, 'Word': _w, 'Count': _c})
                if _rows:
                    display(pd.DataFrame(_rows))
    else:
        display(Markdown('*No feedback data yet.*'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scores By Model
    Same breakdowns as Scores by Op, but grouped by the model associated with each call.
    Boolean scorers use grouped bars (True vs False), numeric scorers show mean ± std dev,
    string enums use grouped bars per value, and free-text strings use word clouds.
    """)
    return


@app.cell
def _(
    CHART_HEIGHT,
    CHART_WIDTH,
    Markdown,
    WordCloud,
    alt,
    display,
    enriched_df,
    pd,
    plt,
    scorer_ref_to_type,
    scorers,
):
    if not enriched_df.empty:
        for _ref_uri, _spec in scorers.items():
            _stype = scorer_ref_to_type[_ref_uri]
            _group = enriched_df[enriched_df['annotation_ref'] == _ref_uri].copy()
            if _group.empty:
                continue
            display(Markdown(f'### {_spec.name}'))
            if _stype == 'boolean':
                _group['value_str'] = _group['value'].astype(str)
                _agg = _group.groupby(['model', 'value_str']).size().reset_index(name='Count')
                _bar = alt.Chart(_agg).mark_bar().encode(x=alt.X('model:N', title='Model'), y=alt.Y('Count:Q', title='Count'), color=alt.Color('value_str:N', title='Value'), xOffset='value_str:N', tooltip=['model', 'value_str', 'Count']).properties(title=f'{_spec.name} by Model', width=CHART_WIDTH, height=CHART_HEIGHT)
                display(_bar)
            elif _stype in ('integer', 'number'):
                _group['num'] = pd.to_numeric(_group['value'], errors='coerce')
                _scored = _group.dropna(subset=['num'])
                if _scored.empty:
                    continue
                model_stats = _scored.groupby('model')['num'].agg(['mean', 'std', 'count']).reset_index()
                model_stats['std'] = model_stats['std'].fillna(0)
                model_stats['upper'] = model_stats['mean'] + model_stats['std']
                model_stats['lower'] = model_stats['mean'] - model_stats['std']
                _bars = alt.Chart(model_stats).mark_bar().encode(x=alt.X('model:N', title='Model', sort='-y'), y=alt.Y('mean:Q', title='Mean Score'), color=alt.Color('model:N', legend=None), tooltip=['model', 'mean', 'std', 'count'])
                _errorbars = alt.Chart(model_stats).mark_errorbar().encode(x=alt.X('model:N'), y=alt.Y('lower:Q', title='Mean Score'), y2=alt.Y2('upper:Q'))
                _chart = (_bars + _errorbars).properties(title=f'{_spec.name} by Model (Mean ± Std Dev)', width=CHART_WIDTH, height=CHART_HEIGHT)
                display(_chart)
                _best = model_stats.loc[model_stats['mean'].idxmax()]
                _worst = model_stats.loc[model_stats['mean'].idxmin()]
                _delta = _best['mean'] - _worst['mean']
                display(Markdown(f'**Best:** {_best['model']} ({_best['mean']:.2f}) · **Worst:** {_worst['model']} ({_worst['mean']:.2f}) · **Delta:** {_delta:.2f}'))
            elif _stype == 'string_enum':
                _agg = _group.groupby(['model', 'value']).size().reset_index(name='Count')
                _bar = alt.Chart(_agg).mark_bar().encode(x=alt.X('model:N', title='Model'), y=alt.Y('Count:Q', title='Count'), color=alt.Color('value:N', title='Value'), xOffset='value:N', tooltip=['model', 'value', 'Count']).properties(title=f'{_spec.name} by Model', width=CHART_WIDTH, height=CHART_HEIGHT)
                display(_bar)
            elif _stype == 'string_free':
                model_groups = _group.groupby('model')['value'].apply(list)
                _all_vals = [v for _vals in model_groups for v in _vals if pd.notna(v)]
                _wc_freq = dict(_top_words(_all_vals, n=200))
                if _wc_freq:
                    _wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(_wc_freq)
                    _fig, _ax = plt.subplots(figsize=(10, 5))
                    _ax.imshow(_wc, interpolation='bilinear')
                    _ax.axis('off')
                    _ax.set_title(f'{_spec.name} by Model — Word Cloud')
                    plt.tight_layout()
                    display(_fig)
                    plt.close(_fig)
                display(Markdown(f'**Top 10 words by Model:**'))
                _rows = []
                for model_name, _vals in model_groups.items():
                    for _rank, (_w, _c) in enumerate(_top_words(_vals), 1):
                        _rows.append({'Model': model_name, '#': _rank, 'Word': _w, 'Count': _c})
                if _rows:
                    display(pd.DataFrame(_rows))
    else:
        display(Markdown('*No feedback data yet.*'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scorer: Correlation
    Pearson correlation heatmap between all numeric scorers. Each cell shows
    how strongly two scorers co-vary across calls. Values near +1 indicate
    strong positive correlation; near -1 indicates inverse correlation.
    """)
    return


@app.cell
def _(
    CHART_HEIGHT,
    CHART_WIDTH,
    Markdown,
    alt,
    display,
    feedback_df,
    pd,
    scorer_ref_to_type,
):
    if not feedback_df.empty:
        _numeric_refs = [ref for ref, st in scorer_ref_to_type.items() if st in ('integer', 'number')]
        if len(_numeric_refs) >= 2:
            fb_numeric = feedback_df[feedback_df['annotation_ref'].isin(_numeric_refs)].copy()
            fb_numeric['num'] = pd.to_numeric(fb_numeric['value'], errors='coerce')
            _pivot = fb_numeric.pivot_table(index='call_id', columns='scorer_name', values='num', aggfunc='mean')
            corr = _pivot.corr()
            corr_long = corr.reset_index().melt(id_vars='scorer_name', var_name='Scorer B', value_name='Correlation')
            corr_long = corr_long.rename(columns={'scorer_name': 'Scorer A'})
            _heatmap = alt.Chart(corr_long).mark_rect().encode(x=alt.X('Scorer A:N', title=''), y=alt.Y('Scorer B:N', title=''), color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='blues', domain=[-1, 1])), tooltip=['Scorer A', 'Scorer B', alt.Tooltip('Correlation:Q', format='.3f')]).properties(title='Scorer Correlation Matrix', width=CHART_WIDTH, height=CHART_HEIGHT)
            _text = alt.Chart(corr_long).mark_text(fontSize=14).encode(x=alt.X('Scorer A:N'), y=alt.Y('Scorer B:N'), text=alt.Text('Correlation:Q', format='.2f'))
            display(_heatmap + _text)
        else:
            display(Markdown('*Need 2+ numeric scorers for correlation analysis.*'))
    else:
        display(Markdown('*No feedback data yet.*'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Annotator Result Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inter-Annotator Agreement
    Measures how consistently different annotators score the same calls.
    Boolean scorers are skipped. For numeric scorers, the top-5 most disagreed
    calls (highest within-call std dev) are listed. For string enum scorers,
    the distribution of agreement and top-5 most divergent calls are shown.
    """)
    return


@app.cell
def _(
    Markdown,
    call_model_map,
    call_op_map,
    call_url,
    display,
    feedback_df,
    pd,
    scorer_ref_to_type,
    scorers,
    short_op,
):
    if not feedback_df.empty:
        for _ref_uri, _spec in scorers.items():
            _stype = scorer_ref_to_type[_ref_uri]
            _group = feedback_df[feedback_df['annotation_ref'] == _ref_uri].copy()
            multi = _group.groupby('call_id').filter(lambda g: g['annotator'].nunique() > 1)
            if multi.empty:
                continue
            if _stype in ('boolean', 'string_free'):
                continue
            n_calls = multi['call_id'].nunique()
            display(Markdown(f'### {_spec.name} ({n_calls} calls with 2+ annotators)'))
            if _stype in ('integer', 'number'):
                multi = multi.copy()
                multi['num'] = pd.to_numeric(multi['value'], errors='coerce')
                _per_call = multi.groupby('call_id')['num'].agg(['std', 'min', 'max', 'count']).reset_index()
                _per_call = _per_call.dropna(subset=['std'])
                worst5 = _per_call.nlargest(5, 'std')
                display(Markdown('**Top 5 most disagreed calls:**'))
                _rows = []
                for _, row in worst5.iterrows():
                    cid = row['call_id']
                    _rows.append({'Call ID': cid, 'Op': short_op(call_op_map.get(cid, '(unknown)')), 'Model': call_model_map.get(cid, '(unknown)'), 'URL': call_url(cid), 'Spread': f'{row['max'] - row['min']:.1f}', 'Std Dev': f'{row['std']:.2f}'})
                display(pd.DataFrame(_rows))
            elif _stype == 'string_enum':
                exact = multi.groupby('call_id')['value'].apply(lambda v: v.nunique() == 1)
                agreement_rate = exact.mean() * 100
                display(Markdown(f'**Exact agreement rate:** {agreement_rate:.1f}%'))
                disagreed = exact[~exact].index.tolist()[:5]
                if disagreed:
                    display(Markdown('**Top 5 most divergent calls:**'))
                    display(_make_call_table(disagreed))
    else:
        display(Markdown('*No feedback data yet.*'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Annotator Scorer Matrix
    Heatmap of annotator x scorer average scores. Each cell shows the mean
    numeric score an annotator gave for a particular scorer, revealing
    systematic patterns across annotators and scoring dimensions.
    """)
    return


@app.cell
def _(
    CHART_HEIGHT,
    CHART_WIDTH,
    Markdown,
    alt,
    display,
    feedback_df,
    pd,
    scorer_ref_to_type,
):
    if not feedback_df.empty:
        _numeric_refs = [ref for ref, st in scorer_ref_to_type.items() if st in ('integer', 'number')]
        numeric_fb = feedback_df[feedback_df['annotation_ref'].isin(_numeric_refs)].copy()
        if not numeric_fb.empty:
            numeric_fb['num'] = pd.to_numeric(numeric_fb['value'], errors='coerce')
            _pivot = numeric_fb.pivot_table(index='annotator', columns='scorer_name', values='num', aggfunc='mean')
            heatmap_data = _pivot.reset_index().melt(id_vars='annotator', var_name='Scorer', value_name='Avg Score')
            heatmap_data = heatmap_data.dropna(subset=['Avg Score'])
            _heatmap = alt.Chart(heatmap_data).mark_rect().encode(x=alt.X('Scorer:N', title='Scorer'), y=alt.Y('annotator:N', title='Annotator'), color=alt.Color('Avg Score:Q', scale=alt.Scale(scheme='blues')), tooltip=['annotator', 'Scorer', alt.Tooltip('Avg Score:Q', format='.2f')]).properties(title='Annotator × Scorer Matrix (Avg Score)', width=CHART_WIDTH, height=CHART_HEIGHT)
            _text = alt.Chart(heatmap_data).mark_text(fontSize=12).encode(x=alt.X('Scorer:N'), y=alt.Y('annotator:N'), text=alt.Text('Avg Score:Q', format='.2f'))
            display(_heatmap + _text)
        else:
            display(Markdown('*No numeric scorer data available.*'))
    else:
        display(Markdown('*No feedback data yet.*'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Annotator Bias
    For each numeric scorer, shows each annotator's deviation from the global mean
    as a diverging bar chart — bars above zero (blue) indicate the annotator scores
    higher than average, bars below (red) indicate lower. The dashed line marks the mean.
    """)
    return


@app.cell
def _(
    CHART_HEIGHT,
    CHART_WIDTH,
    Markdown,
    alt,
    display,
    feedback_df,
    pd,
    scorer_ref_to_type,
    scorers,
):
    if not feedback_df.empty:
        for _ref_uri, _spec in scorers.items():
            _stype = scorer_ref_to_type[_ref_uri]
            if _stype not in ('integer', 'number'):
                continue
            _group = feedback_df[feedback_df['annotation_ref'] == _ref_uri].copy()
            _group['num'] = pd.to_numeric(_group['value'], errors='coerce')
            _scored = _group.dropna(subset=['num'])
            if _scored.empty:
                continue
            global_mean = _scored['num'].mean()
            global_std = _scored['num'].std()
            bias_rows = []
            for ann, agrp in _scored.groupby('annotator'):
                m = agrp['num'].mean()
                z = (m - global_mean) / global_std if global_std > 0 else 0
                bias_rows.append({'Annotator': ann or '(unknown)', 'Mean': m, 'Bias': m - global_mean, 'z-score': z, 'n': len(agrp), 'Significant': abs(z) > 1.5})
            bias_df = pd.DataFrame(bias_rows)
            _bars = alt.Chart(bias_df).mark_bar().encode(x=alt.X('Annotator:N', title='Annotator'), y=alt.Y('Bias:Q', title='Deviation from Mean', scale=alt.Scale(zero=True)), color=alt.Color('Annotator:N', legend=None), tooltip=['Annotator', alt.Tooltip('Mean:Q', format='.2f'), alt.Tooltip('Bias:Q', format='+.2f'), alt.Tooltip('z-score:Q', format='+.2f'), 'n'])
            rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[4, 4], color='black', strokeWidth=1.5).encode(y='y:Q')
            _chart = (_bars + rule).properties(title=f'{_spec.name} — Annotator Bias (global mean = {global_mean:.2f})', width=CHART_WIDTH, height=CHART_HEIGHT)
            display(_chart)
    else:
        display(Markdown('*No feedback data yet.*'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw Data
    The full feedback DataFrame used to generate this report, with op and model columns joined in.
    """)
    return


@app.cell
def _(Markdown, display, enriched_df):
    if not enriched_df.empty:
        display(enriched_df)
    else:
        display(Markdown("*No data.*"))
    return


if __name__ == "__main__":
    app.run()

