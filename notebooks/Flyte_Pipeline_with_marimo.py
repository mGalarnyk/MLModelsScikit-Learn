import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import inspect
    import json
    import re
    import time
    from urllib.request import urlretrieve

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn import metrics

    import flyte
    import fraud_pipeline as fp

    # Flyte's local run records: which tasks ran, which came from the cache, retries, and how long each took.
    from flyte._persistence._run_store import RunStore

    FLYTE_LOGO = "https://raw.githubusercontent.com/flyteorg/static-resources/main/common/flyte_circle_gradient_1_4x4.png"


@app.function(hide_code=True)
def task_statuses(actions):
    """Turn Flyte's run records into {task: ("ran" or "cached", seconds, failed attempts)}. Model tasks are keyed by (model, params)."""
    statuses = {}
    for a in actions:
        task = a.task_name.split(".")[-1]
        if task == "fraud_pipeline":
            continue
        if task == "evaluate_model":
            inputs = json.loads(a.inputs)
            task = (inputs["model_name"], json.dumps(inputs["params"], sort_keys=True))
        # Each failed attempt Flyte retried, as the error type it recorded (for example ConnectionError)
        failures = [re.search(r'code: "(\w+)"', at["error"] or "") for at in json.loads(a.attempts_json or "[]")
                    if at.get("status") == "failed"]
        failures = [m.group(1) if m else "error" for m in failures]
        statuses[task] = ("cached" if a.cache_hit else "ran", a.end_time - a.start_time, failures)
    return statuses


@app.function(hide_code=True)
def pipeline_mermaid(n_rows, fraud_rate, experiments, results, best, statuses):
    """Draw one Flyte run as a Mermaid flowchart. Tasks that ran are green with animated edges, reused tasks are gray,
    and each failed attempt that Flyte retried gets an orange arrow ending in ✗."""
    short = {"Logistic Regression": "Logistic Regression", "Random Forest": "Random Forest",
             "Gradient-Boosted Trees": "Gradient-Boosted Trees"}
    scores = {(r["model"], json.dumps(r["params"], sort_keys=True)): r["PR-AUC"] for r in results}

    retried = []

    def status(key, node_id):
        state, seconds, failures = statuses.get(key, ("unknown", 0, []))
        label = {"ran": f"✓ ran in {seconds:.1f}s", "cached": "⚡ reused from cache", "unknown": ""}[state]
        if failures:
            retried.append((node_id, failures))
            label += f"<br/>retried after {failures[0]}"
        return state, label

    lines = ['%%{init: {"flowchart": {"curve": "basis", "nodeSpacing": 18, "rankSpacing": 55}}}%%', "flowchart LR",
             f'    data[("<b>Transactions</b><br/>{n_rows:,} rows · {fraud_rate:.2%} fraud")]:::data']
    tag = "<small>FLYTE TASK</small><br/>"
    edges, classes = [], []

    def task_node(node_id, title, key):
        state, label = status(key, node_id)
        lines.append(f'    {node_id}("{tag}<b>{title}</b><br/><small>{label}</small>")')
        classes.append(f"    class {node_id} {state}")
        return state

    fe = task_node("fe", "engineer_features", "engineer_features")
    sp = task_node("sp", "split_data", "split_data")
    edges += [("data", "fe", fe), ("fe", "sp", sp)]

    lines.append(f'    subgraph compare["Flyte runs {sum(len(p) for p in experiments.values())} model tasks in parallel"]')
    model_ids = []
    best_key = (best["model"], json.dumps(best["params"], sort_keys=True))
    for family, param_list in experiments.items():
        for params in param_list:
            key = (family, json.dumps(params, sort_keys=True))
            node_id = f"m{len(model_ids)}"
            label = ", ".join(f"{k}={v}" for k, v in params.items() if k != "min_samples_leaf")
            state, status_label = status(key, node_id)
            lines.append(f'    {node_id}("{tag}<b>{short[family]}</b><br/>{label}<br/>PR-AUC <b>{scores[key]:.3f}</b><br/><small>{status_label}</small>")')
            classes.append(f"    class {node_id} {state}")
            if key == best_key:
                classes.append(f"    class {node_id} best")
            edges.append(("sp", node_id, state))
            model_ids.append(node_id)
    lines.append("    end")

    sb = task_node("sb", f"score_best_model", "score_best_model")
    lines[-1] = lines[-1].replace("</b><br/><small>", f"</b><br/>Winner: {best['model']}<br/><small>")
    edges += [(m, "sb", sb) for m in model_ids]
    lines.append('    decide[/"<b>Decide in marimo</b><br/>threshold · cost · capacity"/]:::marimo')
    edges.append(("sb", "decide", "ran"))

    for i, (a, b, state) in enumerate(edges):
        lines.append(f"    {a} e{i}@--> {b}")
        if state == "ran":
            lines.append(f"    e{i}@{{ animate: true }}")
    # Each failed attempt Flyte retried: an orange arrow into the task that ends in ✗, next to the arrow that succeeded
    n_links = len(edges)
    for node_id, failures in retried:
        source = next(a for a, b, _ in edges if b == node_id)
        for attempt, error in enumerate(failures, start=1):
            lines.append(f'    {source} -. "✗ attempt {attempt} · {error}" .-x {node_id}')
            lines.append(f"    linkStyle {n_links} stroke:#ea580c,stroke-width:3px,color:#9a3412")
            n_links += 1
    lines += classes
    lines += ["    classDef data fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a",
              "    classDef ran fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d",
              "    classDef cached fill:#f3f4f6,stroke:#9ca3af,stroke-width:2px,stroke-dasharray:5 4,color:#4b5563",
              "    classDef unknown fill:#ffffff,stroke:#9ca3af,stroke-width:2px,color:#374151",
              "    classDef best stroke:#f59e0b,stroke-width:4px",
              "    classDef marimo fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f",
              "    style compare fill:#f5f3ff,stroke:#c4b5fd,color:#5b21b6"]
    return "\n".join(lines)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # The Fraud Detection Pipeline with [Flyte 2](https://github.com/flyteorg/flyte) and [marimo](https://github.com/marimo-team/marimo)

    In `Anomaly_Detection_Evaluation.ipynb` and `Turning_Scores_into_Decisions.ipynb` we built a fraud detection workflow one cell at a time. Here the same workflow is a **pipeline**:

    * [**Flyte 2**](https://github.com/flyteorg/flyte) runs the expensive steps (feature engineering, comparing models, scoring) as tasks. The tasks live in a plain Python file, `fraud_pipeline.py`, and run on your laptop or on a cluster without changes.
    * [**marimo**](https://github.com/marimo-team/marimo) is the interactive front end. [marimo](https://github.com/marimo-team/marimo) notebooks are **reactive**: when you change a control, every cell that depends on it reruns automatically. This notebook is itself a Python file, so it works well with git and can be served as an app with `marimo run`.

    Launch it from the `notebooks` folder with `marimo edit Flyte_Pipeline_with_marimo.py`.
    """)
    return


@app.cell(hide_code=True)
def _():
    run_form = mo.md("""
    {models}

    {where}

    {lose_worker}
    """).batch(
        models=mo.ui.multiselect(options=list(fp.EXPERIMENTS), value=["Logistic Regression", "Random Forest"],
                                 label="Models to compare"),
        where=mo.ui.radio(options=["Locally (no Docker needed)", "Flyte devbox (needs Docker)"],
                          value="Locally (no Docker needed)", label="Where to run", inline=True),
        lose_worker=mo.ui.checkbox(value=True, label="Simulate losing a worker: the last model task fails its first attempt, and Flyte retries it"),
    ).form(submit_button_label="Run pipeline with Flyte")

    get_history, set_history = mo.state([])

    mo.vstack([
        mo.md("""## Run the Pipeline
    The pipeline runs locally with two model families when the notebook opens (about a minute). Then add **Gradient-Boosted Trees** and run it again: [Flyte](https://github.com/flyteorg/flyte) only runs the new tasks and reuses everything else from its cache.

    To run on the [Flyte](https://github.com/flyteorg/flyte) devbox instead, start it first (see the README) and create `config.yaml` in this folder with `flyte create config --devbox`."""),
        run_form,
    ])
    return get_history, run_form, set_history


@app.cell(hide_code=True)
def _(run_form, set_history):
    settings = run_form.value or {"models": ["Logistic Regression", "Random Forest"], "where": "Locally (no Docker needed)",
                                  "lose_worker": True}
    on_devbox = settings["where"].startswith("Flyte devbox")
    config_path = mo.notebook_dir() / "config.yaml"

    mo.stop(not settings["models"], mo.callout(mo.md("Pick at least one model."), kind="warn"))
    mo.stop(on_devbox and not config_path.exists(),
            mo.callout(mo.md("No `config.yaml` found. Run `flyte create config --devbox` in the `notebooks` folder."), kind="warn"))

    _data_path = mo.notebook_dir() / "data" / "creditcard.csv"
    if not _data_path.exists():
        # Download the data once (about 150 MB)
        urlretrieve("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv", _data_path)
    raw = pd.read_csv(_data_path)
    experiments = {m: fp.EXPERIMENTS[m] for m in settings["models"]}
    _n_tasks = sum(len(p) for p in experiments.values())

    if on_devbox:
        flyte.init_from_config(config_path, root_dir=mo.notebook_dir())
    else:
        flyte.init(local_persistence=True)

    _start = time.time()
    with mo.status.spinner(title=f"Running {_n_tasks} model tasks {'on the devbox' if on_devbox else 'locally'}..."):
        if on_devbox:
            run = flyte.run(fp.fraud_pipeline, raw=raw, model_names=settings["models"], lose_worker=settings["lose_worker"])
            run.wait()
            # Read back only the output. run.outputs() would also download the inputs from the cluster's storage.
            output = run.typed_outputs({"o0": dict})["o0"]
            statuses = {}
        else:
            _name = f"fraud-{int(time.time() * 1000)}"
            run = flyte.with_runcontext(name=_name).run(fp.fraud_pipeline, raw=raw, model_names=settings["models"],
                                                        lose_worker=settings["lose_worker"])
            output = run.outputs()[0]
            statuses = task_statuses(RunStore.list_actions_for_run_sync(_name))
    run_seconds = time.time() - _start
    ran_where = "the Flyte devbox" if on_devbox else "locally"

    _diagram = pipeline_mermaid(len(raw), raw["Class"].mean(), experiments,
                                output["results"], output["best"], statuses)
    _reused = sum(state == "cached" for state, _, _ in statuses.values())
    set_history(lambda h: h + [{"run": len(h) + 1, "diagram": _diagram, "seconds": run_seconds,
                                "reused": _reused, "total": len(statuses), "where": ran_where}])
    return experiments, output, ran_where, raw, run, run_seconds, statuses


@app.cell(hide_code=True)
def _(get_history, output, ran_where, raw, run, run_seconds, statuses):
    _history = get_history()
    _latest = _history[-1]
    _reused = f"{_latest['reused']} of {_latest['total']}" if statuses else "see Flyte UI"
    _stats = mo.hstack([
        mo.stat(f"{len(raw):,}", label="Transactions", caption=f"{raw['Class'].mean():.2%} fraud", bordered=True),
        mo.stat(f"{run_seconds:.1f}s", label="Run time", caption=f"ran {ran_where}", bordered=True),
        mo.stat(_reused, label="Tasks reused from cache", caption="Flyte skips unchanged work", bordered=True),
        mo.stat(output["best"]["model"], label="Best model", caption=f"PR-AUC {output['best']['PR-AUC']:.3f}", bordered=True),
    ], widths="equal", gap=1)

    _tabs = mo.ui.tabs({(f"Run {h['run']} (latest)" if h is _latest else f"Run {h['run']}"):
                        mo.vstack([mo.md(f"<small>{h['where']} · {h['seconds']:.1f}s · {h['reused']} of {h['total']} tasks reused</small>"),
                                   mo.mermaid(h["diagram"])])
                        for h in reversed(_history)})
    _link = f" Follow the run in the [Flyte UI]({run.url})." if ran_where != "locally" else ""

    mo.vstack([
        mo.hstack([mo.image(FLYTE_LOGO, width=44, alt="Flyte logo"), mo.md("## The [Flyte](https://github.com/flyteorg/flyte) Pipeline, Live")],
                  justify="start", align="center", gap=0.75),
        mo.md(f"Each box is a [Flyte](https://github.com/flyteorg/flyte) task, drawn from the run that just finished.{_link} "
              "**Green** tasks ran (their arrows move), **gray dashed** tasks were reused from [Flyte](https://github.com/flyteorg/flyte)'s cache, "
              "an **orange ✗ arrow** is an attempt that failed and [Flyte](https://github.com/flyteorg/flyte) retried, "
              "and the **gold** box is the best model. Earlier runs stay in their own tabs."),
        _stats,
        _tabs,
    ])
    return


@app.cell(hide_code=True)
def _():
    _tasks = [fp.engineer_features, fp.split_data, fp.evaluate_model, fp.score_best_model, fp.fraud_pipeline]
    mo.vstack([
        mo.hstack([mo.image(FLYTE_LOGO, width=44, alt="Flyte logo"), mo.md("## [Flyte](https://github.com/flyteorg/flyte) Tasks")],
                  justify="start", align="center", gap=0.75),
        mo.md("""A [Flyte](https://github.com/flyteorg/flyte) task is an ordinary Python function with an `@env.task` decorator. Options on the decorator add [Flyte](https://github.com/flyteorg/flyte) features: `cache` reuses results when the inputs haven't changed, and `retries` reruns a task that fails. The `TaskEnvironment` says which packages and resources each task gets on a cluster (locally these settings are ignored). Open a task to read its code."""),
        mo.accordion({f"`{t.func.__name__}`": mo.ui.code_editor(inspect.getsource(t.func), language="python", disabled=True)
                      for t in _tasks}),
    ])
    return


@app.cell(hide_code=True)
def _(output):
    _results = (pd.DataFrame(output["results"])
                .assign(params=lambda d: d["params"].astype(str))
                .sort_values("PR-AUC", ascending=False)
                .round(3)
                .reset_index(drop=True))
    mo.accordion({"All model results": mo.ui.table(_results, selection=None)})
    return


@app.cell(hide_code=True)
def _():
    cost_review = mo.ui.slider(1, 50, step=1, value=5, label="Cost of reviewing a transaction ($)", show_value=True)
    fee = mo.ui.slider(0, 100, step=5, value=0, label="Extra cost per missed fraud ($)", show_value=True)
    threshold = mo.ui.slider(0.01, 0.9, step=0.01, value=0.5, label="Your threshold", show_value=True)

    mo.vstack([
        mo.md("""## Turning Scores into Decisions
    [Flyte](https://github.com/flyteorg/flyte) did the heavy lifting once. Everything below reruns instantly on the scores it returned.

    A missed fraud costs its **transaction amount** (plus any extra cost you add, such as a chargeback fee), and every review costs analyst time. Move the sliders: the lowest-cost threshold is chosen on the **validation** set, and the results are reported on the **test** set."""),
        mo.hstack([cost_review, fee], justify="start", gap=2),
        threshold,
    ])
    return cost_review, fee, threshold


@app.cell(hide_code=True)
def _(cost_review, fee, output, threshold):
    val = {k: np.array(output[f"val_{k}"]) for k in ["labels", "amounts", "scores"]}
    test = {k: np.array(output[f"test_{k}"]) for k in ["labels", "amounts", "scores"]}

    def total_cost(data, t):
        flagged = data["scores"] >= t
        missed = (data["labels"] == 1) & ~flagged
        return data["amounts"][missed].sum() + missed.sum() * fee.value + flagged.sum() * cost_review.value

    _thresholds = np.linspace(0.01, 0.9, 90)
    _costs = [total_cost(val, t) for t in _thresholds]
    best_threshold = float(_thresholds[np.argmin(_costs)])

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_thresholds, _costs, color="#4c72b0", linewidth=2)
    for _t, _label, _color in [(best_threshold, f"Lowest cost: {best_threshold:.2f}", "#c44e52"),
                               (threshold.value, f"Your threshold: {threshold.value:.2f}", "black")]:
        _ax.axvline(_t, color=_color, linestyle="--", linewidth=1.5)
        _ax.annotate(_label, (_t, max(_costs)), xytext=(4, -4), textcoords="offset points",
                     va="top", color=_color, fontsize=10)
    _ax.set_xlabel("Threshold")
    _ax.set_ylabel("Total cost on validation set ($)")
    for _side in ["top", "right"]:
        _ax.spines[_side].set_visible(False)
    _fig.tight_layout()
    _fig
    return best_threshold, test, total_cost


@app.cell(hide_code=True)
def _(best_threshold, test, threshold, total_cost):
    test_labels, test_scores = test["labels"], test["scores"]

    def summarize(t, policy):
        flagged = test_scores >= t
        scale = 100000 / len(test_labels)
        return {"policy": policy,
                "threshold": round(t, 2),
                "flagged": round(flagged.sum() * scale),
                "caught": round(((test_labels == 1) & flagged).sum() * scale),
                "missed": round(((test_labels == 1) & ~flagged).sum() * scale),
                "false alarms": round(((test_labels == 0) & flagged).sum() * scale),
                "precision": round(metrics.precision_score(test_labels, flagged, zero_division=0), 3),
                "recall": round(metrics.recall_score(test_labels, flagged), 3),
                "cost ($)": round(total_cost(test, t) * scale)}

    _summary = pd.DataFrame([summarize(0.5, "Default (0.5)"),
                             summarize(best_threshold, "Lowest cost"),
                             summarize(threshold.value, "Your threshold")])

    mo.vstack([
        mo.md("### What Each Policy Means per 100,000 Transactions (Test Set)"),
        mo.ui.table(_summary, selection=None, show_column_summaries=False),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why [Flyte](https://github.com/flyteorg/flyte) and [marimo](https://github.com/marimo-team/marimo) Together

    * **Same code, laptop or cluster.** `fraud_pipeline.py` is plain Python. `flyte.init()` runs it locally, and `flyte.init_from_config()` runs each task in its own container on a cluster, with inputs, outputs, and logs recorded for every task.
    * **Failures don't restart the pipeline.** A task that loses its worker is retried on its own (`retries=2`), and every task that already finished stays cached.
    * **Expensive work once, decisions instantly.** [Flyte](https://github.com/flyteorg/flyte) runs the model comparison. [marimo](https://github.com/marimo-team/marimo) reruns only the cells that depend on a slider, so exploring thresholds and costs never retrains a model.
    * **Shareable.** The notebook is a `.py` file that diffs cleanly in git, and `marimo run Flyte_Pipeline_with_marimo.py` serves it as an app for stakeholders who just want the sliders and the table.
    """)
    return


if __name__ == "__main__":
    app.run()
