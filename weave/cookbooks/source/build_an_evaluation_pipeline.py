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
    # Learn how to build an evaluation pipeline with Weave Models and Evaluations

    Evaluations help you iterate and improve your applications by testing them against a set of examples after you make changes. Weave provides first-class support for tracking evaluations with `Model` and `Evaluation` classes. The APIs are designed with minimal assumptions, allowing flexibility for a wide array of use cases.

    The guide walks you through how to:

    * Set up a `Model`
    * Create a dataset to test an LLM's responses against
    * Define a scoring function to compare model output to expected outputs
    * Run an evaluation that tests the model against dataset using the scoring function and an additional built-in scorer
    * View the results of the evaluation in the Weave UI
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Install and import the necessary libraries and functions

    Import the following libraries into your script:
    """)
    return


app._unparsable_cell(
    r"""
    pip install weave openai
    """,
    name="_"
)


@app.cell
def _():
    import json
    import openai
    import asyncio
    import weave
    from weave.scorers import MultiTaskBinaryClassificationF1

    return MultiTaskBinaryClassificationF1, json, openai, weave


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set OpenAI API key and W&B team

    The following cell prompts you to enter your OpenAI API key and the name of the W&B team you want to send output to. If you do not have a W&B team, [create one](https://docs.wandb.ai/platform/hosting/iam/access-management/manage-organization#create-a-team).

    The notebook also prompts you to enter you to enter your W&B API key when you run `weave.init()` the first time.
    """)
    return


@app.cell
def _():
    # Enter your OpenAI API key and your W&B team name when prompted
    OPENAI_API_KEY = input("Enter your OpenAI API key: ")
    WB_TEAM_NAME = input("Enter your W&B team name: ")
    return OPENAI_API_KEY, WB_TEAM_NAME


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build a `Model`

    In Weave, [`Models` are objects](/weave/guides/core-types/models) that capture both the behavior of your model/agent (logic, prompt, parameters) and its versioned metadata (parameters, code, micro-config) so you can track, compare, evaluate and iterate reliably.

    When you instantiate a `Model`, Weave automatically captures its configuration and behaviors and updates the version when there are changes. This allows you to track its performance over time as you iterate on it.

    `Model`s are declared by subclassing `Model` and implementing a `predict` function definition, which takes one example and returns the response.

    The following example model uses OpenAI to extract the names, colors, and flavors of alien fruits from sentences sent to it.
    """)
    return


@app.cell
def _(OPENAI_API_KEY, json, openai, weave):
    class ExtractFruitsModel(weave.Model):
        model_name: str
        prompt_template: str

        @weave.op()
        async def predict(self, sentence: str) -> dict:
            client = openai.AsyncClient(api_key=OPENAI_API_KEY)

            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": self.prompt_template.format(sentence=sentence)}
                ],
            )
            result = response.choices[0].message.content
            if result is None:
                raise ValueError("No response from model")
            parsed = json.loads(result)
            return parsed

    return (ExtractFruitsModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `ExtractFruitsModel` class inherits from (or subclasses) `weave.Model` so that Weave can track the instantiated object. `@weave.op` decorates the predict function to track its inputs and outputs.

    You can instantiate `Model` objects like this:
    """)
    return


@app.cell
async def _(ExtractFruitsModel, WB_TEAM_NAME, weave):
    weave.init(WB_TEAM_NAME + '/eval_pipeline_quickstart')

    model = ExtractFruitsModel(
        model_name='gpt-3.5-turbo-1106',
        prompt_template='Extract fields ("fruit": <str>, "color": <str>, "flavor": <str>) from the following text, as json: {sentence}'
    )

    sentence = "There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy."

    await model.predict(sentence)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create a dataset

    Next, you need a dataset to evaluate your model on. A [`Dataset` is a collection of examples stored as a Weave object](/weave/guides/core-types/datasets).

    The following example dataset defines three example input sentences and their correct answers (`labels`), and then formats them in a JSON table table format that scoring functions can read.

    This example builds a list of examples in code, but you can also log them one at a time from your running application.
    """)
    return


@app.cell
def _():
    sentences = ["There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy.",
    "Pounits are a bright green color and are more savory than sweet.",
    "Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, and a pale orange tinge to them."]
    labels = [
        {'fruit': 'neoskizzles', 'color': 'purple', 'flavor': 'candy'},
        {'fruit': 'pounits', 'color': 'bright green', 'flavor': 'savory'},
        {'fruit': 'glowls', 'color': 'pale orange', 'flavor': 'sour and bitter'}
    ]
    examples = [
        {'id': '0', 'sentence': sentences[0], 'target': labels[0]},
        {'id': '1', 'sentence': sentences[1], 'target': labels[1]},
        {'id': '2', 'sentence': sentences[2], 'target': labels[2]}
    ]
    return (examples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then create your dataset using the `weave.Dataset()` class and publish it:
    """)
    return


@app.cell
def _(WB_TEAM_NAME, examples, weave):
    weave.init(WB_TEAM_NAME + '/eval_pipeline_quickstart')
    dataset = weave.Dataset(name='fruits', rows=examples)
    weave.publish(dataset)
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define custom scoring functions

    When using Weave evaluations, Weave expects a `target` to compare `output` against. The following scoring function takes two dictionaries (`target` and `output`) and returns a dictionary of boolean values indicating whether the output matches the target. The `@weave.op()` decorator enables Weave to track the scoring function's execution.
    """)
    return


@app.cell
def _(weave):
    @weave.op()
    def fruit_name_score(target: dict, output: dict) -> dict:
        return {'correct': target['fruit'] == output['fruit']}

    return (fruit_name_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To make your own scoring function, learn more in the [Scorers](/weave/guides/evaluation/scorers) guide.

    In some applications, you may want to create custom `Scorer` classes. For example, you might create a standardized `LLMJudge` class with specific parameters (such as chat model or prompt), specific row scoring, and aggregate score calculation. See the tutorial on defining a `Scorer` class in the next chapter on [Model-Based Evaluation of RAG applications](/weave/tutorial-rag#optional-defining-a-scorer-class) for more information.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use a built-in scorer and run the evaluation

    Along with custom scoring functions, you can also use [Weave's built-in scorers](/weave/guides/evaluation/builtin_scorers). In the following evaluation, `weave.Evaluation()` uses the `fruit_name_score` function defined in the previous section and the built-in `MultiTaskBinaryClassificationF1` scorer, which computes [F1 scores](https://en.wikipedia.org/wiki/F-score).

    The following example runs an evaluation of `ExtractFruitsModel` on the `fruits` dataset using the scoring the two functions and logs the results to Weave.
    """)
    return


@app.cell
async def _(
    MultiTaskBinaryClassificationF1,
    WB_TEAM_NAME,
    dataset,
    fruit_name_score,
    model,
    weave,
):
    weave.init(WB_TEAM_NAME + '/eval_pipeline_quickstart''eval_pipeline_quickstart')

    evaluation = weave.Evaluation(
        name='fruit_eval',
        dataset=dataset,
        scorers=[
            MultiTaskBinaryClassificationF1(class_names=["fruit", "color", "flavor"]),
            fruit_name_score
        ],
    )

    # if you're in a Jupyter Notebook, run:
    await evaluation.evaluate(model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## View your evaluation results

    Weave automatically captures traces of each prediction and score. Click on the link printed by the evaluation to view the results in the Weave UI.

    ![Evaluation results](/images/evals-hero.png)

    ## Learn more about Weave evaluations

    * Learn more about how to [build and use scorers](/weave/guides/evaluation/scorers).
    * Check out Weave's [built-in scoring functions](/weave/guides/evaluation/builtin_scorers).
    * Learn about [Model-Based Evaluation](/weave/guides/evaluation/scorers#model-based-evaluation) for using LLMs as judges.

    ## Next Steps

    [Build a RAG application](/weave/tutorial-rag) to learn about evaluating retrieval-augmented generation.
    """)
    return


if __name__ == "__main__":
    app.run()

