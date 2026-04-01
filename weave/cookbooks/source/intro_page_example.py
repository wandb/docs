# /// script
# dependencies = ["openai", "weave"]
# ///

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
    # Parallel Evaluation with W&B Weave

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/docs/blob/main/weave/cookbooks/source/parallel_evaluation_example.ipynb)

    This notebook demonstrates how to use W&B Weave to send math questions to OpenAI and evaluate the responses for correctness in parallel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Installation

    First, install the required packages:
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: weave openai !pip install weave openai -qU
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup API Keys

    Add your W&B and OpenAI API keys:
    """)
    return


@app.cell
def _():
    import os
    from getpass import getpass

    # Set your OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

    # Log in to W&B
    import wandb
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parallel Evaluation Example

    Run the evaluation example:
    """)
    return


@app.cell
async def _():
    import weave
    from openai import OpenAI
    from weave import Scorer
    import asyncio

    # Initialize Weave
    weave.init("parallel-evaluation")

    # Create OpenAI client
    client = OpenAI()

    # Define your model as a weave.op function
    @weave.op
    def math_model(question: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content

    # Create a dataset with questions and expected answers
    dataset = [
        {"question": "What is 2+2?", "expected": "4"},
        {"question": "What is 5+3?", "expected": "8"},
        {"question": "What is 10-7?", "expected": "3"},
        {"question": "What is 12*3?", "expected": "36"},
        {"question": "What is 100/4?", "expected": "25"},
    ]

    # Define a class-based scorer
    class CorrectnessScorer(Scorer):
        """Scorer that checks if the answer is correct"""
    
        @weave.op
        def score(self, question: str, expected: str, output: str) -> dict:
            """Check if the model output contains the expected answer"""
            import re
        
            # Extract numbers from the output
            numbers = re.findall(r'\d+', output)
        
            if numbers:
                answer = numbers[0]
                correct = answer == expected
            else:
                correct = False
        
            return {
                "correct": correct,
                "extracted_answer": numbers[0] if numbers else None,
                "contains_expected": expected in output
            }

    # Instantiate the scorer
    correctness_scorer = CorrectnessScorer()

    # Create an evaluation
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[correctness_scorer]
    )

    # Run the evaluation - automatically evaluates examples in parallel
    await evaluation.evaluate(math_model)
    return asyncio, evaluation, math_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Note for Google Colab Users

    If you're running this notebook in Google Colab, you may need to handle async differently. Use this version instead:
    """)
    return


@app.cell
def _(asyncio, evaluation, math_model):
    # For Google Colab, use this approach:
    import nest_asyncio
    nest_asyncio.apply()

    # Then run the evaluation
    asyncio.run(evaluation.evaluate(math_model))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## View Results

    After running the evaluation, you can view the results in the W&B Weave dashboard. The evaluation shows:

    1. **Parallel execution**: All examples are evaluated simultaneously for faster results
    2. **Correctness scores**: Each response is scored based on whether it contains the correct answer
    3. **Detailed metrics**: Including extracted answers and whether the expected value was found

    Visit your [W&B Weave dashboard](https://wandb.ai/home) to explore the evaluation results in detail.
    """)
    return


if __name__ == "__main__":
    app.run()

