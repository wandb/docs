# /// script
# dependencies = ["datasets", "wandb", "weave"]
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
    <!-- docusaurus_head_meta::start
    ---
    title: Using HuggingFace Datasets in evaluations with `preprocess_model_input`
    ---
    docusaurus_head_meta::end -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using HuggingFace Datasets in evaluations with `preprocess_model_input`

    ## Note: This is a temporary workaround
    > This guide demonstrates a workaround for using HuggingFace Datasets with Weave evaluations.<br /><br/>
    We are actively working on developing more seamless integrations that will simplify this process.\
    > While this approach works, expect improvements and updates in the near future that will make working with external datasets more straightforward.

    ## Setup and imports
    First, we initialize Weave and connect to Weights & Biases for tracking experiments.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: datasets wandb weave !pip install datasets wandb weave
    return


@app.cell
def _():
    # Initialize variables
    HUGGINGFACE_DATASET = "wandb/ragbench-test-sample"
    WANDB_KEY = ""
    WEAVE_TEAM = ""
    WEAVE_PROJECT = ""

    # Init weave and required libraries
    import asyncio

    import nest_asyncio
    import wandb
    from datasets import load_dataset

    import weave
    from weave import Evaluation

    # Login to wandb and initialize weave
    wandb.login(key=WANDB_KEY)
    client = weave.init(f"{WEAVE_TEAM}/{WEAVE_PROJECT}")

    # Apply nest_asyncio to allow nested event loops (needed for some notebook environments)
    nest_asyncio.apply()
    return Evaluation, HUGGINGFACE_DATASET, asyncio, load_dataset, weave


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load and prepare HuggingFace dataset

    - We load a HuggingFace dataset.
    - Create an index mapping to reference the dataset rows.
    - This index approach allows us to maintain references to the original dataset.

    > **Note:**<br/>
    In the index, we encode the `hf_hub_name` along with the `hf_id` to ensure each row has a unique identifier.\
    This unique digest value is used for tracking and referencing specific dataset entries during evaluations.
    """)
    return


@app.cell
def _(HUGGINGFACE_DATASET, load_dataset):
    # Load the HuggingFace dataset
    ds = load_dataset(HUGGINGFACE_DATASET)
    row_count = ds["train"].num_rows

    # Create an index mapping for the dataset
    # This creates a list of dictionaries with HF dataset indices
    # Example: [{"hf_id": 0}, {"hf_id": 1}, {"hf_id": 2}, ...]
    hf_index = [{"hf_id": i, "hf_hub_name": HUGGINGFACE_DATASET} for i in range(row_count)]
    return ds, hf_index


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define processing and evaluation functions

    ### Processing pipeline
    - `preprocess_example`: Transforms the index reference into the actual data needed for evaluation
    - `hf_eval`: Defines how to score the model outputs
    - `function_to_evaluate`: The actual function/model being evaluated
    """)
    return


@app.cell
def _(ds, weave):
    @weave.op()
    def preprocess_example(example):
        """
        Preprocesses each example before evaluation.
        Args:
            example: Dict containing hf_id
        Returns:
            Dict containing the prompt from the HF dataset
        """
        hf_row = ds["train"][example["hf_id"]]
        return {"prompt": hf_row["question"], "answer": hf_row["response"]}


    @weave.op()
    def hf_eval(hf_id: int, output: dict) -> dict:
        """
        Scoring function for evaluating model outputs.
        Args:
            hf_id: Index in the HF dataset
            output: The output from the model to evaluate
        Returns:
            Dict containing evaluation scores
        """
        hf_row = ds["train"][hf_id]
        return {"scorer_value": True}


    @weave.op()
    def function_to_evaluate(prompt: str):
        """
        The function that will be evaluated (e.g., your model or pipeline).
        Args:
            prompt: Input prompt from the dataset
        Returns:
            Dict containing model output
        """
        return {"generated_text": "testing "}

    return function_to_evaluate, hf_eval, preprocess_example


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create and run evaluation

    - For each index in hf_index:
      1. `preprocess_example` gets the corresponding data from the HF dataset.
      2. The preprocessed data is passed to `function_to_evaluate`.
      3. The output is scored using `hf_eval`.
      4. Results are tracked in Weave.
    """)
    return


@app.cell
def _(
    Evaluation,
    asyncio,
    function_to_evaluate,
    hf_eval,
    hf_index,
    preprocess_example,
):
    # Create evaluation object
    evaluation = Evaluation(
        dataset=hf_index,  # Use our index mapping
        scorers=[hf_eval],  # List of scoring functions
        preprocess_model_input=preprocess_example,  # Function to prepare inputs
    )


    # Run evaluation asynchronously
    async def main():
        await evaluation.evaluate(function_to_evaluate)


    asyncio.run(main())
    return


if __name__ == "__main__":
    app.run()

