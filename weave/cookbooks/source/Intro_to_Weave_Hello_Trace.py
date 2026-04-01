# /// script
# dependencies = ["jedi", "openai", "weave"]
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
    # Introduction to Traces

    <img src="https://cdn.prod.website-files.com/62ba1fb86485b6d5029975c4/69b1b3d9c77f6e5294374be1_Endorsed_primary_goldblack.png" width="400" alt="Weights & Biases" />

    Weave is a toolkit for developing AI-powered applications.

    Use Weave traces to capture the inputs, outputs, and internal structure of your Python function automatically to observe and debug LLM applications.

    When you decorate a function with `@weave.op`, Weave records a rich trace of the function while it runs, including any nested operations or external API calls. Use the trace to to debug, understand, and visualize interactions between your code and LLM models, without leaving your notebook.

    To get started, complete the prerequisites. Then, define a function decorated with `@weave.op` decorator and run it on an example input to track LLM calls. Weave captures and visualizes the trace automatically.
    """)
    return


@app.cell
def _():
    # Ensure your dependencies are installed with:
    # packages added via marimo's package management: jedi openai weave !pip install --quiet jedi openai weave
    return


@app.cell
def _():
    import os
    import getpass

    #@title Set up your credentials
    inference_provider = "W&B Inference" #@param ["W&B Inference", "OpenAI"]

    # Set up your W&B project and credentials
    os.environ["WANDB_ENTITY_PROJECT"] = input("Set up your W&B project (team name/project name): ")
    os.environ["WANDB_API_KEY"] = getpass.getpass("Set up your W&B API key (Create an API key at https://wandb.ai/settings): ")

    # Set up your OpenAI API key
    if inference_provider == "OpenAI":
      os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key (Find it at https://platform.openai.com/api-keys): ")
    return inference_provider, os


@app.cell
def _(inference_provider, os):
    from openai import OpenAI
    import weave

    weave.init(os.environ["WANDB_ENTITY_PROJECT"])

    @weave.op  # Decorator to track requests
    def create_completion(message: str) -> str:
        if inference_provider == "W&B Inference":
          client = OpenAI(
              base_url="https://api.inference.wandb.ai/v1",
              api_key=os.environ["WANDB_API_KEY"],
              project=os.environ["WANDB_ENTITY_PROJECT"],
          )
          model_name: str = "OpenPipe/Qwen3-14B-Instruct"
        if inference_provider == "OpenAI":
          client = OpenAI()
          model_name: str = "gpt-4.1-nano"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ],
        )
        return response.choices[0].message.content


    message = "Tell me a joke."
    create_completion(message)
    return


if __name__ == "__main__":
    app.run()

