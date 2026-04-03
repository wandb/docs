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
    # Learn Weave with W&B Inference

    This guide shows you how to use W&B Weave with [W&B Inference](https://docs.wandb.ai/inference/). Using W&B Inference, you can build and trace LLM applications using live open-source models without setting up your own infrastructure or managing API keys from multiple providers. Just [create a W&B API key](https://wandb.ai/settings) to interact with [all models hosted by W&B Inference](https://docs.wandb.ai/inference/models/).

    ## What you'll learn

    This guide shows you how to:

    - Set up Weave and W&B Inference
    - Build a basic LLM application with automatic tracing
    - Compare multiple models
    - Evaluate model performance on a dataset
    - View your results in the Weave UI

    ## Install Weave and OpenAI
    """)
    return


app._unparsable_cell(
    r"""
    pip install weave openai
    """,
    name="_"
)


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
    WANDB_API_KEY = input("Enter your W&B API key: ")
    WB_TEAM_NAME = input("Enter your W&B team name: ")
    return WANDB_API_KEY, WB_TEAM_NAME


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Trace your first LLM call

    Start with a basic example that uses Llama 3.1-8B through W&B Inference.

    When you run this code, Weave:
    - Traces your LLM call automatically
    - Logs inputs, outputs, latency, and token usage
    - Provides a link to view your trace in the Weave UI
    """)
    return


@app.cell
def _(WANDB_API_KEY, WB_TEAM_NAME):
    import weave
    import openai
    from google.colab import userdata

    team_project = WB_TEAM_NAME + "/inference-quickstart"

    # Initialize Weave - replace with your-team/your-project
    weave.init(team_project)

    # Create an OpenAI-compatible client pointing to W&B Inference
    client = openai.OpenAI(
        base_url='https://api.inference.wandb.ai/v1',
        api_key= WANDB_API_KEY,  # Set your API as the WANDB_API_KEY environment variable
        project=team_project,  # Required for usage tracking
    )

    # Decorate your function to enable tracing; use the standard OpenAI client
    @weave.op()
    def ask_llama(question: str) -> str:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
        )
        return response.choices[0].message.content

    # Call your function - Weave automatically traces everything
    result = ask_llama("What are the benefits of using W&B Weave for LLM development?")
    print(result)
    return openai, team_project, weave


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Build a text summarization application

    Next, try running this code, which is a basic summarization app that shows how Weave traces nested operations:
    """)
    return


@app.cell
def _(WANDB_API_KEY, openai, team_project, weave):
    weave.init(team_project)
    client_1 = openai.OpenAI(base_url='https://api.inference.wandb.ai/v1', api_key=WANDB_API_KEY, project=team_project)

    # Initialize Weave - replace with your-team/your-project
    @weave.op()
    def extract_key_points(text: str) -> list[str]:
        """Extract key points from a text."""
        response = client_1.chat.completions.create(model='meta-llama/Llama-3.1-8B-Instruct', messages=[{'role': 'system', 'content': 'Extract 3-5 key points from the text. Return each point on a new line.'}, {'role': 'user', 'content': text}])
        return [line for line in response.choices[0].message.content.strip().splitlines() if line.strip()]  # Set your API as the WANDB_API_KEY environment variable
      # Required for usage tracking
    @weave.op()
    def create_summary(key_points: list[str]) -> str:
        """Create a concise summary based on key points."""
        points_text = '\n'.join((f'- {point}' for point in key_points))
        response = client_1.chat.completions.create(model='meta-llama/Llama-3.1-8B-Instruct', messages=[{'role': 'system', 'content': 'Create a one-sentence summary based on these key points.'}, {'role': 'user', 'content': f'Key points:\n{points_text}'}])
        return response.choices[0].message.content

    @weave.op()
    def summarize_text(text: str) -> dict:
        """Main summarization pipeline."""
        key_points = extract_key_points(text)
        summary = create_summary(key_points)
        return {'key_points': key_points, 'summary': summary}  # Returns response without blank lines
    sample_text = '\nThe Apollo 11 mission was a historic spaceflight that landed the first humans on the Moon\non July 20, 1969. Commander Neil Armstrong and lunar module pilot Buzz Aldrin descended\nto the lunar surface while Michael Collins remained in orbit. Armstrong became the first\nperson to step onto the Moon, followed by Aldrin 19 minutes later. They spent about\ntwo and a quarter hours together outside the spacecraft, collecting samples and taking photographs.\n'
    result_1 = summarize_text(sample_text)
    print('Key Points:', result_1['key_points'])
    # Try it with sample text
    print('\nSummary:', result_1['summary'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Compare multiple models

    W&B Inference provides access to multiple models. Use the following code to compare the performance of Llama and DeepSeek's respective responses:
    """)
    return


@app.cell
def _(WANDB_API_KEY, openai, team_project, weave):
    weave.init(team_project)
    client_2 = openai.OpenAI(base_url='https://api.inference.wandb.ai/v1', api_key=WANDB_API_KEY, project=team_project)

    # Initialize Weave - replace with your-team/your-project
    class InferenceModel(weave.Model):
        model_name: str

        @weave.op()
        def predict(self, question: str) -> str:  # Set your API as the WANDB_API_KEY environment variable
            response = client_2.chat.completions.create(model=self.model_name, messages=[{'role': 'user', 'content': question}])  # Required for usage tracking
            return response.choices[0].message.content
    llama_model = InferenceModel(model_name='meta-llama/Llama-3.1-8B-Instruct')
    # Define a Model class to compare different LLMs
    deepseek_model = InferenceModel(model_name='deepseek-ai/DeepSeek-V3-0324')
    test_question = 'Explain quantum computing in one paragraph for a high school student.'
    print('Llama 3.1 8B response:')
    print(llama_model.predict(test_question))
    print('\n' + '=' * 50 + '\n')
    print('DeepSeek V3 response:')
    # Create instances for different models
    # Compare their responses
    print(deepseek_model.predict(test_question))
    return InferenceModel, deepseek_model, llama_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: Evaluate model performance

    Evaluate how well a model performs on a Q&A task using Weave's built-in `EvaluationLogger`. This provides structured evaluation tracking with automatic aggregation, token usage capture, and rich comparison features in the UI.

    Append the following code to the script you used in step 3:
    """)
    return


@app.cell
def _(InferenceModel, deepseek_model, llama_model, weave):
    from typing import Optional
    from weave import EvaluationLogger

    # Create a simple dataset
    dataset = [
        {"question": "What is 2 + 2?", "expected": "4"},
        {"question": "What is the capital of France?", "expected": "Paris"},
        {"question": "Name a primary color", "expected_one_of": ["red", "blue", "yellow"]},
    ]

    # Define a scorer
    @weave.op()
    def accuracy_scorer(expected: str, output: str, expected_one_of: Optional[list[str]] = None) -> dict:
        """Score the accuracy of the model output."""
        output_clean = output.strip().lower()

        if expected_one_of:
            is_correct = any(option.lower() in output_clean for option in expected_one_of)
        else:
            is_correct = expected.lower() in output_clean

        return {"correct": is_correct, "score": 1.0 if is_correct else 0.0}

    # Evaluate a model using Weave's EvaluationLogger
    def evaluate_model(model: InferenceModel, dataset: list[dict]):
        """Run evaluation on a dataset using Weave's built-in evaluation framework."""
        # Initialize EvaluationLogger BEFORE calling the model to capture token usage
        # This is especially important for W&B Inference to track costs
        # Convert model name to a valid format (replace non-alphanumeric chars with underscores)
        safe_model_name = model.model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        eval_logger = EvaluationLogger(
            model=safe_model_name,
            dataset="qa_dataset"
        )

        for example in dataset:
            # Get model prediction
            output = model.predict(example["question"])

            # Log the prediction
            pred_logger = eval_logger.log_prediction(
                inputs={"question": example["question"]},
                output=output
            )

            # Score the output
            score = accuracy_scorer(
                expected=example.get("expected", ""),
                output=output,
                expected_one_of=example.get("expected_one_of")
            )

            # Log the score
            pred_logger.log_score(
                scorer="accuracy",
                score=score["score"]
            )

            # Finish logging for this prediction
            pred_logger.finish()

        # Log summary - Weave automatically aggregates the accuracy scores
        eval_logger.log_summary()
        print(f"Evaluation complete for {model.model_name} (logged as: {safe_model_name}). View results in the Weave UI.")

    # Compare multiple models - a key feature of Weave's evaluation framework
    models_to_compare = [
        llama_model,
        deepseek_model,
    ]

    for model in models_to_compare:
        evaluate_model(model, dataset)

    # In the Weave UI, navigate to the Evals tab to compare results across models
    return


if __name__ == "__main__":
    app.run()

