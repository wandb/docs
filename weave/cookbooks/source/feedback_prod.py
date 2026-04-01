# /// script
# dependencies = ["openai", "set-env-colab-kaggle-dotenv", "streamlit", "wandb", "weave"]
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
    title: Log User Feedback from Production
    ---
    docusaurus_head_meta::end -->

    <!--- @wandbcode{feedback-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is often hard to automatically evaluate a generated LLM response so, depending on your risk tolerance, you can gather direct user feedback to find areas to improve.

    In this tutorial, we'll use a custom chatbot as an example app from which to collect user feedback.
    We'll use Streamlit to build the interface and we'll capture the LLM interactions and feedback in Weave.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: weave openai streamlit wandb !pip install weave openai streamlit wandb
    # packages added via marimo's package management: set-env-colab-kaggle-dotenv !pip install set-env-colab-kaggle-dotenv -q
    # for env var
    return


@app.cell
def _():
    # Add a .env file with your OpenAI and WandB API keys
    from set_env import set_env

    _ = set_env("OPENAI_API_KEY")
    _ = set_env("WANDB_API_KEY")
    return (set_env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, create a file called `chatbot.py` with the following contents:
    """)
    return


@app.cell
def _(set_env):
    # chatbot.py
    import openai
    import streamlit as st
    import wandb
    import weave
    _ = set_env('OPENAI_API_KEY')
    _ = set_env('WANDB_API_KEY')
    wandb.login()
    weave_client = weave.init('feedback-example')
    oai_client = openai.OpenAI()

    def init_states():
        """Set up session_state keys if they don't exist yet."""
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        if 'calls' not in st.session_state:
            st.session_state['calls'] = []
        if 'session_id' not in st.session_state:
            st.session_state['session_id'] = '123abc'

    @weave.op
    def chat_response(full_history):
        """
        Calls the OpenAI API in streaming mode given the entire conversation history so far.
        full_history is a list of dicts: [{"role":"user"|"assistant","content":...}, ...]
        """
        stream = oai_client.chat.completions.create(model='gpt-4', messages=full_history, stream=True)
        response_text = st.write_stream(stream)
        return {'response': response_text}

    def render_feedback_buttons(call_idx):
        """Renders thumbs up/down and text feedback for the call."""
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button('👍', key=f'thumbs_up_{call_idx}'):
                st.session_state.calls[call_idx].feedback.add_reaction('👍')
                st.success('Thanks for the feedback!')
        with col2:
            if st.button('👎', key=f'thumbs_down_{call_idx}'):
                st.session_state.calls[call_idx].feedback.add_reaction('👎')
                st.success('Thanks for the feedback!')
        with col3:
            feedback_text = st.text_input('Feedback', key=f'feedback_input_{call_idx}')
            if st.button('Submit Feedback', key=f'submit_feedback_{call_idx}') and feedback_text:
                st.session_state.calls[call_idx].feedback.add_note(feedback_text)
                st.success('Feedback submitted!')  # Thumbs up button

    def display_old_messages():
        """Displays the conversation stored in st.session_state.messages with feedback buttons"""
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message['role']):
                st.markdown(message['content'])  # Thumbs down button
                if message['role'] == 'assistant':
                    assistant_idx = len([m for m in st.session_state.messages[:idx + 1] if m['role'] == 'assistant']) - 1
                    if assistant_idx < len(st.session_state.calls):
                        render_feedback_buttons(assistant_idx)

    def display_chat_prompt():  # Text feedback
        """Displays the chat prompt input box."""
        if (prompt := st.chat_input('Ask me anything!')):
            with st.chat_message('user'):
                st.markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            full_history = [{'role': msg['role'], 'content': msg['content']} for msg in st.session_state.messages]
            with st.chat_message('assistant'):
                with weave.attributes({'session': st.session_state['session_id'], 'env': 'prod'}):
                    result, call = chat_response.call(full_history)
                    st.session_state.messages.append({'role': 'assistant', 'content': result['response']})
                    st.session_state.calls.append(call)
                    new_assistant_idx = len([m for m in st.session_state.messages if m['role'] == 'assistant']) - 1
                    if new_assistant_idx < len(st.session_state.calls):
                        render_feedback_buttons(new_assistant_idx)

    def main():
        st.title('Chatbot with immediate feedback forms')  # If it's an assistant message, show feedback form
        init_states()
        display_old_messages()  # Figure out index of this assistant message in st.session_state.calls
        display_chat_prompt()
    if __name__ == '__main__':
        main()  # Render thumbs up/down & text feedback  # Immediately render new user message  # Save user message in session  # Prepare chat history for the API  # Attach Weave attributes for tracking of conversation instances  # Call the OpenAI API (stream)  # Store the assistant message  # Store the weave call object to link feedback to the specific response  # Render feedback buttons for the new message  # Render feedback buttons
    return (weave,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can run this with `streamlit run chatbot.py`.

    Now, you can interact with this application and click the feedback buttons after each response.
    Visit the Weave UI to see the attached feedback.

    ## Explanation

    If we consider our decorated prediction function as:
    """)
    return


@app.cell
def _(weave):
    weave.init('feedback-example')

    @weave.op
    def predict(input_data):
        some_result = 'hello world'
        return some_result  # Your prediction logic here

    return (predict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use it as usual to deliver some model response to the user:
    """)
    return


@app.cell
def _(predict, weave):
    with weave.attributes(
        {"session": "123abc", "env": "prod"}
    ):  # attach arbitrary attributes to the call alongside inputs & outputs
        result = predict(input_data="your data here")  # user question through the App UI
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To attach feedback, you need the `call` object, which is obtained by using the `.call()` method _instead of calling the function as normal_:
    """)
    return


@app.cell
def _(predict):
    result_1, call = predict.call(input_data='your data here')
    return (call,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This call object is needed for attaching feedback to the specific response.
    After making the call, the output of the operation is available using `result` above.
    """)
    return


@app.cell
def _(call):
    call.feedback.add_reaction("👍")  # user reaction through the App UI
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    In this tutorial, we built a chat UI with Streamlit which had inputs & outputs captured in Weave, alongside 👍👎 buttons to capture user feedback.
    """)
    return


if __name__ == "__main__":
    app.run()

