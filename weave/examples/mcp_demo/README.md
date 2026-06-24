# MCP + Weave integration example

This example demonstrates an integration between the Model Context Protocol (MCP) and Weave for tracing. It shows how to instrument both the client and server components to capture traces of their interactions. By following this example, you see how Weave captures the interactions between an MCP client and server, giving you visibility into tool calls, resource reads, and prompt usage in your own MCP-based applications.

## Features

This example demonstrates the following capabilities:

- Tools: Functions that clients can call.
- Resources: Static and dynamic data that clients can read.
- Prompts: Templated messages for consistent interactions.

## Files

- `example_server.py`: A demo MCP server with tools, resources, and prompts built with `FastMCP`.
- `example_client.py`: A client that connects to the server and provides a command-line interface to interact with all server capabilities.

## Setup

Follow these steps to install the example and its dependencies before you run it.

1. Clone the repository and set up the environment:

```bash
git clone https://github.com/wandb/weave
cd weave
uv venv
source .venv/bin/activate
uv sync
```

2. Add `OPENAI_API_KEY` to an `.env` file. The example uses this key to authenticate with OpenAI.

```bash
touch examples/mcp_demo/.env
```

3. Install the required dependencies:

```bash
uv pip install -e ".[mcp]"
```

4. Run the example:

```bash
python examples/mcp_demo/example_client.py examples/mcp_demo/example_server.py
```

After the client is running, you can use its interactive command-line interface to exercise the server's tools, resources, and prompts. For a list of available commands, see [Use the interactive client](#use-the-interactive-client).

## Use the interactive client

The client provides a command-line interface to interact with all server features. Use the following commands to explore each capability:

- `tools` - List all available tools.
- `resources` - List all available resources.
- `prompts` - List all available prompts.
- `add <a> <b>` - Add two numbers.
- `bmi <weight_kg> <height_m>` - Calculate BMI.
- `weather <city>` - Get the weather for a city.
- `greeting <name>` - Get a personalized greeting.
- `user <id>` - Get a user profile.
- `config` - Get the application configuration.
- `code-review <code>` - Get a code review prompt.
- `debug <error>` - Get a debug error prompt.
- `demo` - Run demos for all server features.
- `q` - Exit the session.
