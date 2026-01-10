# Codelab: Using Google Gemini File Search Tool for RAG

# About this Repo

- This repo: [gemini-file-search-demo](https://github.com/derailed-dash/gemini-file-search-demo)
- Author: Darren "Dazbo" Lester
- Created: 2024-01-10

## Key Links

- [My related blog - Using Gemini File Search Tool for RAG (a Rickbot Blog)](https://medium.com/google-cloud/using-gemini-file-search-tool-for-rag-a-rickbot-blog-b6c4f117e5d3).
- [Google Codelabs](https://codelabs.developers.google.com/)
- [This Codelab](https://codelabs.developers.google.com/gemini-file-search-for-rag) - This does not yet exist.

# Introduction

This codelab will teach you how to use the Gemini File Search Tool for RAG.

## What You'll Learn

- ✅ The basics of RAG and why we need it.
- ✅ What Gemini File Search is and its advantages.
- ✅ How to create a File Search Store.
- ✅ How to use the Gemini File Search Tool for RAG.
- ✅ How to use the Gemini File Search Tool alongside Google "native" tools like Google Search.
- ✅ How to use the Gemini File Search Tool in an agentic solution build using the Google Agent Development Kit (ADK).

## What You'll Do

1. ✅ Create a Google Cloud Project and setup your development environment.
2. ✅ Create a simple Gemini-based agent using the Google Gen AI SDK (but without ADK) that has the ability to use Google search, but no RAG capability.
3. ✅ Demonstrate its _inability_ to provide accurate, high quality information for bespoke information.
4. ✅ Create a Jupyter notebook (which you can run locally, or, say, on Google Colab) for creating and managing a Gemini File Search Store.
5. ✅ Use the notebook to upload bespoke content to the File Search Store.
6. ✅ Create an agent that has the File Search Store attached, and prove it is able to produce better responses.
7. ✅ Convert our initial "basic" agent to an ADK agent, complete with Google Search tool.
8. ✅ Test the agent using ADK Web UI.
9. ✅ Incorporate the File Search Store into the ADK agent, using the Agent-As-A-Tool pattern to allow us to use the File Search Tool alongside the Google Search tool.

# What is RAG and Why We Need It

So... RAG. **Retrieval Augmented Generation**.

If you're here, you probably know what it is, but let's do a quick recap, just in case. LLMs (like Gemini) are brilliant, but they suffer from two main issues:

1.  **They are frozen in time**: They only know what they learned during training.
2.  **They don't know your proprietary information**: They haven't read your internal documents, your blogs, or your Jira tickets.

RAG is simply the process of looking up relevant information from *your* data source and feeding it to the model alongside the user's question. It grounds the model in your reality.

Usually, this involves:
*   Spinning up a Vector Database (Pinecone, Weaviate, Postgres with pgvector...).
*   Writing a chunking script to slice up your documents (e.g. PDFs, markdown, whatever).
*   Generating embeddings (vectors) for those chunks, using an embedding model.
*   Storing the vectors in the Vector Database.

But friends don't let friends over-engineer things. What if I told you there's an easier way?

# Prerequisites

Let's get the boring stuff out of the way. You can't build a spaceship without a wrench.

## Create a Google Cloud Project

You need a Google Cloud Project to run this codelab. You can use a project you already have, or [create a new one](https://console.cloud.google.com/projectcreate). 

Make sure [billing](https://console.cloud.google.com/billing) is enabled on your project. See [this guide](https://docs.cloud.google.com/billing/docs/how-to/verify-billing-enabled) to see how to check billing status of your projects.

_Note that completing this codelab is not expected to cost you anything. At most, a few pennies._

Go ahead and get your project ready. I'll wait.

## Clone the Repo

I have created a repo with guided content for this codelab. You're going to need it!

You will need to run this from your terminal, or from Cloud Shell with its integrated [Cloud Shell Editor](https://ide.cloud.google.com/). Cloud Shell is very convenient, as it has all the commands you need pre-installed and everything here just runs "out-of-the-box".

```bash
git clone https://github.com/derailed-dash/gemini-file-search-demo
cd gemini-file-search-demo
```

If you're not using Cloud Shell, go ahead and open this folder in your favourite editor. (Have you used Antigravity yet? If not, now would be a good time to [try it out](https://medium.com/google-cloud/tutorial-getting-started-with-google-antigravity-b5cc74c103c2).) 

## Setup Your Dev Environment

I would recommend using `uv` for package management because it's so much easier and faster than `pip`. 

For convenience, I've provided a `Makefile` to simplify many of the commands you need to run. Instead of remembering specific commands, you can just run something like `make <target>`. However, `make` is only available in Linux / MacOS / WSL environments. If you're using Windows, you'll need to run the full commands that the `make` targets contain.

```bash
# Install dependencies
make install

# If you don't have make...
uv sync --extra jupyter
```

This is what it looks like, if you run `make install` in the Cloud Shell Editor:

![Cloud Shell Editor: make install](media/make-install.png)

## Create a Gemini API Key

To use the Gemini Developer API (which we need to use the Gemini File Search Tool), you need a **Gemini API key**. The easiest way to get an API key is to use the [Google AI Studio](https://aistudio.google.com/), which provides a convenient interface to associate your Google Cloud project(s) and then generate an API key for any given project.

See [this guide](https://ai.google.dev/gemini-api/docs/api-key) for the specific steps.

Once your API key is created, copy it and **keep it safe**.

You now need to set this API key as an environment variable. Copy the included `.env.example` as a new file called `.env`. The file should look like this:

```
export GEMINI_API_KEY="your-api-key"
export MODEL="gemini-2.5-flash"
export STORE_NAME="demo-file-store"
```

Go ahead and replace `your-api-key` with your actual API key. Now it should look something like this:

![Cloud Shell Editor: make install](media/env-api.png)

# The Basic Agent

First, let's establish a baseline. We're going to use the raw `google-genai` SDK to run a simple agent.

## The Code

Take a look at `app/sdk_agent.py`. It's a minimal implementation that:

- Instantiates a `genai.Client`.
- Enables the `types.GoogleSearch()` tool.
- That's it. No RAG.

Have a look through the code and make sure you understand what it does.

## Running It

```bash
# With make
make sdk-agent

# Without make
uv run python app/sdk_agent.py
```

Let's ask it a general question:
> `What is the stock price of Google?`

It should answer correctly using Google Search.

Now, let's ask it a question it doesn't know how to answer:
> `Who pilots the 'Too Many Pies' ship?`

It will fail. It might hallucinate, or it might admit defeat. It definitely won't know the answer, because that information exists only in my head (and a file in this repo).

# Gemini File Search: Explained

This is where **Gemini File Search** comes in.

It is a **managed RAG service**.
- **You upload files**: Gemini accepts PDFs, Markdown, CSVs, etc.
- **Gemini handles the plumbing**: It chunks them, embeds them, and stores the vectors.
- **You use it as a tool**: You just pass the store name to the model.

It features a **1 Million Token Context Window** and **Context Caching** (which makes it surprisingly cheap for repeated queries).

# Jupyter Notebook to Create and Manage a Gemini File Search Store

Since managing your knowledge base is an "admin" task, we shouldn't jam that code into our runtime agent. We use a Jupyter Notebook.

### The Notebook (`notebooks/file_search_store.ipynb`)
Open this file. I've designed it to:
1.  Authenticate with your Gemini API Key.
2.  Create a **File Search Store** (a container for your docs).
3.  Upload files.

### The Data (`data/story.md`)
We are uploading `data/story.md`. It's a gripping sci-fi yarn called *The Wormhole Incursion*.
It contains specific facts:
- **Too Many Pies**: An Anaconda class ship.
- **Relativistic Value**: A Cobra Mk3.
- **Attitude Adjuster**: A Corvette.

**Action:** Run the notebook cells to create your store and upload the story. **Make a note of the `STORE_NAME` generated by the notebook.**

# Implement Gemini File Search RAG in our Agent

Now we have a store, let's access it.

### The "Tool-as-an-Agent" Pattern (`app/sdk_rag_agent.py`)
Here's a "gotcha" to watch out for. At the time of writing, you **cannot** use the native `GoogleSearch` tool and the `FileSearch` tool in the same request. You get a `400 INVALID_ARGUMENT` error.

**The Solution:**
We use a pattern I call **Tool-as-an-Agent**.
1.  We define a Python function `google_search_tool(query)`.
2.  Inside this function, we spin up a *sub-agent* (a fresh `genai.Client`).
3.  This sub-agent uses the native Google Search tool to get the verified answer.
4.  Our main agent treats this function as just a regular user tool, so it plays nicely with `FileSearch`.

Groovy, right?

### Running It
Set your environment variables (create a `.env` file):
```bash
STORE_NAME=your-store-name-from-notebook
GEMINI_API_KEY=your-key
```

Run the agent:
```bash
make sdk-rag-agent
```

Ask it again:
> "Who pilots the 'Too Many Pies' ship?"

**Success!** It should now tell you it's an Anaconda, referencing information from "File Search".

# Convert our Agent to an ADK Agent

Scripts are fine for demos, but for production, we want structure. We want the **Google Agent Development Kit (ADK)**.

### The Basic ADK Agent (`app/basic_agent_adk/agent.py`)
This mimics our first agent but uses ADK classes.
- **SearchAgent**: An `Agent` configured with `google_search`.
- **RootAgent**: Delegates to the `SearchAgent`.
- **Performance**: It uses "fail-fast" instructions to stop it from spiralling if it can't find an answer.

Run it with the ADK playground:
```bash
make adk-playground
```

# Incorporate the File Search Store into the ADK agent

Now for the final form. A hierarchical agent using ADK + RAG.

### The ADK RAG Agent (`app/rag_agent_adk/agent.py`)
We have a root agent that orchestrates two specialists:
1.  **RagAgent**: Your bespoke knowledge expert.
2.  **SearchAgent**: Your general knowledge expert.

### The Custom Tool (`app/rag_agent_adk/tools_custom.py`)
Since ADK doesn't have a built-in wrapper for `FileSearch` yet, I wrote a custom middleware class `FileSearchTool`. It injects the `file_search_store_names` configuration into the low-level model request.

**Reflecting on this...**
This took me a few tries to get right. Intercepting the request object sounds complicated, but it's actually the robust way to handle API features that haven't made it into the SDK wrapper layer yet.

### Final Test
1.  Ensure your `STORE_NAME` is set in the `.env` file!
2.  Run the playground:
    ```bash
    make adk-playground
    ```
3.  Switch to the `rag_agent_adk` in the UI.
4.  Ask about "Too Many Pies". Observe the `RagAgent` taking the lead.
5.  Ask about "Google Stock". Observe the `SearchAgent` taking over.

# Conclusion

We've gone from a simple script to a fully agentic, RAG-enabled system without touching a vector database.

We learned:
- **Managed RAG** saves time and sanity.
- **The "Tool-as-an-Agent" pattern** solves tool compatibility issues.
- **ADK** gives us the structure we need for complex apps.

Now, if you'll excuse me, I'm going to find a pina colada. Or maybe just a cup of tea.

# Appendix (Ultimately to be deleted from this file)

## Running Google Search and Gemini File Search together:

Error without the Agent-As-A-Tool pattern:

```text
ERROR - An error occurred: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Search as a tool and file search tool are not supported together', 'status': 'INVALID_ARGUMENT'}}
Error: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Search as a tool and file search tool are not supported together', 'status': 'INVALID_ARGUMENT'}}
```