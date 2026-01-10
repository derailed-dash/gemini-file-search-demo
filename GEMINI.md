# gemini-file-search-demo

This repo is intended to provide a hands-on demo and walkthrough of using the Gemini File Search Tool for RAG. The demo will be consumable as a Google Codelab.

It is based on my blog [Using Gemini File Search Tool for RAG (a Rickbot Blog)](https://medium.com/google-cloud/using-gemini-file-search-tool-for-rag-a-rickbot-blog-b6c4f117e5d3). (This is a draft.)

## Key Links

- [This Repo](https://github.com/derailed-dash/gemini-file-search-demo)
- [My related blog - Using Gemini File Search Tool for RAG (a Rickbot Blog)](https://medium.com/google-cloud/using-gemini-file-search-tool-for-rag-a-rickbot-blog-b6c4f117e5d3).
- [Google Codelabs](https://codelabs.developers.google.com/)
- [This Codelab](https://codelabs.developers.google.com/gemini-file-search-for-rag) - This does not yet exist.

## Structure of the Demo

1. Demonstrate a Gemini-based agent (no ADK) that has a search tool, but no RAG. Demonstrate its inability to provide accurate, high quality information for bespoke information.
1. Demonstrate a Jupyter notebook (e.g. with Google Colab) for generating a File Search Store, and for uploading bespoke content.
1. Demonstrate adding the File Search Store to the agent, and prove it is able to produce better reesponses.
1. Convert our agent to an ADK agent, and incorporate the File Search Store. Note that attaching the tool is not quite so trivial.
1. Demonstrate the ADK agent, using "ADK web" UI.

## Plan

- This repo will be cloneable as part of the codelab.
- It will contain sample agents at various stages of development. (I will not be making use of the Rickbot agents referenced in the blog.)
- It will contain a sample Jupyter notebook for generating File Search Stores, as well as managing and deleting stores and store contents.

## Rules

- The `README.md` will provide identical content to the Codelab walkthrough.
- Each H1 of the `README.md` will represent the next section of the codelab.

## Example Codelabs

Here are some other codelabs that can be used for inspiration in terms of structure and guidance. Read these and use them for inspiration and guidance. However, I will use my own conversational style.

- [Create multi agent system with ADK, deploy in Agent Engine and get started with A2A protocol](https://codelabs.developers.google.com/codelabs/create-multi-agents-adk-a2a)
- [Developing LLM Apps with the Vertex AI SDK](https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/1-developing-apps-that-use-llms/developing-LLM-apps-with-Vertex-AI-SDK)
- [ADK:From Basics to Multi-Tool Agents](https://codelabs.developers.google.com/multi-tools-ai-agent-adk)
- [Build your own "Bargaining Shopkeeper" Agent with Gemini 3 and ADK](https://codelabs.developers.google.com/agentic-app-gemini-3-adk)
- [Building Agents with Retrieval-Augmented Generation](https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/7-advanced-agent-capabilities/building-agents-with-retrieval-augmented-generation?hl=en#0)