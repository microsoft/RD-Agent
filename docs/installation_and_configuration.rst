==============================
Installation and Configuration
==============================

Installation
============

**Install RDAgent**: For different scenarios

- for purely users: please use ``pip install rdagent`` to install RDAgent
- for dev users: `See development <development.html>`_

**Install Docker**: RDAgent is designed for research and development, acting like a human researcher and developer. It can write and run code in various environments, primarily using Docker for code execution. This keeps the remaining dependencies simple. Users must ensure Docker is installed before attempting most scenarios. Please refer to the `official 🐳Docker page <https://docs.docker.com/engine/install/>`_ for installation instructions.
Ensure the current user can run Docker commands **without using sudo**. You can verify this by executing `docker run hello-world`.

LiteLLM Backend Configuration (Default)
=======================================

.. note::
   🔥 **Attention**: We now provide experimental support for **DeepSeek** models! You can use DeepSeek's official API for cost-effective and high-performance inference. See the configuration example below for DeepSeek setup.

Option 1: Unified API base for both models
------------------------------------------

   .. code-block:: Properties

      # Set to any model supported by LiteLLM.
      CHAT_MODEL=gpt-4o 
      EMBEDDING_MODEL=text-embedding-3-small
      # Configure unified API base
      # The backend api_key fully follows the convention of litellm.
      OPENAI_API_BASE=<your_unified_api_base>
      OPENAI_API_KEY=<replace_with_your_openai_api_key>

Option 2: Separate API bases for Chat and Embedding models
----------------------------------------------------------

   .. code-block:: Properties

      # Set to any model supported by LiteLLM.
      
      # CHAT MODEL:
      CHAT_MODEL=gpt-4o 
      OPENAI_API_BASE=<your_chat_api_base>
      OPENAI_API_KEY=<replace_with_your_openai_api_key>

      # EMBEDDING MODEL:
      # TAKE siliconflow as an example, you can use other providers.
      # Note: embedding requires litellm_proxy prefix
      EMBEDDING_MODEL=litellm_proxy/BAAI/bge-large-en-v1.5
      LITELLM_PROXY_API_KEY=<replace_with_your_siliconflow_api_key>
      LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1

Configuration Example: DeepSeek Setup
-------------------------------------

Many users encounter configuration errors when setting up DeepSeek. Here's a complete working example:

   .. code-block:: Properties

      # CHAT MODEL: Using DeepSeek Official API
      CHAT_MODEL=deepseek/deepseek-chat 
      DEEPSEEK_API_KEY=<replace_with_your_deepseek_api_key>

      # EMBEDDING MODEL: Using SiliconFlow for embedding since DeepSeek has no embedding model.
      # Note: embedding requires litellm_proxy prefix
      EMBEDDING_MODEL=litellm_proxy/BAAI/bge-m3
      LITELLM_PROXY_API_KEY=<replace_with_your_siliconflow_api_key>
      LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1

Necessary parameters include:

- `CHAT_MODEL`: The model name of the chat model.

- `EMBEDDING_MODEL`: The model name of the embedding model.

- `OPENAI_API_BASE`: The base URL of the API. If `EMBEDDING_MODEL` does not start with `litellm_proxy/`, this is used for both chat and embedding models; otherwise, it is used for `CHAT_MODEL` only.

Optional parameters (required if your embedding model is provided by a different provider than `CHAT_MODEL`):

- `LITELLM_PROXY_API_KEY`: The API key for the embedding model, required if `EMBEDDING_MODEL` starts with `litellm_proxy/`.

- `LITELLM_PROXY_API_BASE`: The base URL for the embedding model, required if `EMBEDDING_MODEL` starts with `litellm_proxy/`.

**Note:** If you are using an embedding model from a provider different from the chat model, remember to add the `litellm_proxy/` prefix to the `EMBEDDING_MODEL` name.


The `CHAT_MODEL` and `EMBEDDING_MODEL` parameters will be passed into LiteLLM's completion function. 

Therefore, when utilizing models provided by different providers, first review the interface configuration of LiteLLM. The model names must match those allowed by LiteLLM.

Additionally, you need to set up the the additional parameters for the respective model provider, and the parameter names must align with those required by LiteLLM.

For example, if you are using a DeepSeek model, you need to set as follows:

   .. code-block:: Properties

      # For some models LiteLLM requires a prefix to the model name.
      CHAT_MODEL=deepseek/deepseek-chat
      DEEPSEEK_API_KEY=<replace_with_your_deepseek_api_key>

Besides, when you are using reasoning models, the response might include the thought process. For this case, you need to set the following environment variable:
   
   .. code-block:: Properties
      
      REASONING_THINK_RM=True

For more details on LiteLLM requirements, refer to the `official LiteLLM documentation <https://docs.litellm.ai/docs>`_.


Configuration(deprecated)
=========================

To run the application, please create a `.env` file in the root directory of the project and add environment variables according to your requirements.

If you are using this deprecated version,  you should set `BACKEND` to `rdagent.oai.backend.DeprecBackend`.

   .. code-block:: Properties

      BACKEND=rdagent.oai.backend.DeprecBackend

Here are some other configuration options that you can use:

OpenAI API
------------

Here is a standard configuration for the user using the OpenAI API.

   .. code-block:: Properties

      OPENAI_API_KEY=<your_api_key>
      EMBEDDING_MODEL=text-embedding-3-small
      CHAT_MODEL=gpt-4-turbo

Azure OpenAI
------------

The following environment variables are standard configuration options for the user using the OpenAI API.

   .. code-block:: Properties

      USE_AZURE=True

      EMBEDDING_OPENAI_API_KEY=<replace_with_your_azure_openai_api_key>
      EMBEDDING_AZURE_API_BASE=  # The endpoint for the Azure OpenAI API.
      EMBEDDING_AZURE_API_VERSION=  # The version of the Azure OpenAI API.
      EMBEDDING_MODEL=text-embedding-3-small

      CHAT_OPENAI_API_KEY=<replace_with_your_azure_openai_api_key>
      CHAT_AZURE_API_BASE=  # The endpoint for the Azure OpenAI API.
      CHAT_AZURE_API_VERSION=  # The version of the Azure OpenAI API.
      CHAT_MODEL=  # The model name of the Azure OpenAI API.

Use Azure Token Provider
------------------------

If you are using the Azure token provider, you need to set the `CHAT_USE_AZURE_TOKEN_PROVIDER` and `EMBEDDING_USE_AZURE_TOKEN_PROVIDER` environment variable to `True`. then 
use the environment variables provided in the `Azure Configuration section <installation_and_configuration.html#azure-openai>`_.


☁️ Azure Configuration
- Install Azure CLI:

   ```sh
   curl -L https://aka.ms/InstallAzureCli | bash
   ```

- Log in to Azure:

   ```sh
   az login --use-device-code
   ```

- `exit` and re-login to your environment (this step may not be necessary).


Configuration List
------------------

.. TODO: use `autodoc-pydantic` .

- OpenAI API Setting

+-----------------------------------+-----------------------------------------------------------------+-------------------------+
| Configuration Option              | Meaning                                                         | Default Value           |
+===================================+=================================================================+=========================+
| OPENAI_API_KEY                    | API key for both chat and embedding models                      | None                    |
+-----------------------------------+-----------------------------------------------------------------+-------------------------+
| EMBEDDING_OPENAI_API_KEY          | Use a different API key for embedding model                     | None                    |
+-----------------------------------+-----------------------------------------------------------------+-------------------------+
| CHAT_OPENAI_API_KEY               | Set to use a different API key for chat model                   | None                    |
+-----------------------------------+-----------------------------------------------------------------+-------------------------+
| EMBEDDING_MODEL                   | Name of the embedding model                                     | text-embedding-3-small  |
+-----------------------------------+-----------------------------------------------------------------+-------------------------+
| CHAT_MODEL                        | Name of the chat model                                          | gpt-4-turbo             |
+-----------------------------------+-----------------------------------------------------------------+-------------------------+
| EMBEDDING_AZURE_API_BASE          | Base URL for the Azure OpenAI API                               | None                    |
+-----------------------------------+-----------------------------------------------------------------+-------------------------+
| EMBEDDING_AZURE_API_VERSION       | Version of the Azure OpenAI API                                 | None                    |
+-----------------------------------+-----------------------------------------------------------------+-------------------------+
| CHAT_AZURE_API_BASE               | Base URL for the Azure OpenAI API                               | None                    |
+-----------------------------------+-----------------------------------------------------------------+-------------------------+
| CHAT_AZURE_API_VERSION            | Version of the Azure OpenAI API                                 | None                    |
+-----------------------------------+-----------------------------------------------------------------+-------------------------+
| USE_AZURE                         | True if you are using Azure OpenAI                              | False                   |
+-----------------------------------+-----------------------------------------------------------------+-------------------------+
| CHAT_USE_AZURE_TOKEN_PROVIDER     | True if you are using an Azure Token Provider in chat model     | False                   |
+-----------------------------------+-----------------------------------------------------------------+-------------------------+
| EMBEDDING_USE_AZURE_TOKEN_PROVIDER| True if you are using an Azure Token Provider in embedding model| False                   |
+-----------------------------------+-----------------------------------------------------------------+-------------------------+

- Globol Setting

+-----------------------------+--------------------------------------------------+-------------------------+
| Configuration Option        | Meaning                                          | Default Value           |
+=============================+==================================================+=========================+
| max_retry                   | Maximum number of times to retry                 | 10                      |
+-----------------------------+--------------------------------------------------+-------------------------+
| retry_wait_seconds          | Number of seconds to wait before retrying        | 1                       |
+-----------------------------+--------------------------------------------------+-------------------------+
+ log_trace_path              | Path to log trace file                           | None                    |
+-----------------------------+--------------------------------------------------+-------------------------+
+ log_llm_chat_content        | Flag to indicate if chat content is logged       | True                    |
+-----------------------------+--------------------------------------------------+-------------------------+


- Cache Setting

.. TODO: update Meaning for caches

+------------------------------+--------------------------------------------------+-------------------------+
| Configuration Option         | Meaning                                          | Default Value           |
+==============================+==================================================+=========================+
| dump_chat_cache              | Flag to indicate if chat cache is dumped         | False                   |
+------------------------------+--------------------------------------------------+-------------------------+
| dump_embedding_cache         | Flag to indicate if embedding cache is dumped    | False                   |
+------------------------------+--------------------------------------------------+-------------------------+
| use_chat_cache               | Flag to indicate if chat cache is used           | False                   |
+------------------------------+--------------------------------------------------+-------------------------+
| use_embedding_cache          | Flag to indicate if embedding cache is used      | False                   |
+------------------------------+--------------------------------------------------+-------------------------+
| prompt_cache_path            | Path to prompt cache                             | ./prompt_cache.db       |
+------------------------------+--------------------------------------------------+-------------------------+
| max_past_message_include     | Maximum number of past messages to include       | 10                      |
+------------------------------+--------------------------------------------------+-------------------------+




Loading Configuration
---------------------

For users' convenience, we provide a CLI interface called `rdagent`, which automatically runs `load_dotenv()` to load environment variables from the `.env` file.
However, this feature is not enabled by default for other scripts. We recommend users load the environment with the following steps:


- ⚙️ Environment Configuration
    - Place the `.env` file in the same directory as the `.env.example` file.
        - The `.env.example` file contains the environment variables required for users using the OpenAI API (Please note that `.env.example` is an example file. `.env` is the one that will be finally used.)

    - Export each variable in the .env file:

      .. code-block:: sh

          export $(grep -v '^#' .env | xargs)
    
    - If you want to change the default environment variables, you can refer to the above configuration and edith the `.env` file.

