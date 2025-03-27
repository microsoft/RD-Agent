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

LiteLLM Backend Configuration
=============================

Please create a `.env` file in the root directory of the project and add environment variables.

Here is a sample configuration for using OpenAI's gpt-4o via LiteLLM. 

   .. code-block:: Properties

      BACKEND=rdagent.oai.backend.LiteLLMAPIBackend
      # It can be modified to any model supported by LiteLLM.
      CHAT_MODEL=gpt-4o
      EMBEDDING_MODEL=text-embedding-3-small
      # The backend api_key fully follows the convention of litellm.
      OPENAI_API_KEY=<replace_with_your_openai_api_key>

Necessary parameters include:

- `BACKEND`: The backend to use. The default is `rdagent.oai.backend.DeprecBackend`. To use the LiteLLM backend, set it to `rdagent.oai.backend.LiteLLMAPIBackend`.

- `CHAT_MODEL`: The model name of the chat model. 

- `EMBEDDING_MODEL`: The model name of the embedding model.

The `CHAT_MODEL` and `EMBEDDING_MODEL` parameters will be passed into LiteLLM's completion function. 

Therefore, when utilizing models provided by different providers, first review the interface configuration of LiteLLM. The model names must match those allowed by LiteLLM.

Additionally, you need to set up the the additional parameters for the respective model provider, and the parameter names must align with those required by LiteLLM.

For example, if you are using a DeepSeek model, you need to set as follows:

   .. code-block:: Properties

      # For some models LiteLLM requires a prefix to the model name.
      CHAT_MODEL=deepseek/deepseek-chat
      DEEPSEEK_API_KEY=<replace_with_your_deepseek_api_key>

For more details on LiteLLM requirements, refer to the `official LiteLLM documentation <https://docs.litellm.ai/docs>`_.


Configuration(deprecated)
=========================

To run the application, please create a `.env` file in the root directory of the project and add environment variables according to your requirements.

The standard configuration options for the user using the OpenAI API are provided in the `.env.example` file.

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

