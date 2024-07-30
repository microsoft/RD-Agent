=========================
Installation
=========================

Installation
============

For different scenarios
- for purely users:
- for dev users: link to development

Configuration
=============

To run the application, please create a `.env` file in the root directory of the project and add environment variables according to your requirements.

The standard configuration options for the user using the OpenAI API are provided in the `.env.example` file.

Here are some other configuration options that you can use:

OpenAI API
------------

You can use different OpenAI API keys for embedding model and chat model.

   .. code-block:: Properties

      EMBEDDING_OPENAI_API_KEY=<replace_with_your_openai_api_key>
      EMBEDDING_MODEL=text-embedding-3-small

      CHAT_OPENAI_API_KEY=<replace_with_your_openai_api_key>
      CHAT_MODEL=gpt-4-turbo

Azure OpenAI
------------

The following environment variables are standard configuration options for the user using the OpenAI API.

   .. code-block:: Properties
      
      USE_AZURE=True

      OPENAI_API_KEY=<replace_with_your_openai_api_key>
      
      EMBEDDING_MODEL=text-embedding-3-small
      EMBEDDING_AZURE_API_BASE= # The base URL for the Azure OpenAI API.
      EMBEDDING_AZURE_API_VERSION = # The version of the Azure OpenAI API.

      CHAT_MODEL=gpt-4-turbo
      CHAT_AZURE_API_VERSION = # The version of the Azure OpenAI API.

Use Azure Token Provider
------------------------

If you are using the Azure token provider, you need to set the `USE_AZURE_TOKEN_PROVIDER` environment variable to `True`. then 
use the environment variables provided in the `Azure Configuration section <installation.html#azure-openai>`_.

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

