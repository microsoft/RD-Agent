"""
TODO:
It is not complete now.

Please refer to rdagent/oai/llm_utils.py:APIBackend for the future design
"""

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import openai
from pydantic_settings import BaseSettings


class AzureConf(BaseSettings):
    """
    TODO: move more settings here
    """
    use_azure_token_provider: bool = False
    managed_identity_client_id: str | None = None
    chat_model: str = "gpt-4-turbo"

    chat_azure_api_base: str = ""
    chat_azure_api_version: str = ""


class BaseAPI:
    """
    TOOD: there may be some more shared methods in the BaseAPI
    """
    pass


class AzureAPI(BaseAPI):

    def _get_credential(self):
        dac_kwargs = {}
        if AZURE_CONF.managed_identity_client_id is not None:
            dac_kwargs["managed_identity_client_id"] = self.managed_identity_client_id
        credential = DefaultAzureCredential(**dac_kwargs)
        return credential

    def _get_client(self):
        kwargs = {}
        if AZURE_CONF.use_azure_token_provider:
            kwargs["azure_ad_token_provider"]= get_bearer_token_provider(
                self._get_credential(),
                "https://cognitiveservices.azure.com/.default",
            )
        return openai.AzureOpenAI(
            api_version=AZURE_CONF.chat_azure_api_version,
            azure_endpoint=AZURE_CONF.chat_azure_api_base,
            **kwargs,
        )

    # def list_deployments(self):
    #     client = self._get_client()
    #     try:
    #         deployments = client.deployments.list()
    #         return [deployment for deployment in deployments]
    #     except Exception as e:
    #         print(f"An error occurred while listing deployments: {e}")
    #         return []

AZURE_CONF = AzureConf()


# if __name__ == "__main__":
#     api = AzureAPI()
#     deployments = api.list_deployments()
#     print(deployments)
