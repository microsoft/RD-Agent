import os
import socket

import docker
import fire
import litellm
import typer
from litellm import completion, embedding
from litellm.utils import ModelResponse
from typing_extensions import Annotated

from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import cleanup_container


def check_docker_status() -> None:
    container = None
    try:
        client = docker.from_env()
        client.images.pull("hello-world")
        container = client.containers.run("hello-world", detach=True)
        logs = container.logs().decode("utf-8")
        print(logs)
        logger.info(f"The docker status is normal")
    except docker.errors.DockerException as e:
        logger.error(f"An error occurred: {e}")
        logger.warning(
            f"Docker status is exception, please check the docker configuration or reinstall it. Refs: https://docs.docker.com/engine/install/ubuntu/."
        )
    finally:
        cleanup_container(container, "health check")


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def check_and_list_free_ports(start_port=19899, max_ports=10) -> None:
    is_occupied = is_port_in_use(port=start_port)
    if is_occupied:
        free_ports = []
        for port in range(start_port, start_port + max_ports):
            if not is_port_in_use(port):
                free_ports.append(port)
        logger.warning(
            f"Port 19899 is occupied, please replace it with an available port when running the `rdagent ui` command. Available ports: {free_ports}"
        )
    else:
        logger.info(f"Port 19899 is not occupied, you can run the `rdagent ui` command")


def test_chat(chat_model, chat_api_key, chat_api_base):
    logger.info(f"🧪 Testing chat model: {chat_model}")
    try:
        if chat_api_base is None:
            response: ModelResponse = completion(
                model=chat_model,
                api_key=chat_api_key,
                messages=[
                    {"role": "user", "content": "Hello!"},
                ],
            )
        else:
            response: ModelResponse = completion(
                model=chat_model,
                api_key=chat_api_key,
                api_base=chat_api_base,
                messages=[
                    {"role": "user", "content": "Hello!"},
                ],
            )
        logger.info(f"✅ Chat test passed.")
        return True
    except Exception as e:
        logger.error(f"❌ Chat test failed: {e}")
        return False


def test_embedding(embedding_model, embedding_api_key, embedding_api_base):
    logger.info(f"🧪 Testing embedding model: {embedding_model}")
    try:
        response = embedding(
            model=embedding_model,
            api_key=embedding_api_key,
            api_base=embedding_api_base,
            input="Hello world!",
        )
        logger.info("✅ Embedding test passed.")
        return True
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {e}")
        return False


def env_check():
    if "BACKEND" not in os.environ:
        logger.warning(
            f"We did not find BACKEND in your configuration, please add it to your .env file. "
            f"You can run a command like this: `dotenv set BACKEND rdagent.oai.backend.LiteLLMAPIBackend`"
        )

    if "DEEPSEEK_API_KEY" in os.environ:
        chat_api_key = os.getenv("DEEPSEEK_API_KEY")
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        embedding_api_key = os.getenv("LITELLM_PROXY_API_KEY")
        embedding_api_base = os.getenv("LITELLM_PROXY_API_BASE")
        if "DEEPSEEK_API_BASE" in os.environ:
            chat_api_base = os.getenv("DEEPSEEK_API_BASE")
        elif "OPENAI_API_BASE" in os.environ:
            chat_api_base = os.getenv("OPENAI_API_BASE")
        else:
            chat_api_base = None
    elif "OPENAI_API_KEY" in os.environ:
        chat_api_key = os.getenv("OPENAI_API_KEY")
        chat_api_base = os.getenv("OPENAI_API_BASE")
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        embedding_api_key = chat_api_key
        embedding_api_base = chat_api_base
    else:
        logger.error("No valid configuration was found, please check your .env file.")

    logger.info("🚀 Starting test...\n")
    result_embedding = test_embedding(
        embedding_model=embedding_model, embedding_api_key=embedding_api_key, embedding_api_base=embedding_api_base
    )
    result_chat = test_chat(chat_model=chat_model, chat_api_key=chat_api_key, chat_api_base=chat_api_base)

    if result_chat and result_embedding:
        logger.info("✅ All tests completed.")
    else:
        logger.error(" One or more tests failed. Please check credentials or model support.")


def health_check(
    check_env: Annotated[bool, typer.Option("--check-env/--no-check-env", "-e/-E")] = True,
    check_docker: Annotated[bool, typer.Option("--check-docker/--no-check-docker", "-d/-D")] = True,
    check_ports: Annotated[bool, typer.Option("--check-ports/--no-check-ports", "-p/-P")] = True,
):
    """
    Run the RD-Agent health check:
    - Check if Docker is available
    - Check that the default ports are not occupied
    - (Optional) Check that the API Key and model are configured correctly.

    Args:
        check_env (bool): Whether to check API Key and model configuration.
        check_docker (bool): Checks if Docker is installed and running.
        check_ports (bool): Whether to check if the default port (19899) is occupied.
    """
    check_any = False

    if check_env:
        check_any = True
        env_check()
    if check_docker:
        check_any = True
        check_docker_status()
    if check_ports:
        check_any = True
        check_and_list_free_ports()

    if not check_any:
        logger.warning("⚠️ All health check items are disabled. Please enable at least one check.")


if __name__ == "__main__":
    typer.run(health_check)
