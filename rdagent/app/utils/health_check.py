import os
import socket

import docker
import fire

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
            f"Port 19899 is occupied, please replace it with an available port when running the `rdagent ui/server_ui` command. Available ports: {free_ports}"
        )
    else:
        logger.info(f"Port 19899 is not occupied, you can run the `rdagent ui/server_ui` command")


def test_chat(chat_model, chat_api_key, chat_api_base):
    logger.info(f"🧪 Testing chat model: {chat_model}")
    try:
        from rdagent.oai.backend.litellm import LiteLLMAPIBackend

        backend = LiteLLMAPIBackend()
        backend.build_messages_and_create_chat_completion(user_prompt="Hello!")
        logger.info("✅ Chat test passed.")
        return True
    except Exception as e:
        logger.error(f"❌ Chat test failed: {e}")
        return False


def test_embedding(embedding_model, embedding_api_key, embedding_api_base):
    logger.info(f"🧪 Testing embedding model: {embedding_model}")
    try:
        from rdagent.oai.backend.litellm import LiteLLMAPIBackend

        backend = LiteLLMAPIBackend()
        backend.create_embedding(input_content="Hello world!")
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
    elif "CHAT_OPENAI_COMPATIBLE_API_KEY" in os.environ or "EMBEDDING_OPENAI_COMPATIBLE_API_KEY" in os.environ:
        chat_api_key = os.getenv("CHAT_OPENAI_COMPATIBLE_API_KEY")
        chat_api_base = os.getenv("CHAT_OPENAI_COMPATIBLE_API_BASE")
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        embedding_api_key = os.getenv("EMBEDDING_OPENAI_COMPATIBLE_API_KEY")
        embedding_api_base = os.getenv("EMBEDDING_OPENAI_COMPATIBLE_API_BASE")
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
    check_env: bool = True,
    check_docker: bool = True,
    check_ports: bool = True,
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
    fire.Fire(health_check)
