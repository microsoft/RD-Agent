import socket

import docker

from rdagent.log import rdagent_logger as logger


def check_docker() -> None:
    try:
        client = docker.from_env()
        client.images.pull("hello-world")
        container = client.containers.run("hello-world", detach=True)
        logs = container.logs().decode("utf-8")
        print(logs)
        container.remove()
        logger.info(f"The docker status is normal")
    except docker.errors.DockerException as e:
        logger.error(f"An error occurred: {e}")
        logger.warning(
            f"Docker status is exception, please check the docker configuration or reinstall it. Refs: https://docs.docker.com/engine/install/ubuntu/."
        )


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


def health_check():
    """
    Check that docker is installed correctly,
    and that the ports used in the sample README are not occupied.
    """
    check_docker()
    check_and_list_free_ports()
