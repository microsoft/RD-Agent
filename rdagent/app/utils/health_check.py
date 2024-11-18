import shutil
import subprocess

from rdagent.log import rdagent_logger as logger


def check_command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def check_command_execution(command: str) -> bool:
    command = command.split(" ")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if "Hello from Docker!" in result.stdout:
        return True
    else:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False


def check_docker():
    if check_command_exists("docker"):
        if check_command_execution("docker run hello-world"):
            logger.info(f"The docker status is normal")
        else:
            if check_command_execution("sudo docker run hello-world"):
                logger.warning(f"Please add the user to the docker user group.")
            else:
                logger.error(
                    f"Docker status is exception, please check the docker configuration or reinstall it. Refs: https://docs.docker.com/engine/install/ubuntu/."
                )
    else:
        logger.warning(
            f"Docker is not installed, please install docker. Refs: https://docs.docker.com/engine/install/ubuntu/."
        )


def health_check():
    check_docker()
