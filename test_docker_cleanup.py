#!/usr/bin/env python3
"""
Test script to demonstrate and verify Docker container cleanup issue.
This test verifies that containers are properly cleaned up even when exceptions occur.
"""
import docker
import unittest
from pathlib import Path
from rdagent.utils.env import QTDockerEnv, QlibDockerConf


class TestDockerContainerCleanup(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.client = docker.from_env()
        self.test_workspace = Path("/tmp/test_docker_cleanup")
        self.test_workspace.mkdir(exist_ok=True)
        
        # Get initial container count
        self.initial_containers = len(self.client.containers.list(all=True, filters={"status": "exited"}))
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if self.test_workspace.exists():
            shutil.rmtree(self.test_workspace)
    
    def test_container_cleanup_on_success(self):
        """Test that containers are cleaned up after successful execution."""
        qtde = QTDockerEnv()
        qtde.prepare()
        
        # Run a simple successful command
        result, return_code = qtde.run_ret_code(entry='echo "success"', local_path=str(self.test_workspace))
        
        # Verify success
        self.assertEqual(return_code, 0)
        self.assertIn("success", result)
        
        # Check that no additional exited containers were left behind
        final_containers = len(self.client.containers.list(all=True, filters={"status": "exited"}))
        self.assertEqual(final_containers, self.initial_containers, 
                        "Container was not cleaned up after successful execution")
    
    def test_container_cleanup_on_command_error(self):
        """Test that containers are cleaned up when command fails (non-zero exit code)."""
        qtde = QTDockerEnv()
        qtde.prepare()
        
        # Run a command that fails with non-zero exit code
        result, return_code = qtde.run_ret_code(entry='exit 1', local_path=str(self.test_workspace))
        
        # Verify the command failed as expected
        self.assertNotEqual(return_code, 0)
        
        # Check that no additional exited containers were left behind
        final_containers = len(self.client.containers.list(all=True, filters={"status": "exited"}))
        self.assertEqual(final_containers, self.initial_containers,
                        "Container was not cleaned up after command failure")
    
    def test_container_cleanup_on_timeout(self):
        """Test that containers are cleaned up when timeout occurs."""
        # Create a config with very short timeout
        dc = QlibDockerConf()
        dc.running_timeout_period = 1  # 1 second timeout
        qtde = QTDockerEnv(dc)
        qtde.prepare()
        
        # Run a command that should timeout
        result, return_code = qtde.run_ret_code(entry='sleep 5', local_path=str(self.test_workspace))
        
        # Verify timeout occurred
        self.assertEqual(return_code, 124)  # timeout exit code
        
        # Check that no additional exited containers were left behind
        final_containers = len(self.client.containers.list(all=True, filters={"status": "exited"}))
        self.assertEqual(final_containers, self.initial_containers,
                        "Container was not cleaned up after timeout")


if __name__ == "__main__":
    unittest.main()