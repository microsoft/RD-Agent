"""
PoC test for CWE-94: Unauthenticated arbitrary process execution via /upload endpoint.

The Flask server in rdagent/log/server/app.py listens on 0.0.0.0 with no
authentication. Any network-reachable client can POST to /upload with a
crafted scenario and attacker-controlled parameters, causing the server to
spawn a new Process that executes rdagent internal targets with those parameters.

This test:
1. Validates that /upload currently accepts requests with NO authentication
   and successfully creates an RDAgentTask (returns 200).
2. Validates that /receive, /control, /user_interaction/submit also lack auth.
3. After the fix, these endpoints should require an API token and return 401/403
   when called without one.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# The rdagent package is properly importable from this repo.
# We only need to mock the subprocess spawning.


class TestUnauthenticatedAccess(unittest.TestCase):
    """Test that mutating endpoints require authentication."""

    @classmethod
    def setUpClass(cls):
        os.makedirs("/tmp/rdagent_test_static", exist_ok=True)
        os.makedirs("/tmp/rdagent_test_traces", exist_ok=True)

        from rdagent.log.server.app import app as flask_app, rdagent_processes

        cls.flask_app = flask_app
        cls.rdagent_processes = rdagent_processes
        flask_app.config["TESTING"] = True

    def setUp(self):
        self.client = self.flask_app.test_client()
        self.rdagent_processes.clear()

    def test_upload_no_auth_spawns_process(self):
        """
        /upload should reject unauthenticated requests.

        BEFORE FIX: returns 200, creates a task, spawns a process
        AFTER FIX: returns 401 (Unauthorized)
        """
        with patch("rdagent.log.server.app.Process") as MockProcess:
            mock_proc = MagicMock()
            mock_proc.is_alive.return_value = True
            MockProcess.return_value = mock_proc

            response = self.client.post(
                "/upload",
                data={
                    "scenario": "Finance Data Building",
                    "loops": "1",
                    "all_duration": "1",
                },
                content_type="multipart/form-data",
            )

            # AFTER FIX: This should return 401, not 200
            self.assertEqual(
                response.status_code,
                401,
                f"Expected 401 Unauthorized for unauthenticated /upload, "
                f"got {response.status_code}. Response: {response.get_data(as_text=True)}",
            )

    def test_receive_no_auth_injects_messages(self):
        """
        /receive should reject unauthenticated requests.

        BEFORE FIX: returns 200, injects message into trace
        AFTER FIX: returns 401 (Unauthorized)
        """
        response = self.client.post(
            "/receive",
            json={"id": "test/trace", "msg": {"tag": "injected", "content": "malicious"}},
        )
        self.assertEqual(
            response.status_code,
            401,
            f"Expected 401 Unauthorized for unauthenticated /receive, "
            f"got {response.status_code}",
        )

    def test_control_no_auth_stops_process(self):
        """
        /control should reject unauthenticated requests.

        BEFORE FIX: returns 200/400, can stop any running process
        AFTER FIX: returns 401 (Unauthorized)
        """
        response = self.client.post(
            "/control",
            json={"id": "test/trace", "action": "stop"},
        )
        self.assertEqual(
            response.status_code,
            401,
            f"Expected 401 Unauthorized for unauthenticated /control, "
            f"got {response.status_code}",
        )

    def test_user_interaction_no_auth(self):
        """
        /user_interaction/submit should reject unauthenticated requests.

        BEFORE FIX: returns 200, enqueues payload
        AFTER FIX: returns 401 (Unauthorized)
        """
        response = self.client.post(
            "/user_interaction/submit",
            json={"id": "test/trace", "payload": {"answer": "yes"}},
        )
        self.assertEqual(
            response.status_code,
            401,
            f"Expected 401 Unauthorized for unauthenticated /user_interaction/submit, "
            f"got {response.status_code}",
        )

    def test_upload_with_valid_token(self):
        """
        /upload should accept requests with a valid API token.
        """
        with patch("rdagent.log.server.app.Process") as MockProcess:
            mock_proc = MagicMock()
            mock_proc.is_alive.return_value = True
            MockProcess.return_value = mock_proc

            token = self.flask_app.config.get("API_TOKEN", "")
            if not token:
                self.skipTest("No API_TOKEN configured")

            response = self.client.post(
                "/upload",
                data={
                    "scenario": "Finance Data Building",
                    "loops": "1",
                    "all_duration": "1",
                },
                headers={"Authorization": f"Bearer {token}"},
                content_type="multipart/form-data",
            )

            self.assertEqual(
                response.status_code,
                200,
                f"Expected 200 for authenticated /upload, got {response.status_code}",
            )

    def test_read_only_endpoints_accessible(self):
        """
        Read-only endpoints (/, /traces, /test) should remain accessible
        without auth since they don't mutate state or spawn processes.
        """
        response = self.client.get("/traces")
        self.assertIn(
            response.status_code,
            [200],
            f"Expected /traces to remain accessible, got {response.status_code}",
        )

        response = self.client.get("/test")
        self.assertIn(
            response.status_code,
            [200],
            f"Expected /test to remain accessible, got {response.status_code}",
        )


if __name__ == "__main__":
    unittest.main()
