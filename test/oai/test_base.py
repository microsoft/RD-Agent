import pytest


class MockBackend:
    def __init__(self):
        self.messages = []

    def _add_json_in_prompt(self, new_messages):
        self.messages.append("JSON_ADDED")


def test_json_added_once():
    backend = MockBackend()
    try_n = 3
    json_added = False
    new_messages = ["msg1"]

    for _ in range(try_n):
        if not json_added:
            backend._add_json_in_prompt(new_messages)
            json_added = True

    assert backend.messages.count("JSON_ADDED") == 1
