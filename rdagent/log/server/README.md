# RDAgent Log Server

## Backend

### Step 1.

1. Mock real rdagent behavior using demo_traces
2. Base Flask server to return messages

### Step 2.

1. control (pause/rerun) rdagent process
2. watch logs when rdagent process is running

## Interface

Message type in each RD Loop:
1. Research:
    - Hypothesis
        - raw
        - reason
        - concise_reason,
        - concise_observation,
        - concise_justification,
        - concise_knowledge,
    - Tasks * m
        - name
        - description
        - (LaTeX formulation)
        - (LaTeX variables)
2. Development:
    - evolving parts * n (n=10)
        - code * m
        - feedback * m
            - final decision (✅ or ❌)
            - final feedback
            - execution feedback
            - code feedback
            - value feedback
            - (shape feedback)
            - (...feedback)
3. Feedback:
    - (Config) (in qlib scenario)
    - (Returns) (in qlib scenario)
    - metric
    - hypothesis feedback
        - decision (✅ or ❌)
        - observations
        - evaluation
        - reason
        - new hypothesis
4. END

### POST /upload

- Request

```json
{
    "scenario": "",
    "files": [],
    "max_loop": 0,
}
```
- Response

```json
{
    "id": 0,
    "success": "True",
}
```

### POST /control

- request
```json
{
    "id": 0,
    "pause_flag": "True",
    "added_loops": 3,
}
```


### POST /trace

- Request
```json
{
    "id": 0,
    "all": "True",
    "reset": "True", // 从第一个log msg开始返回
}
```

- Response
```json
[
    {
        "tag": "r.h",
        "content": {} // hypothesis
    },
    {
        "tag": "r.t",
        "content": [
            {} // * m tasks
        ]
    },
    {
        "tag": "d.e",
        "content": [
            {} // * n evolving rounds
        ]
    },
    {
        "tag": "END",
        "content": {}
    }
]
```

