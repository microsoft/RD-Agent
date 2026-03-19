#ALFWorldQuest

## Objective
Train the model to achieve higher task success rates in the ALFWorld text game environment. This is an **interactive** task: the model needs to make decisions (rollout) in the environment in multiple steps, rather than generating an answer in one go.

## Environment Overview
ALFWorld is a text simulation home environment (TextWorld engine). The model acts as an agent and uses text instructions to navigate the room and operate items to complete tasks.

## Task type (6 types)
1. **pick_and_place**: Pick up the item and put it in the specified location
2. **pick_clean_then_place**: Clean the items and put them in the designated location
3. **pick_heat_then_place**: Heat the item and put it in the specified location
4. **pick_cool_then_place**: Cool the item and put it in the designated location
5. **look_at_obj_in_light**: View items under light
6. **pick_two_obj_and_place**: Pick up two items and put them in the specified location

## Rollout process

Interactive loop for each game:

```
Initialization: ob, info = env.reset() # Get initial observation (room description + task goal)

Loop (per step):
action = model (observation history) # The model generates actions (text) based on history
ob, reward, done, info = env.step([action]) # The environment executes the action and returns a new observation
  if done:
      break
```

**A rollout example (pick_and_place):**
```
Task: "put a pencil in/on shelf."

Step 1: Observation: "You are in the middle of a room. Looking around you, you see a bed 1, a desk 1, a shelf 1..."
Action: "go to desk 1"
Step 2: Observation: "On the desk 1, you see a pencil 1, a book 2."
Action: "take pencil 1 from desk 1"
Step 3: Observation: "You pick up the pencil 1 from the desk 1."
Action: "go to shelf 1"
Step 4: Observation: "You arrive at shelf 1. On the shelf 1, you see nothing."
Action: "put pencil 1 in/on shelf 1"
Step 5: Observe: "You put the pencil 1 in/on the shelf 1."
Result: Task completed
```

## Available action space
Agent's actions are free text. Common actions include:
- Navigation: `go to {object} {id}` (eg `go to desk 1`, `go to fridge 1`)
- Take: `take {object} {id} from {location} {id}`
- Place: `put {object} {id} in/on {location} {id}`
- Open/Close: `open {object} {id}`, `close {object} {id}`
- Heating/cooling: `heat {object} {id} with microwave {id}`, `cool {object} {id} with fridge {id}`
- Cleaning: `clean {object} {id} with sinkbasin {id}`
- Use: `use {object} {id}` (such as `use desklamp 1`)
- Thinking: `think: {reasoning}` (does not affect the environment state)

## Evaluation indicators
- **Success Rate** = Number of successful tasks / Total number of tasks

## Reference code
The complete implementation of environment interaction and evaluation can be found in `eval.py`.
