#WebShop tasks

## Objective
The training model achieves a higher success rate of shopping tasks in the WebShop e-commerce website environment. This is an **interactive** task: the model needs to make a multi-step decision (rollout) in a web environment, search for and purchase matching products based on user instructions.

## Environment Overview

WebShop is a simulated e-commerce website environment containing 1.18 million real products and user instructions. Agent needs to complete shopping tasks according to text instructions.

The environment has 4 page states:
- **search** - search page, including search box
- **results** - Search results page listing matching products
- **item** - Product details page
- **item-detail** - Product details page

## Action Space

Agent's actions are in text format and have two types:

1. **Search**: `search[query]` - used on the search page
- Example: `search[red running shoes]`

2. **Choose**: `choose[option]` - Select an option based on the current page
- `choose[Back to Search]` - Return to the search page
- `choose[Next >]` / `choose[< Prev]` - turn page
- `choose[Product Title]` - choose product
- `choose[Option]` - choose color/size variations
- `choose[Description]` - View details
- `choose[Buy Now]` - buy product

## Rollout process

The interactive loop of each round of shopping tasks:

```python
# initialization
obs, info = env.reset(idx=instruction_idx) # Get initial observations (search page)

done = False
for step in range(max_steps):
# 1. The model generates actions based on instructions, history, and current observations.
    action = model(instruction, history, obs)
    
# 2. Environment execution action
    obs, reward, done, info = env.step(action)
    
# 3. Record history
    history.append((action, obs))
    
    if done:
        break

# reward: final reward (0-1), reflecting the degree of product matching
```

**A rollout example**:

```
Instruction: "I'm looking for a quick-release replacement fitness strap band;
       it should match my chic teal fitbit, and price lower than 40.00 dollars"

Step 1: Observe: "WebShop [SEP] Search [SEP]"
Action: "search[quick-release fitness strap band teal fitbit]"

Step 2: Observe: "WebShop [SEP] Results [SEP] [Back to Search] [Next >]
               [Teal Silicone Sport Band for Fitbit... $12.99] 
               [Quick Release Nylon Band Teal... $15.99]..."
Action: "choose[Teal Silicone Sport Band for Fitbit Charge 2, Large, $12.99]"

Step 3: Observe: "WebShop [SEP] Item [SEP] Teal Silicone Sport Band...
               [Buy Now] [Back to Search] [Description] [Size Large] [Size Small]"
Action: "choose[Buy Now]"

Step 4: Observation: "WebShop [SEP] Episode finished [SEP] reward = 0.95"
Result: Task completed, reward 0.95 (high matching degree)
```

## Observation format

Observations returned by the environment are in text format:

```
WebShop [SEP] {Page Type} [SEP] {Content}
```

- `WebShop` - fixed prefix
- `{Page Type}` - Page type: Search / Results / Item
- `{Content}` - page content, including available options

## Evaluation indicators

- **Success Rate** = The proportion of successful purchases of matching products (reward >= 0.5 is considered successful)
- **Average reward** = the average reward value (0-1) of all tasks, calculated based on product type, attributes, and price matching

## Reference code

The complete implementation of environment interaction and evaluation can be found in `eval.py`.
