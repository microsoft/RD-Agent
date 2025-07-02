# Folder structure design

## Concepts
When we are optimizing solutions, we have the following strategies.
- `draft`: create a new solution
- `idea`: propose an idea to improve solutions
- `merge`: merge different solutions


Optimization is a long journey; we may switch between different strategies. These strategies are called routers.
- `router`: a meta strategy to route between different strategies.  Router may have different implementations.


Tools:
- `select`: It provides features to be used by other strategies or steps.
  1) `submit`: before we end the optimization, we have to select the only one solution to submit.
  2) `expand`: we may have select one point to start the next expansion.


## Suggest folder structure
So the suggested folder structure is:
```
- router/
- idea/
  - samll_step(or refine?).py
  - normal.py
- draft/
- merge/
- select/
  - expand.py
  - submit.py
```
