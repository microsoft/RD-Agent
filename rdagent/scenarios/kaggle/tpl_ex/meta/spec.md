

Information to generate spec


```python
def feature_eng(x: {{type of the feature}}) -> {{type of the feature}}:
    """
    
    x: np.ndarray
          {{description}}
    """
```

Standard to generate the qualified specification

| field       | requireemtnnts                                |
| --          | --                                            |
| description | fully describe the data, including dimension (number,meaning,  exmaple)|

Example of generated specification
```python
def feature_eng(x: {{type of the feature}}) -> {{type of the feature}}:
    """

    x: np.ndarray
        3 dimension, the meaning of the dimensions will be:
        - channel
        - high
        - width
    """
```


