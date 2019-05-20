---
layout: post
title: Creating Custom Keras Layers
categories: [Machine Learning, Programming]
image: /assets/images/foo.png
published: false
excerpt: Bladiebla
---

## The Basics

Something about an embedding lookup layer needed

## Static Layer

Static layer (which doen't have variables which are learned during training).

```python
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, arg1, arg2):
        super(MyLayer, self).__init__()
        self.arg1 = arg1
        self.arg2 = arg2
        
    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                       shape=[int(input_shape[-1]),                           self.num_outputs])
    
    def call(self, input):
        return tf.matmul(input, self.kernel)
```

### Init

These are created when the layer class is called and an instance is created (`my_layer = MyLayer(...)`). 

### Build

This is done when the model.compile step is run. Here weights are initialized and are set as trainable. Some precalculation can be done here.

### Call

This does the actual logic of the layers calculation. Take the input as input, and return the output as output.

## Composed Layer

Built of existing tensorflow layers

```python
class ComposedLayer(tf.keras.Model):
    def __init__(self, arg1, arg2):
        super(ComposedLayer, self).__init__()
        self.dense1 = tf.keras.layers.Dense(arg1)
        self.dense2 = tf.keras.layers.Dense(arg2)
        
    def call(self, input_tensor):
        x = self.dense1(input_tensor)
        x = self.dense2(x)
        
        x += input_tensor
        return tf.nn.relu(x)
```



## Embedding Lookup Layer

For my use case, I wanted a model which, given an dictionary of embeddings, would take a vector as input, lookup the closest embedding in this dictionary according to cosine-similarity, and return the index as the output.

Here is the (static) implementation

```python
def EmbeddingLookup(tf.keras.layers.Layer):
    def __init__(self, embedding_matrix)
```

## References

- https://keras.io/layers/writing-your-own-keras-layers/
- https://www.tensorflow.org/tutorials/eager/custom_layers