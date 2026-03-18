[Neural machine translation with a Transformer and Keras  |  Text  |  TensorFlow](https://www.tensorflow.org/text/tutorials/transformer#the_transformer) 

* Transformer replace the traditional RNN’s/LSTMs by processing an entire sequence rather than doing step-by-step  
* The model utilizes an encoder-decoder architecture to translate input text into another language.  
* The Attention mechanisms allows the model to focus on important words in a sentence when translating  
* Positional encoding is used because Transformers don’t inherently understand word order.  
* With Keras/TensorFlow it simplifies building and training a Transformer

[Deeper vs. Broader Neural Networks: Which is Right for Your Model? | by Kaushiktd | Artificial Intelligence in Plain English](https://ai.plainenglish.io/deeper-vs-broader-neural-networks-which-is-right-for-your-model-acf64dfec331)

* The deeper the network (more layers) it learns much more complex hierarchical patterns such as edges \-\> shapes \-\> objects  
* The broader the network (more neurons per layer) it can capture the more features at the same level  
* Deep networks are generally more powerful but harder to train due to gradients, overfitting and vanishing.  
* Wide networks train faster but may not capture deep relationships in data

[The Depth of Deep Learning — What More Layers Can Do | by Pruthvi | Medium](https://medium.com/@jvpnath/the-depth-of-deep-learning-what-more-layers-can-do-bff9922beeb8)

* Increasing layers allows models to learn more abstract and complex representations  
* Earlier layers learn the simple features while the deeper layers learn the high-level patterns  
* Too many layers can lead to problems like vanishing gradients and overfitting  
* ReLU activation, batch normalization and skip connections can assist with deep networks training

[\[1706.03762\] Attention Is All You Need](https://arxiv.org/abs/1706.03762)

* Introduces the Transformer architecture, removing any necessity of RNN’s entirely  
* Uses self-attention to model relationships between all words in a sentence simultaneously  
* Introduces multi-head attention to capture different types of relationships in data

[Transformer Neural Networks: A Step-by-Step Breakdown | Built In](https://builtin.com/artificial-intelligence/transformer-neural-network)

* Transformers consist of encoder ans decoder blocks stacked together  
* Self-attention calculates how much each word related to every other word in a sequence  
* Multi-head attention improves learning by looking at multiple relationships at once