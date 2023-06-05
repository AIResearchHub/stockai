
# Stock AI


Using Natural Language Processing and Reinforcement Learning to predict stock prices from 
google news feed using techniques from many state of the art papers


## Transformer mock data test run

![alt text](https://github.com/YHL04/stockai/blob/master/plots/transformer.png)


## Currently implementing


Longformer (Sliding window attention to scale compute linearly with sequence length)

https://arxiv.org/pdf/2004.05150.pdf

Big Bird (Window attention combined with random attention)

https://arxiv.org/pdf/2007.14062.pdf

Transformer XL (Using cached query and keys for extra long sequences)

https://arxiv.org/pdf/1901.02860.pdf


## Papers implemented in this algorithm


Seed RL (Scalable and Efficient Deep RL with Centralized Inference): 

https://arxiv.org/abs/1910.06591

R2D2 (Rainbow Deep Q Network with LSTM): 

https://openreview.net/pdf?id=r1lyTjAqYX

PCGrad (Gradient Surgery for Multi Task Learning):  

https://arxiv.org/pdf/2001.06782.pdf

Block Recurrent Transformer (Variant of Transformer with Recurrent Attention): 

https://arxiv.org/abs/2203.07852

Implicit Quantile Network (Distributed RL for accelerated learning and risk sensitive policies): 

https://arxiv.org/abs/1806.06923

