# nano-GPT-tutorial
This repository follows the tutorial of Andrej Karpathy on the tiny-Shakespeare dataset. A Transformer Decoder only network that generates Shakespeare-like text, to experiment and improve understanding of Transformer Networks.  
Let's build GPT: from scratch, in code, spelled out.: https://www.youtube.com/watch?v=kCc8FmEb1nY&list=WL&index=6&ab_channel=AndrejKarpathy  
nanogpt-lecture: https://github.com/karpathy/ng-video-lecture

Run the bigram.py to recreate the results achieved in the video.  
-----------------------------------------------------------------
Best performing configuration:
learning_rate = 3e-4, dropout = 0.2, # 90 / 10 split, AdamW
Default initialization
step 4500: train loss 1.1121, validation loss: 1.4781 
------------------------------------------------------
