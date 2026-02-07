The structure of the entire setup is as follows:

```
|___________/Codes
|	       |___ /CELDM
|	       		|___ CELDM.py	# script to train cross entropy loss-based counseling dialogue model.
|
|	       |___ /RL_Fine-Tuning
|	       		|___ dataset.py				# script to load the custom dataset by creation of a pytorch Dataset class
|	       		|___ rlmain.py				# script to fine-tune PPO loss-based dialogue model.
|	       		|___ rlutils.py                     # script containing utility functions and reward functions for the RL fine-tuning task
|                       |___ ppo.py 				# script containing implementation of buffer memory
|                       |___ loss.py 				# script containing implementation of Sequence Cross Entropy Loss
|              		|___ rlinference.py 			# script to interact with the fine-tuned model.
|
|
|	       |___ /Classifiers
|                       |___ classifier.py 			# script to implement classifiers
|
|
|
|            |___ /Evaluation
|	       		|___ interact.py				# script to evaluate the proposed system.
|
|___________/Datasets	 				 	         
|              |___ HEAL.csv						# Mental Health and Legal Counseling Dialogue Dataset with counseling act, politeness strategy and empathy strategy information. 
```

****REQUIREMENTS****
1. numpy: version '1.21.2'
2. pandas: version '1.3.4'
3. transformers: version '4.11.2'
4. tqdm: version: version '4.62.3'
5. torch: version '1.10.0'


## Training

python rlmain.py


## Inferencing

python interact.py

