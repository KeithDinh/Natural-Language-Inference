Stanford Natural Language Inference
	Milestone 2 - Data Modeling

===================================================
Folder contains: 
1. SNLI_Temp: main folder contains the project.
2. A README.txt file
3. Report-2-Data-Modeling.docs : report file.

===================================================

Instructions for SNLI_Temp
1. Folder .idea and _pycache_ : project folders of Pycharm
2. Folder data:  contains the event files of Model 1, built by Tensorflow using Tensorboard.
		They can be used for visualizing train/val loss, model structure.
3. Folder log: contains the training's log files of both models. Log files can be opened with Notepad.
4. Folder model: there are five saved models of model 1. They are exported by using Tensorflow.
5. Folder model_structure: graphs showing structure of both models. Using Tensorflow to draw them.
6. Folder plot: mainly contains plots train/val loss of model 2, overall train/test accuracy of both model
7. Folder testing_result: contains all logs file from two models. Each log file stored the
		hyper-paramets used, loss/test accuracy, their precision, recall, F1-score, 
		confusion matrix. Log files can be opened with Notepad.
8. File training_clean.csv, testing_clean.csv, dev_clean.csv
		clean dataset after milestone 1
9. Text file train.txt, test.txt, dev.txt: transformed file from csv. Only contain 3 features
		gold_label ||| sentence1 ||| sentence2
		These three text files are mainly used in the model 1 and model 2. Upon this time,
		we don't use csv files anymore.
10. Text file vocab.txt: a vocabulary text file is built with the pre-trained GloVe model + train.txt
11. Text file glove.840B.300d.txt: pre-trained GloVe model file dowloaded from Stanford Natural Inference
		offical site.
		http://nlp.stanford.edu/data/glove.840B.300d.zip
12. WE file glove.840B.300d.we: file is used for constructing the embedding layer of model 1.
13. NPY file GloVe_weights.npy: storing the weights from GloVe model of model 2.
14. SNLI_Train.py: model 1's source code for building and training.
15. SNLI_Test.py: model 1's source code for loading specific model file from folder model 
		and performing evaluation.
16. SNLI_LSTM_GRU: model 2's source code for building, training, and testing.

======================================================

Terminal commands:
	- For visualizing train/val loss, model struture of model 1
		Terminal command: tensorboard --logdir=data
	- For starting training model 1, you have to modify hyper-parameters in the
	source code file. Then, executing this command
		Terminal command: python SNLI_Train.py
	- For testing accuracy and evaluation on model 1, run this command in terminal
		Terminal command: python SNLI_Test.py -m model/SNLI_0_G
		-m: path to model file. This file will have a .meta extension.
		The above command perform testing with model named SNLI_0_G
	- For training and testing on model 2
		Terminal command: python SNLI_LSTM_GRU.py > log/log.data-time
		This command will export all training log file from Tensorflow and 
		store all of them into folder log with a specific name. It also creates a 
		log file named hyper-parameters.data-time.

=========================================================
Due to the large file size (~8G), zip file (~4G). We will remove these file from the 
	submission folder. We will add instructions below in order to build these files
	throught the project. Removed files:
		glove.840B.300d.txt
		glove.840B.300d.we
		GloVe_weights.npy
		
Go to SNLI pre-trained GloVe and download the model.
	http://nlp.stanford.edu/data/glove.840B.300d.zip
	Copy and paste the text file named glove.840B.300d.txt into project folder

Build glove.840B.300d.we for model 1 by executing these lines of code in SNLI_Train.py. 
	We commented them in SNLI_Train.py from line 592 to line 597
	    	# Execute this function to transform the embedding to
    		# a convinience text file
    		# transform_embedding_to_txt('glove.840B.300d.txt', 2196017, 300)

	    	#Build vocab file using pre-trained GloVe and training data
    		build_vocab('train.txt', 'vocab.txt')

Build GloVe_weights.npy for model 2 by running the python file SNLI_LSTM_GRU.py.
	Our code will automatically check if the GloVe.weights.npy is existed.
	If it existed, just skip.
	If it does not existed, it will build one.
	