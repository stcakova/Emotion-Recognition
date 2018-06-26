# Система за разпознаване на емоции от видео запис

1. Install NumPy, Tensorflow, TFLearn, OpenCV

2. Ask request to the fer2013 database from  https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
	* Copy recieved fer2013.csv file to the data folder
	```
	$ cd data
	$ python parse_dataset.py in order to create both training and validation set
	```

3. Train the neural network by running ``` $python neural-network.py 'train' ```
4. Start the application by running ``` $python neural-network.py 'run' ```
