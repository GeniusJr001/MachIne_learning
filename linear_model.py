#titanic passenger modelling
#testing between the sex and survival
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
from silence_tensorflow import silence_tensorflow
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv") #training data (i.e data used in training)
dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv") #testing data(i.e data used to test the model)
#print(dfeval.head())
y_train = dftrain.pop("survived")#this removes the survived row from the dftrain and adds it to the Y-train
y_eval = dfeval.pop("survived")#this removes the survived row from the dftrain and adds it to the Y-eval
#print(dfeval.head()) #this shows the first 5 lines in the dataset
#print(y_train)
#print(dftrain.describe()) #This tells the mean value the standard deviation, the percentiles and the minimum and maximum value
#print(dftrain.shape) # This tells the number of rows and  columns
#print(dftrain.age.hist(bins=30)) #This prints an histogram of the age distribution in the file
CATEGORICAL_COLUMNS = ["sex", "n_siblings_spouses", "parch", "class", "deck", "embark_town", "alone"]
NUMERIC_COLUNMS = ["age", "fare"]
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocalbulary = dftrain[feature_name].unique() #this removes the unique data sets for each column in the categorecal column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocalbulary))
    
for feature_name in NUMERIC_COLUNMS:
    vocalbulary = dftrain[feature_name].unique() #this removes the unique data sets for each column in the categorecal column
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs) #split dataset into batches of 32 and repeat for number of epochs
        return ds #return a batch of dataset
    return input_function 
train_input_fn = make_input_fn(dftrain, y_train, num_epochs=10, shuffle=True)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

#creating the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
#result = linear_est.evaluate(eval_input_fn)
result = list(linear_est.predict(eval_input_fn))
clear_output()
print(dfeval.loc[4]) #all the data at location 0
print(result) #this outputs the list of predictions for each value in the list
#print(y_eval.loc[4]) #if the prson did survive
print(result[4]["probabilities"][1])

def input_fn(features, batch_size=256):
    #convert input to dataset without labels
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
'''code to accept users input'''
features = ["age", "fare"]
predict = {}

print("please type numeric values as prompted")
for feature in features:
    valid = True
    while valid:
        val = input(feature+ ": ")
        if not val.isdigit(): valid=False
        
    predict[feature] = [float(val)]
    
predictions = linear_est.predict(input_fn=lambda: input_fn(predict))
for pre_dict in predictions:
    class_id = pre_dict["class_ids"][0]
    probability = pre_dict["probabilities"][class_id]
    
    print('Prediction is "{}" ({:.1f}%)'.format(CATEGORICAL_COLUMNS[class_id], 100*probability))