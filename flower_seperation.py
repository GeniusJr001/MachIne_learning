from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

CSV_COLUMN_NAMES = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
SPECIES = ["Setosa", "Versicolor", "virginica"]
train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
#passing it to pandas as it must be read by pandas first (keras just saves the file in the program)
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
print(train)
train_y = train.pop("Species")
test_y = test.pop("Species")

#input function for classification
def input_fn(features, labels, training=True, batch_size=256):
    #convert the input data into a dataset_    
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
        
    return dataset.batch(batch_size)
print(test)
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#using DNN classifier instead Linear classifier as it is tensorflow best choice
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns, 
    hidden_units=[30, 10], 
    n_classes=3)

#training the model
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True), 
    steps=5000)

eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
#print("\n Test set accuracy. {accuracy:0.3f}\n".format(**eval_result))

def input_fn(features, batch_size=256):
    #convert input to dataset without labels
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
'''code to accept users input'''
features = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
predict = {}

print("please type numeric values as prompted")
for feature in features:
    valid = True
    while valid:
        val = input(feature+ ": ")
        if not val.isdigit(): valid=False
        
    predict[feature] = [float(val)]
    
predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pre_dict in predictions:
    class_id = pre_dict["class_ids"][0]
    probability = pre_dict["probabilities"][class_id]
    
    print('Prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100*probability))
    #5.5         2.4          3.7         1.0
    print(pre_dict)