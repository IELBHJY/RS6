from tensorflow.examples.tutorials.mnist import input_data
from sklearn import tree
import numpy as np

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

batch_x, batch_y = mnist.train.next_batch(60000)

tree = tree.DecisionTreeClassifier()
tree.fit(batch_x,batch_y)

batch_test_x ,batch_test_y= mnist.test.images[0:10000,:],mnist.test.labels[0:10000,:]
predicts = tree.predict(batch_test_x)
correct = np.equal(np.argmax(predicts,1),np.argmax(batch_test_y,1))
print("accuracy:",np.mean(correct))