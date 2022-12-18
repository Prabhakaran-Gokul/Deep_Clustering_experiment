import sys
import sklearn
import matplotlib
import numpy as np

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Training Data: {}'.format(x_train.shape))
print('Training Labels: {}'.format(y_train.shape))

import matplotlib.pyplot as plt

# python magic function
#matplotlib inline

# create figure with 3x3 subplots using matplotlib.pyplot
fig, axs = plt.subplots(3, 3, figsize = (12, 12))
plt.gray()

# loop through subplots and add mnist images
for i, ax in enumerate(axs.flat):
    ax.matshow(x_train[i])
    ax.axis('off')
    ax.set_title('Number {}'.format(y_train[i]))
    
# display the figure
fig.show()

# preprocessing the images
acc_final = np.zeros([15])

for i in range(15):

    # convert each image to 1 dimensional array
    X = x_train.reshape(len(x_train),-1)
    Y = y_train

    # normalize the data to 0 - 1
    X = X.astype(float) / 255.

    print(X.shape)
    print(X[0].shape)

    from sklearn.cluster import MiniBatchKMeans

    n_digits = len(np.unique(y_test))
    print(n_digits)

    # Initialize KMeans model
    kmeans = MiniBatchKMeans(n_clusters = n_digits)

    # Fit the model to the training data
    kmeans.fit(X)
    kmeans.labels_
    print("aditya1")
    print(kmeans.labels_, len(kmeans.labels_))
    print("aditya2")

    y_pred = kmeans.predict(X)
    print(y_pred)

    acc = np.sum(y_pred == Y)/float(len(Y))
    print("acc1",acc)

    def infer_cluster_labels(kmeans, actual_labels):
        """
        Associates most probable label with each cluster in KMeans model
        returns: dictionary of clusters assigned to each label
        """

        inferred_labels = {}

        for i in range(kmeans.n_clusters):

            # find index of points in cluster
            labels = []
            index = np.where(kmeans.labels_ == i)

            # append actual labels for each point in cluster
            labels.append(actual_labels[index])

            # determine most common label
            if len(labels[0]) == 1:
                counts = np.bincount(labels[0])
            else:
                counts = np.bincount(np.squeeze(labels))

            # assign the cluster to a value in the inferred_labels dictionary
            if np.argmax(counts) in inferred_labels:
                # append the new number to the existing array at this slot
                inferred_labels[np.argmax(counts)].append(i)
            else:
                # create a new array in this slot
                inferred_labels[np.argmax(counts)] = [i]

            #print(labels)
            #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))
            
        return inferred_labels 

    def infer_data_labels(X_labels, cluster_labels):
        """
        Determines label for each array, depending on the cluster it has been assigned to.
        returns: predicted labels for each array
        """
        
        # empty array of len(X)
        predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
        
        for i, cluster in enumerate(X_labels):
            for key, value in cluster_labels.items():
                if cluster in value:
                    predicted_labels[i] = key
                    
        return predicted_labels
    # test the infer_cluster_labels() and infer_data_labels() functions
    cluster_labels = infer_cluster_labels(kmeans, Y)
    X_clusters = kmeans.predict(X)
    predicted_labels = infer_data_labels(X_clusters, cluster_labels)
    print(predicted_labels[:20])
    print(Y[:20])
    print(predicted_labels.shape)


    from sklearn import metrics
    acc_final[i] = metrics.accuracy_score(predicted_labels,Y)
    print(metrics.accuracy_score(predicted_labels,Y))
#print(metrics.accuracy_score(number_labels,y_train))

print("Final accuracy", np.sum(acc_final)/ len(acc_final))

################### TEST ####################

X_test = x_test.reshape(len(x_test),-1)
Y_test = y_test

# normalize the data to 0 - 1
X_test = X_test.astype(float) / 255.

kmeans_lables_test = kmeans.predict(X_test)
print("aditya22")

acc11 = np.sum(kmeans_lables_test == Y_test)/float(len(Y_test))
print(acc11)


#cluster_labels_test = infer_cluster_labels(gmm, Y_test)
X_clusters_test = kmeans.predict(X_test)
predicted_labels_test = infer_data_labels(X_clusters_test, cluster_labels)
print(predicted_labels_test[:20])
print(Y_test[:20])

from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score
print(metrics.accuracy_score(predicted_labels_test,Y_test))
print(normalized_mutual_info_score(Y_test, predicted_labels_test))

