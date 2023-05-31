# classifiers.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
from classificationMethod import ClassificationMethod
import numpy as np


class LinearRegressionClassifier(ClassificationMethod):
    """
    Classifier with Linear Regression.
    """
    def __init__(self, legalLabels):
        """

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        """
        super(LinearRegressionClassifier, self).__init__(legalLabels)
        self.legalLabels = legalLabels
        self.type = 'lr'
        self.lambda_ = 1e-4
        self.weights = None

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the Linear Regression Classifier.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.
        """
        n, dim = trainingData.shape
        X = trainingData
        Y = np.zeros((n, len(self.legalLabels)))
        Y[np.arange(n), trainingLabels] = 1
        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X) + self.lambda_*np.eye(dim)), np.dot(X.T, Y))
    
    def classify(self, data):
        """
        Predict which class is in.
        :param data: data to classify which class is in. (in numpy format)
        :return list or numpy array
        """
        return np.argmax(np.dot(data, self.weights), axis=1)
        
        
class KNNClassifier:
    """
    KNN Classifier.
    """
    
    def __init__(self, legalLabels, num_neighbors):
        """

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        :param num_neighbors: number of nearest neighbors.
        """
        self.legalLabels = legalLabels
        self.type = 'knn'
        self.num_neighbors = num_neighbors
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the Linear Regression Classifier by just storing the trainingData and trainingLabels.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.
        """
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
    
    def classify(self, data):
        """
        Predict which class is in.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.sum(a, axis): sum of array elements over a given axis.
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.
        np.sort, np.argsort: return a sorted copy (or indices) of an array.

        :param data: Data to classify which class is in. (in numpy format)
        :return Determine the class of the given data (list or numpy array)
        """

        "*** YOUR CODE HERE ***"
        # should compute (validationData[i] - trainingData[j])^2
        # util.raiseNotDefined()

        """
        data:              validation(1000) x dim(784)
        self.trainingData: training(5000)   x dim(784)
        dist:              validation(1000) x training(5000)
        """
        # dist[i][j] represents the squared Euclidean distance between validationData[i] and trainingData[j]
        v, dim = data.shape # 1000 * 784
        pred = []
        for vi in range(v):
            datavi = data[vi,:]
            # distvi = np.linalg.norm(datavi[:,None] - self.trainingData[None], ord=2, axis=2) 
            distvi = np.linalg.norm(datavi.reshape(1,1,dim) - self.trainingData[None], ord=2, axis=2) 
            kn_indices = np.argsort(distvi, axis=1)[:,:self.num_neighbors]
            kn_label = self.trainingLabels[kn_indices]
            pred.append([np.argmax(np.bincount(label)) for label in kn_label])
        return pred


class PerceptronClassifier:
    """
    Perceptron classifier.
    """
    def __init__( self, legalLabels, max_iterations):
        """
        self.weights/self.bias: parameters to train, can be considered as parameter W and b in a perception.
        self.batchSize: batch size in a mini-batch, used in SGD method
        self.weight_decay: weight decay parameters.
        self.learningRate: learning rate parameters.

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        :param max_iterations: maximum epoches
        """
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.batchSize = 100
        self.weight_decay = 1e-3
        self.learningRate = 1e-2
        
    def setWeights(self, input_dim):
        self.weights = np.random.randn(input_dim, len(self.legalLabels))/np.sqrt(input_dim)
        self.bias = np.zeros(len(self.legalLabels))
    
    def prepareDataBatches(self, traindata, trainlabel):
        """
        Generate data batches with given batch size(self.batchsize)

        :return a list in which each element are in format (batch_data, batch_label). E.g.:
            [(batch_data_1, batch_label_1), (batch_data_2, batch_label_2), ..., (batch_data_n, batch_label_n)]

        """
        index = np.random.permutation(len(traindata))
        traindata = traindata[index]
        trainlabel = trainlabel[index]
        split_no = int(len(traindata) / self.batchSize)
        return zip(np.split(traindata[:split_no*self.batchSize], split_no), np.split(trainlabel[:split_no*self.batchSize], split_no))

    def train(self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.

        Some data structures that may be in use:
        self.weights/self.bias (numpy format): parameters to train,
            can be considered as parameter W and b in a perception.
        self.batchSize (scalar): batch size in a mini-batch, used in SGD method
        self.weight_decay (scalar): weight decay parameters.
        self.learningRate (scalar): learning rate parameters.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.sum(a, axis): sum of array elements over a given axis.
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.
        np.mean(a, axis): mean value of array elements over a given axis
        np.exp(a)
        """

        self.setWeights(trainingData.shape[1])
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        
        # Hyper-parameters. Your can reset them. Default batchSize = 100, weight_decay = 1e-3, learningRate = 1e-2
        "*** YOU CODE HERE ***"
        self.batchSize = 1
        self.weight_decay = 1e-4
        self.learningRate = 1e-2

        for iteration in range(self.max_iterations):
            if iteration % 10 == 0: print ("Starting iteration ", iteration, "...")
            dataBatches = self.prepareDataBatches(trainingData, trainingLabels)
            for batchData, batchLabel in dataBatches:
                "*** YOUR CODE HERE ***"
                """
                self.weights: (input_dim, self.legalLabels = 10) = (784, 10)
                self.bias:    (1, self.legalLabels) = (1, 10)
                batchData: (self.batchSize, dim) = (100,784)
                batchLabel: (self.batchSize, 1) = (100, )
                pi:  (self.batchSize, self.legalLabels) = (100, 10)      
                pi[i][j] = p(y = j | x_i)
                pi[:,j] = (100, )
                grad_wj = (784, 10)
                grad_bj = (1, 10)
                """
                # util.raiseNotDefined()
                wxb = np.dot(batchData, self.weights) + self.bias
                max_wxb = np.max(wxb)
                log_pi = (np.dot(batchData, self.weights) + self.bias) - (max_wxb + np.log(np.sum(np.exp(wxb - max_wxb))))
                pi = np.exp(log_pi)

                grad_wj = np.zeros((784, 10))
                grad_bj = np.zeros((10))
                for j in range(len(self.legalLabels)): # j = 0:10-1
                    for i in range(self.batchSize): # i = 0:100-1
                        if j == batchLabel[i]:
                            grad_wj[:,j] += np.dot(pi[i,j] - 1, batchData[i,:]) # (1, 784)
                            grad_bj[j] += pi[i,j] - 1
                        else:
                            grad_wj[:,j] += np.dot(pi[i,j], batchData[i,:])
                            grad_bj[j] += pi[i,j]
                    
                for j in range(len(self.legalLabels)):
                    self.weights[:,j] -= np.dot(self.learningRate * self.weight_decay, self.weights[:,j]) \
                        + self.learningRate / self.batchSize * grad_wj[:,j]
                    self.bias[j] -= np.dot(self.learningRate * self.weight_decay, self.bias[j]) \
                        + self.learningRate / self.batchSize * grad_bj[j]
                
    def classify(self, data):
        """
        :param data: Data to classify which class is in. (in numpy format)
        :return Determine the class of the given data (list or numpy array)
        """
        
        return np.argmax(np.dot(data, self.weights) + self.bias, axis=1)

    def visualize(self):
        sort_weights = np.sort(self.weights, axis=0)
        _min = 0
        _max = sort_weights[-10]
        return np.clip(((self.weights-_min) / (_max-_min)).T, 0, 1)


class SVMClassifier(ClassificationMethod):
    """
    SVM Classifier
    """
    def __init__(self, legalLabels):
        """
        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        """
        super(SVMClassifier, self).__init__(legalLabels)
        self.type = 'svm'
        self.legalLabels = legalLabels
        
        # you may use this for constructing the svm classifier with sklearn
        self.sklearn_svm = None 
        
    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        training with SVM using sklearn API

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)

        sklearn.svm.SVC should be used in this algorithm. The following parameters should be taken into account:
        C: float
        kernel: string
        gamma: float
        decision_function_shape:  'ovo' or 'ovr'
        """
        from sklearn import svm
         
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        self.sklearn_svm = svm.SVC(C=5.0, kernel='rbf', gamma=1/(2*(10.0)**2), decision_function_shape='ovr')
        self.sklearn_svm.fit(trainingData, trainingLabels)

            
    def classify(self, data):
        """
        classification with SVM using sklearn API
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        predicted_labels = self.sklearn_svm.predict(data)
        return predicted_labels


class BestClassifier(ClassificationMethod):
    """
    SVM Classifier
    """
    def __init__(self, legalLabels):
        """
        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        """
        super(BestClassifier, self).__init__(legalLabels)
        self.type = 'best'
        self.legalLabels = legalLabels
        "*** YOUR CODE HERE (If needed) ***"
        self.sklearn_svm = None 
    
    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        design a classifier using sklearn API

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For passing the autograder, you may import sklearn package HERE. 
        """
        from sklearn import svm
        from sklearn.neural_network import MLPClassifier
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()]
        # NN
        # self.clf = MLPClassifier(hidden_layer_sizes=(180,200), activation='relu', solver='adam', alpha=0.0001,batch_size=16, random_state=1,
        #                           learning_rate='constant', learning_rate_init=0.0009, max_iter=1000000, 
        #                           beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
        # self.clf = MLPClassifier(hidden_layer_sizes=(200,300), activation='relu', solver='adam', alpha=0.000098,batch_size=32, random_state=1,
        #                           learning_rate='constant', learning_rate_init=0.0011, max_iter=1000000)
        self.clf = MLPClassifier(hidden_layer_sizes=(256,256), activation='relu', solver='adam', alpha=0.0001,batch_size=16, random_state=1,
                                  learning_rate='constant', learning_rate_init=0.001, max_iter=1000000)
        self.clf.fit(trainingData, trainingLabels)

        # SVM
        # self.sklearn_svm = svm.SVC(C=5.0, kernel='rbf', gamma='scale', decision_function_shape='ovr', tol=1e-6, cache_size=200)
        # self.sklearn_svm.fit(trainingData, trainingLabels)
 
    def classify(self, data):
        """
        classification with the designed classifier
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # NN
        predicted_labels = self.clf.predict(data)
        return predicted_labels

        # SVM
        # predicted_labels = self.sklearn_svm.predict(data)
        # return predicted_labels
