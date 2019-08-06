import random

class Perceptron(object):

    def __init__(self, num_features=2, n_iter=100):
        """Initialize the percpetron.
        :param num_features: number of features to use
        :param n_iter: times to loop through the data while training
        """

        self.n_iter = n_iter
        self.weights = [0] * num_features
        self.bias = 0
        # How quickly to update a weight when we're wrong.
        self.alpha = .01

    def decide(self, observation):
        """Decision Function

        :param observation: feature vector

        :returns: a float score
        """

        score = self.bias
        for w, v in zip(self.weights, observation):
            score += w * v
        return score

    def perceive(self, observation, label):
        """Train on one labeled observation.

        Updates the perceptron with whether or not we're correct.

        :param observation: feature vector
        :param label: 0 or 1 label of above observation

        :returns: None
        """

        # First, put our observation through decision function
        score = self.decide(observation)

        # Were we right?
        correct = score > 0

        # Calculate if we need to update
        # change | label | correct
        #    0       0        0
        #   -1       0        1
        #    1       1        0
        #    0       1        1
        change = int(bool(label)) - int(correct)

        # Update each weight if we're wrong
        for i in xrange(len(self.weights)):
            self.weights[i] += observation[i] * change * self.alpha
            self.bias += change * self.alpha

    def fit(self, X, y):
        """Train with a dataset.

        :param X: a list of feature vectors
        :param y: a list of labels
        """

        training_set = zip(X, y)

        for n in xrange(self.n_iter):
            random.shuffle(training_set)
            for observation, label in training_set:
                self.perceive(observation, label)
