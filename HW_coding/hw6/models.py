import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.
        #实现 get_prediction（self， x），如果点积为非负数，则返回 1，否则返回 -1。您应该使用 nn.as_scalar 将标量 Node 转换为 Python 浮点数。
        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        not_all_correct = 1
        while not_all_correct == 1:
            not_all_correct = 0
            for x, y in dataset.iterate_once(1):
                predict = self.get_prediction(x)
                if predict != nn.as_scalar(y):
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
                    not_all_correct = 1
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 1
        self.w1 = nn.Parameter(1, 70)
        self.b1 = nn.Parameter(1, 70)
        self.w2 = nn.Parameter(70, 1)
        self.b2 = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # f(x) = relu(x⋅w1+b1)⋅w2+b2
        x_w1 = nn.Linear(x, self.w1)
        x_w1_b1 = nn.AddBias(x_w1, self.b1)
        ReLU_x_w1_b1 = nn.ReLU(x_w1_b1)
        ReLU_x_w1_b1_w2 = nn.Linear(ReLU_x_w1_b1, self.w2)
        ReLU_x_w1_b1_w2_b2 = nn.AddBias(ReLU_x_w1_b1_w2, self.b2)
        return ReLU_x_w1_b1_w2_b2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while 1:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad = nn.gradients(loss, [self.w1, self.w2, self.b1, self.b2])
                self.w1.update(grad[0], -0.01)
                self.w2.update(grad[1], -0.01)
                self.b1.update(grad[2], -0.01)
                self.b2.update(grad[3], -0.01)
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.02:
                break
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(784, 200)
        self.b1 = nn.Parameter(1, 200)
        self.W2 = nn.Parameter(200, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # f(x) = relu(x⋅w1+b1)⋅w2+b2
        x_w1=nn.Linear(x, self.W1)
        x_w1_b1 = nn.AddBias(x_w1, self.b1)
        ReLU_x_w1_b1=nn.ReLU(x_w1_b1)
        ReLU_x_w1_b1_w2=nn.Linear(ReLU_x_w1_b1, self.W2)
        ReLU_x_w1_b1_w2_b2 = nn.AddBias(ReLU_x_w1_b1_w2, self.b2)
        return ReLU_x_w1_b1_w2_b2
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x,y in dataset.iterate_forever(100):
            gradient = nn.gradients(self.get_loss(x,y), [self.W1, self.b1, self.W2, self.b2])
            self.W1.update(gradient[0], -0.35)
            self.b1.update(gradient[1], -0.35)
            self.W2.update(gradient[2], -0.35)
            self.b2.update(gradient[3], -0.35)
            if dataset.get_validation_accuracy() >= 0.975:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w = nn.Parameter(47, 100)
        self.w_hidden = nn.Parameter(100, 100)
        self.w_output = nn.Parameter(100, 5)
        self.batch_size = 0
        self.alpha = -0.01
    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        if self.batch_size == 0:
            self.batch_size = xs[0].data.shape[0]
        h = nn.ReLU(nn.Linear(xs[0], self.w))
        for x in xs[1:]:
            h = nn.Add(nn.Linear(x, self.w), nn.Linear(h, self.w_hidden))
            h = nn.ReLU(h)
        return nn.Linear(h, self.w_output)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while 1:
            if dataset.get_validation_accuracy() > 0.87:
                break
            n = 0
            for x, y in dataset.iterate_once(self.batch_size):
                n += 1
                loss = self.get_loss(x, y)
                grad = nn.gradients(loss, [self.w, self.w_hidden, self.w_output])
                self.w.update(grad[0], self.alpha)
                self.w_hidden.update(grad[1], self.alpha)
                self.w_output.update(grad[2], self.alpha)
            if dataset.get_validation_accuracy() > 0.8:
                self.alpha = -0.003
            if n >= 20000:
                break