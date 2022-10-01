import sys
import json
import numpy as np
from classes import (
    Activation,
    Cost,
    LogLikehoodCost,
    SoftmaxActivation,
    CrossEntropyCost,
    SigmoidActivation,
    ReLUActivation,
    QuadraticCost,
    LinearActivation,
    LinearCost,
    WideCrossEntropyCost,
    HiperbolicTangentActivation,
    ReSoftPlusActivation,
    FullyConnectedLayer,
    SoftmaxLayer,
)
from classes import Layer
from utils import Utils


class Network:
    def __init__(
        self,
        layers: list[Layer],
        save: bool = False,
    ):
        self.num_layers = len(layers)
        self.layers = layers
        self.default_layers_initializer()
        self.do_save = save
        self.rng = np.random.default_rng()

    def default_layers_initializer(self):
        self.layers[0].p_dropout = 0
        self.layers[-1].p_dropout = 0

        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            l2.weights = np.random.randn(l2.size, l1.size) / (
                np.sqrt(l1.size) * (1 - l1.p_dropout)
            )
            l2.weights_velocity = np.random.randn(l2.size, l1.size) / (
                np.sqrt(l1.size) * (1 - l1.p_dropout)
            )

        for l in self.layers[1:]:
            l.biases = np.random.randn(l.size, 1)
            l.biases_velocity = np.random.randn(l.size, 1)

        for l in self.layers:
            l.setReady(True)

    def SGD(
        self,
        training_data,
        mini_batch_size: int,
        eta: float,
        epochs: int = 0,
        lmbda: float = 0.0,
        mu: float = 0,
        stop_no_improvement: int = 0,
        test_data=None,
        monitor_test_cost=False,
        monitor_test_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False,
        not_greather_random_reinitialize=-1,
    ):
        mbs = mini_batch_size
        test_cost, test_accuracy = [], []
        training_cost, training_accuracy = [], []
        epoch = 0
        while epoch < epochs or epochs == 0:
            if not self.hasImprovement(
                test_accuracy, training_accuracy, stop_no_improvement
            ):
                break
            epoch += 1
            training_data = Utils.unison_shuffled_copies(self.rng, *training_data)
            mini_batches = [
                [training_data[i][k : k + mbs] for i in range(len(training_data))]
                for k in range(0, len(training_data[0]), mbs)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, mu, len(training_data[0])
                )

            # print(f"Epoch {ecpoch} training coplete")

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print(
                    f"Accuracy on training data: {accuracy} / {len(training_data[0])}"
                )
            if monitor_test_cost and test_data:
                cost = self.total_cost(test_data, lmbda)
                test_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if monitor_test_accuracy and test_data:
                accuracy = self.accuracy(test_data)
                test_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {accuracy} / {len(test_data[0])}")

            a = test_accuracy[-1:] if test_accuracy else training_accuracy[-1:]
            if a and a[0] <= not_greather_random_reinitialize:
                epoch = 0
                print(f"re {a[0]} <= {not_greather_random_reinitialize}")
                self.default_layers_initializer()

            if self.do_save:
                self.save("data.json")

        return test_cost, test_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, mu, lmbda, n):
        x, y = mini_batch

        mbs = len(y)
        nabla_b, nabla_w = self.backprop(x.transpose(), y.transpose())

        for i in range(1, self.num_layers):
            l = self.layers[i]
            nb = nabla_b[i - 1]
            nw = nabla_w[i - 1]
            l.biases_velocity = mu * l.biases_velocity - nb * eta / mbs
            l.biases += l.biases_velocity

            l.weights_velocity = mu * l.weights_velocity - (eta / mbs) * nw
            l.weights *= 1 - eta * (lmbda / n)
            l.weights += l.weights_velocity

    def backprop(self, x: np.ndarray, y: np.ndarray):
        nabla_b = [None] * (self.num_layers - 1)
        nabla_w = [None] * (self.num_layers - 1)

        ls = self.layers
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for l in ls[1:]:
            z, activation = l.feedforward(activation, self.rng, False)
            activations.append(activation)
            zs.append(z)

        delta = ls[-1].delta(activation, y, zs[-1])
        nabla_b[-1] = delta.sum(1).reshape((len(delta), 1))  # pyright: ignore
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = ls[-l].activation.dfdz(z)
            delta = np.dot(ls[-l + 1].weights.transpose(), delta) * sp
            nabla_b[-l] = delta.sum(1).reshape((len(delta), 1))
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def accuracy(self, data):
        x, y = data

        y1 = np.asarray([np.argmax(i, 0) for i in y])
        a1 = np.argmax(self.feedforward(x.transpose()), 0)
        r1 = sum(map(int, y1 == a1))

        a2 = self.feedforward(x.transpose()).transpose()
        d = np.linalg.norm(a2 - y, axis=1)
        e = 0.05
        d[d < e] = 0
        d[d > e] = 1

        r2 = int(len(d) - d.sum())

        # return f"(max: {r1}, norm: {r2})"
        return r1

    def total_cost(self, data, lmbda=0.0):
        x, y = data
        cost = 0.0
        a = self.feedforward(x.transpose()).transpose()
        cost += (
            self.layers[-1].cost.c(a, y)
            + 0.5
            * lmbda
            * sum(float(np.linalg.norm(l.weights) ** 2) for l in self.layers)
        ) / len(y)
        return cost

    def feedforward(self, a) -> np.ndarray:
        for l in self.layers[1:]:
            _, a = l.feedforward(a, self.rng)
        return a

    def hasImprovement(self, test_accuracy, training_accuracy, stop_no_improvement):
        accuracy = test_accuracy or training_accuracy or []
        if len(accuracy) < stop_no_improvement + 1 or stop_no_improvement == 0:
            return True

        mean_array = accuracy[-10:]
        mean = sum(mean_array) / len(mean_array)

        return accuracy[-stop_no_improvement - 1] < max(accuracy[-1], mean)

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {
            "sizes": [l.size for l in self.layers],
            "weights": [l.weights.tolist() for l in self.layers[1:]],
            "biases": [l.weights.tolist() for l in self.layers[1:]],
            "cost": str(self.layers[-1].cost.__name__),
            "activation": [str(l.activation.__name__) for l in self.layers],
        }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    @classmethod
    def load(cls, filename: str):
        raise NotImplemented
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        cost = getattr(sys.modules[__name__], data["cost"])
        activation = getattr(sys.modules[__name__], data["activation"])
        net = cls(data["sizes"], cost=cost, activation=activation)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net
