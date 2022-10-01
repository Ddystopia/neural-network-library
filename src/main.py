import numpy as np
import mnist_loader
import network
from classes import (
    CrossEntropyCost,
    SigmoidActivation,
    ReLUActivation,
    QuadraticCost,
    LinearActivation,
    LinearCost,
    SoftmaxLayer,
    WideCrossEntropyCost,
    HiperbolicTangentActivation,
    ReSoftPlusActivation,
)
from network import FullyConnectedLayer


def main():
    training, validation, test = mnist_loader.load_data_wrapper("../data/mnist.pkl.gz")

    print("Sigmoid + Softmax")
    net = network.Network(
        layers=[
            FullyConnectedLayer(size=784),
            FullyConnectedLayer(size=100, activation=SigmoidActivation),
            FullyConnectedLayer(size=30, activation=SigmoidActivation),
            SoftmaxLayer(size=10),
        ],
        save=True,
    )
    net.SGD(
        training_data=training,
        # epochs=30,
        mini_batch_size=20,
        eta=0.15,
        lmbda=0.12,
        mu=0.6,
        stop_no_improvement=15,
        test_data=test,
        monitor_test_accuracy=True,
        # monitor_test_cost=False,
        # monitor_training_accuracy=True,
        # monitor_training_cost=True,
        not_greather_random_reinitialize=1000,
    )


try:
    main()
except KeyboardInterrupt:
    print("\nKeyboardInterrup")
    exit()
