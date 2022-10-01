import numpy as _np
import abc as _abc


class Cost(_abc.ABC):
    @staticmethod
    @_abc.abstractmethod
    def c(a, y) -> _np.ndarray:
        pass

    @staticmethod
    @_abc.abstractmethod
    def dcda(a, y) -> _np.ndarray:
        pass


class Activation(_abc.ABC):
    @staticmethod
    @_abc.abstractmethod
    def f(z) -> _np.ndarray:
        pass

    @staticmethod
    @_abc.abstractmethod
    def dfdz(z: _np.ndarray) -> _np.ndarray:
        pass


class Layer(_abc.ABC):
    @_abc.abstractmethod
    def __init__(
        self,
        size: int,
        activation: type[Activation],
        cost: type[Cost],
        p_dropout: float = 0,
    ):
        self.size = 0
        self.activation = Activation
        self.cost = Cost
        self.weights = _np.asarray([])
        self.biases = _np.asarray([])
        self.weights_velocity = _np.asarray([])
        self.biases_velocity = _np.asarray([])
        self.ready = False
        self.p_dropout = p_dropout

    @_abc.abstractmethod
    def setReady(self, is_ready: bool):
        pass

    @_abc.abstractmethod
    def delta(self, activation, y, z) -> _np.ndarray:
        pass

    @_abc.abstractmethod
    def feedforward(
        self, input: _np.ndarray, rng: _np.random.Generator, add_velocity: bool = False
    ) -> tuple[_np.ndarray, _np.ndarray]:
        pass


class CrossEntropyCost(Cost):
    @staticmethod
    def c(a, y):
        return -_np.nan_to_num((y) * _np.log(a) + (1 - y) * _np.log(1 - a)).sum()

    @staticmethod
    def dcda(a, y):
        return _np.nan_to_num((a - y) / (a * (1 - a)))


class WideCrossEntropyCost(Cost):
    @staticmethod
    def c(a, y):
        b = _np.nan_to_num((1 + y) * _np.log(1 + a) + (1 - y) * _np.log(1 - a))
        return -b.sum() / 2

    @staticmethod
    def dcda(a, y):
        return _np.nan_to_num((a - y) / (1 - a**2))


class QuadraticCost(Cost):
    @staticmethod
    def c(a, y):
        return 0.5 * _np.linalg.norm(a - y) ** 2

    @staticmethod
    def dcda(a, y):
        return a - y


class LogLikehoodCost(Cost):
    @staticmethod
    def c(a, y):
        return -_np.log(_np.take_along_axis(a, _np.argmax(y, 0), 0))

    @staticmethod
    def dcda(a, y):
        yy = _np.argmax(_np.asmatrix(y), 0)
        r = _np.zeros(a.shape)
        _np.put_along_axis(r, yy, _np.take_along_axis(a, yy, 0), 0)
        return r


class LinearCost(Cost):
    @staticmethod
    def c(a, y):
        return a - y

    @staticmethod
    def dcda(a, y):
        return 1

class SoftmaxActivation(Activation):
    @staticmethod
    def f(z):
        expd = _np.exp(z)
        return _np.dot(expd, _np.diag(1 / expd.sum(0)))

    @staticmethod
    def dfdz(z):
        pass


class SigmoidActivation(Activation):
    @staticmethod
    def f(z):
        return 1 / (1 + _np.exp(-z))

    @staticmethod
    def dfdz(z):
        d = 1 + _np.exp(-z)
        return _np.exp(-z) / d / d


class ReLUActivation(Activation):
    @staticmethod
    def f(z):
        z[z <= 0] = 0
        return z

    @staticmethod
    def dfdz(z):
        z[z > 0] = 1
        z[z <= 0] = 0
        return z


class ReSoftPlusActivation(Activation):
    @staticmethod
    def f(z):
        k = 1.3
        return _np.log(1 + _np.exp(k * z)) / k

    @staticmethod
    def dfdz(z):
        k = 1.3
        return 1 - 1 / (_np.exp(k * z) + 1)


class LinearActivation(Activation):
    @staticmethod
    def f(z):
        return z

    @staticmethod
    def dfdz(z):
        return _np.ones(z.shape)


class HiperbolicTangentActivation(Activation):
    @staticmethod
    def f(z):
        return _np.tanh(z)

    @staticmethod
    def dfdz(z):
        return 1 / _np.cosh(z) ** 2


class FullyConnectedLayer(Layer):
    def __init__(
        self,
        size,
        activation=Activation,
        cost=Cost,
        p_dropout=0.0,
    ):
        self.size = size
        self.activation = activation
        self.cost = cost
        self.weights = _np.asarray([])
        self.biases = _np.asarray([])
        self.weights_velocity = _np.asarray([])
        self.biases_velocity = _np.asarray([])
        self.ready = False
        self.p_dropout = p_dropout

    def setReady(self, is_ready):
        self.ready = is_ready

    def delta(self, activation, y, z):
        return self.cost.dcda(activation, y) * self.activation.dfdz(z)

    def feedforward(self, input: _np.ndarray, rng, add_velocity=False):
        assert self.ready != False, "Layer is not ready"
        ws, bs = self.weights, self.biases
        if add_velocity:
            ws = bs + self.weights_velocity
            bs = bs + self.biases_velocity
        z = _np.dot(self.weights, input) + self.biases
        activation = self.activation.f(z)

        mask = _np.mgrid[0 : z.shape[0], 0 : z.shape[1]][0]

        for c in mask.T:
            c[c < c.size * self.p_dropout] = 0
            c[c >= c.size * self.p_dropout] = 1

        mask = rng.permuted(1 - mask, axis=0).astype(bool)
        activation[mask] = 0
        z[mask] = 0
        return (z, activation)


class SoftmaxLayer(Layer):
    def __init__(
        self,
        size,
        activation=SoftmaxActivation,
        cost=LogLikehoodCost,
        p_dropout=0.0,
    ):
        self.size = size
        self.activation = activation
        self.cost = cost
        self.weights = _np.asarray([])
        self.biases = _np.asarray([])
        self.weights_velocity = _np.asarray([])
        self.biases_velocity = _np.asarray([])
        self.ready = False
        self.p_dropout = p_dropout

    def setReady(self, is_ready):
        self.ready = is_ready

    def delta(self, activation, y, z):
        if self.activation == SoftmaxActivation:
            return activation - y

        return self.cost.dcda(activation, y) * self.activation.dfdz(z)

    def feedforward(self, input: _np.ndarray, rng, add_velocity=False):
        assert self.ready != False, "Layer is not ready"
        ws, bs = self.weights, self.biases
        if add_velocity:
            ws = bs + self.weights_velocity
            bs = bs + self.biases_velocity

        z = _np.dot(self.weights, input) + self.biases
        activation = self.activation.f(z)

        mask = _np.mgrid[0 : z.shape[0], 0 : z.shape[1]][0]

        for c in mask.T:
            c[c < c.size * self.p_dropout] = 0
            c[c >= c.size * self.p_dropout] = 1

        mask = rng.permuted(1 - mask, axis=0).astype(bool)
        activation[mask] = 0
        z[mask] = 0
        return (z, activation)
