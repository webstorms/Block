from fastsnn.benchmark import LayerBenchmarker


def test_method():
    def validate(method):
        benchmark = LayerBenchmarker(method, t_len=16, n_in=10, n_hidden=20, n_layers=1, heterogeneous_beta=False, beta_requires_grad=False, batch_size=4)
        assert benchmark._get_description()["method"] == method
        assert benchmark._model._layers[0]._method == method
    validate("fast_naive")
    validate("standard")


def test_t_len():
    def validate(t_len):
        benchmark = LayerBenchmarker("fast_naive", t_len=t_len, n_in=10, n_hidden=20, n_layers=1, heterogeneous_beta=False, beta_requires_grad=False, batch_size=4)
        assert benchmark._get_description()["t_len"] == t_len
        assert benchmark._model._layers[0]._t_len == t_len
    validate(16)
    validate(32)


def test_units():
    def validate(units):
        benchmark = LayerBenchmarker("fast_naive", t_len=32, n_in=10, n_hidden=units, n_layers=1, heterogeneous_beta=False, beta_requires_grad=False, batch_size=4)
        assert benchmark._get_description()["units"] == units
        assert benchmark._model._layers[0]._n_out == units
    validate(10)
    validate(100)


def test_n_layers():
    def validate(n_layers):
        benchmark = LayerBenchmarker("fast_naive", t_len=32, n_in=10, n_hidden=100, n_layers=n_layers, heterogeneous_beta=False, beta_requires_grad=False, batch_size=4)
        assert benchmark._get_description()["layers"] == n_layers
        assert len(benchmark._model._layers) == n_layers
    validate(1)
    validate(5)


def test_heterogeneous_beta():
    def validate(heterogeneous_beta):
        benchmark = LayerBenchmarker("fast_naive", t_len=32, n_in=10, n_hidden=100, n_layers=1, heterogeneous_beta=heterogeneous_beta, beta_requires_grad=False, batch_size=4)
        assert benchmark._get_description()["heterogeneous_beta"] == heterogeneous_beta
        if heterogeneous_beta:
            assert len(benchmark._model._layers[0].beta) == 100
        else:
            assert len(benchmark._model._layers[0].beta) == 1

    validate(False)
    validate(True)


def test_beta_requires_grad():
    def validate(beta_requires_grad):
        benchmark = LayerBenchmarker("fast_naive", t_len=32, n_in=10, n_hidden=100, n_layers=1, heterogeneous_beta=False, beta_requires_grad=beta_requires_grad, batch_size=4)
        assert benchmark._get_description()["beta_requires_grad"] == beta_requires_grad
        assert benchmark._model._layers[0].beta.requires_grad == beta_requires_grad
    validate(False)
    validate(True)


def test_batch_size():
    def validate(batch_size):
        benchmark = LayerBenchmarker("fast_naive", t_len=32, n_in=10, n_hidden=100, n_layers=1, heterogeneous_beta=False, beta_requires_grad=False, batch_size=batch_size)
        assert benchmark._get_description()["batch"] == batch_size
    validate(16)
    validate(32)
