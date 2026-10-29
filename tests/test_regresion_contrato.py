from atisbo.env.trazas import generar
from atisbo.estado import Accion, EstadoSim


def test_episodio_es_lista_de_pares():
    ep = generar(0, 2, 0.0)[0]
    assert isinstance(ep, list)
    assert len(ep) > 0
    for s, a in ep:
        assert isinstance(s, EstadoSim)
        assert isinstance(a, Accion)
    assert hasattr(ep, "perturbaciones")
    assert hasattr(ep, "exito")


def test_iniciales_pareados_entre_p():
    n = 10
    eps0 = generar(123, n, 0.0)
    eps5 = generar(123, n, 0.05)
    ini0 = [ep[0][0] for ep in eps0]
    ini5 = [ep[0][0] for ep in eps5]
    assert ini0 == ini5
