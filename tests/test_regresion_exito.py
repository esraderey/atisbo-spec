from atisbo.env.mundo import aplicar, es_exito
from atisbo.env.trazas import generar


def test_exito_coherente_y_discrimina():
    eps = generar(1, 50, 0.05)
    assert any(not ep.exito for ep in eps)
    for ep in eps:
        assert ep.exito == es_exito(aplicar(*ep[-1]))
