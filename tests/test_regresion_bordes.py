from atisbo.env.mundo import aplicar
from atisbo.estado import Accion, EstadoSim, H, W


def _cubos():
    return {"rojo": (0, 0), "verde": (11, 0), "azul": (0, 11)}


def test_bordes_inferior_derecho_y_tomar_sin_efecto():
    s_der = EstadoSim(bot=(W - 1, 5), carga=None, objetos=(), cubos=_cubos(), bloqueo=None, paso=0)
    assert aplicar(s_der, Accion.DER).bot == (W - 1, 5)
    s_aba = EstadoSim(bot=(5, H - 1), carga=None, objetos=(), cubos=_cubos(), bloqueo=None, paso=0)
    assert aplicar(s_aba, Accion.ABAJO).bot == (5, H - 1)
    s_car = EstadoSim(bot=(5, 5), carga="rojo", objetos=((1, "rojo", 5, 5, False),), cubos=_cubos(), bloqueo=None, paso=0)
    s_car2 = aplicar(s_car, Accion.TOMAR)
    assert (s_car2.carga, s_car2.objetos, s_car2.paso) == ("rojo", s_car.objetos, 1)
    s_cubo = EstadoSim(bot=(0, 0), carga=None, objetos=((1, "rojo", 0, 0, True),), cubos=_cubos(), bloqueo=None, paso=0)
    s_cubo2 = aplicar(s_cubo, Accion.TOMAR)
    assert (s_cubo2.carga, s_cubo2.objetos) == (None, s_cubo.objetos)
