import numpy as np

def clamp(x: float, rgbs: np.ndarray) -> np.ndarray:
    """
    Clampea cada componente RGB al máximo `x`.
    rgbs: array de forma (N, 3)
    """
    return np.minimum(rgbs, x)

def ecualization(x: float, rgbs: np.ndarray) -> np.ndarray:
    """
    Normaliza dividiendo por `x`.
    """
    return rgbs / x

def ecualization_inv(x: float, rgbs: np.ndarray) -> np.ndarray:
    """
    Revierta la ecualización, multiplicando por `x`.
    """
    return rgbs * x

def elevateRGBPoints(x: float, rgbs: np.ndarray) -> np.ndarray:
    """
    Eleva los valores RGB a la potencia 1/x.
    """
    return np.power(rgbs, 1.0 / x)

def gammaFunc(x: float, gamma: float, rgbs: np.ndarray) -> np.ndarray:
    """
    Aplica corrección gamma: ecualización -> elevación -> des-ecualización.
    """
    return ecualization_inv(1.0, elevateRGBPoints(gamma, ecualization(x, rgbs)))

def gammaFunc_inv(x: float, gamma: float, rgbs: np.ndarray) -> np.ndarray:
    """
    Revierta la corrección gamma: ecualización -> elevación inversa -> des-ecualización.
    """
    return ecualization_inv(1.0, elevateRGBPoints(1.0 / gamma, ecualization(x, rgbs)))
