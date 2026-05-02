from scipy.optimize import minimize
import numba
import numpy as np


@numba.njit
def cone_model(params, x, y):
    """
    Formula del cono empleada en la implementacion del ajuste del
    cono. Sirve para que scipy.minimize sepa cual es la formula del cono

    Args:
        params (list): la lista de los parametros
        x (np.ndarray): lass posiciones x de los datos
        y (np.ndarray): las posiciones y de los datos

    Returns:
        np.ndarray: la posicion z de los puntos
    """
    a, b, c, d = params
    # Formula generalizada de un cono
    return d - c * np.sqrt((x - a) ** 2 + (y - b) ** 2)


@numba.njit
def objective_function_v2(params, x, y, z):
    """
    Funcion a minimizar por minimize. Para el ajuste de
    un cono buscamos que su error cuadratico medio sea lo
    mas pequeño posible

    Args:
        params (list): lista de los parametros del cono
        x (np.ndarray): los valores de las x de los puntos
        y (np.ndarray): los valores de las y de los puntos
        z (np.ndarray): Los valores de las z de los puntos

    Returns:
        float: el error cuadratico medio
    """
    a, b = params
    # Original phase-1 fixed values. Tried c=2.78 (matched to real
    # IFS-08 cone) on 2026-04-29 — counter-intuitively made things
    # worse: the cost surface is flatter when c matches the data, so
    # phase 1 doesn't move much from the (mean_x, mean_y) init, and
    # phase 2 then recovers shallow c on noisy data. The c=5.5 init
    # forces phase 1 to walk the apex toward the cluster's peak (a
    # steep cone has a sharper minimum). The downstream validation
    # then catches the slightly-too-shallow phase 2 c via the
    # centroid-fallback path in cone_detection.py.
    c = 5.5
    d = 0.35
    z_pred = cone_model([a, b, c, d], x, y)
    return np.mean((z - z_pred) ** 2)


@numba.njit
def get_gammas(x, y, a, b):
    return np.sqrt((x - a) ** 2 + (y - b) ** 2)


@numba.njit
def lst_sqrs_fit(X, y):
    # X.T is a non-contiguous transpose view of a C-contiguous X; using
    # it directly in `@` triggers Numba's slow path and emits a
    # NumbaPerformanceWarning. Materialise once with ascontiguousarray.
    Xt = np.ascontiguousarray(X.T)
    return np.linalg.inv(Xt @ X) @ Xt @ y


def cone_fit_2params(data, solver="L-BFGS-B"):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Estos son los parametros con los que empieza
    # (estan puestos asi porque no estan muy lejos de los valores reales)
    a = np.mean(x)
    b = np.mean(y)

    try:
        result = minimize(
            objective_function_v2,
            [a, b],
            args=(x, y, z),
            method=solver,
            # tol=1e-14,
            # options={"maxiter": 10000},
        )
        a, b = result.x
        gamma = get_gammas(x, y, a, b)
        M = np.vstack((np.ones(len(data)), gamma)).T
        d, c = lst_sqrs_fit(M, z.T)
        return a, b, -c, d

        # print("a", a, "b", b, "c", c, "d", d, "error", result.fun)
        # print(result.fun)
    except Exception as e:
        print(e)
        a, b, c, d = (0, 0, 0, 0)

    return a, b, c, d
