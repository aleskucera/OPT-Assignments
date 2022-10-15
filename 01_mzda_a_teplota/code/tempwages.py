import numpy as np


def fit_wages(t: np.ndarray, M: np.ndarray) -> np.ndarray:
    """ Fit linear function to wages data
    :param t: temperatures
    :param M: wages
    :return: coefficients of linear function
    """
    # a*x = b
    a = np.vstack([np.ones(len(t)), t]).T
    b = M
    return np.linalg.lstsq(a, b, rcond=None)[0]


def wages_function(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """ Predict wages for given temperatures
    :param x: coefficients of linear function
    :param t: temperatures
    :return: wages
    """
    return x[0] + x[1] * t


def quarter2_2009(x: np.ndarray) -> np.ndarray:
    """ Predict wages for quarter 2 2009
    :param x: temperatures
    :return: wages
    """
    time = 2009.25
    return x[0] + x[1] * time


def fit_temps(T: np.ndarray, t: np.ndarray, omega: float) -> np.ndarray:
    """ Fit harmonic function to temperatures data
    :param T: years
    :param t: temperatures
    :param omega: angular frequency
    :return: coefficients of harmonic function
    """
    # a*x = b
    a = np.vstack([np.ones(len(t)), t, np.sin(omega * t), np.cos(omega * t)]).T
    b = T
    return np.linalg.lstsq(a, b, rcond=None)[0]


def temperature_function(x: np.ndarray, omega: float, t: np.ndarray) -> np.ndarray:
    """ Predict temperatures for given years
    :param x: coefficients of harmonic fu*2009.25
    :param omega: angular frequency
    :param t: years
    :return: temperatures
    """
    return x[0] + x[1] * t + x[2] * np.sin(omega * t) + x[3] * np.cos(omega * t)


def main():
    # load data
    wages = np.loadtxt("../data/mzdy.txt")
    t = wages[:, 0]
    M = wages[:, 1]

    # fit wages
    x = fit_wages(t, M)

    # predict wages
    M_pred = quarter2_2009(x)

    # plot wages
    import matplotlib.pyplot as plt
    plt.plot(t, M, 'o', label='original data')
    plt.plot(2009.25, M_pred, 'o', label='predicted wage')
    plt.legend()
    plt.show()

    # load data
    temps = np.loadtxt("../data/teplota.txt")
    t = temps[:, 0]
    T = temps[:, 1]
    omega = 2 * np.pi / 365
    x = fit_temps(T, t, omega)
    print(x.shape)

    # predict temperatures
    t2 = np.linspace(0, 1100, 5000)
    T_pred = temperature_function(x, omega, t2)

    # plot temperatures
    plt.plot(t, T, 'o', label='original data')
    plt.plot(t2, T_pred, 'r', label='predicted temperature')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
