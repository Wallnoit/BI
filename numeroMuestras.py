

import math


def numeroMuestras(nConfianza, error):

    s = (math.log(2) - math.log(1-nConfianza))/(2*(error **(2)))
    return s


nConfianza = 0.95
error = 0.05


n = numeroMuestras(nConfianza, error)

print(f"El número de muestras necesario es: {n}")