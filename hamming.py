def distancia_hamming(cadena1, cadena2):
  """
  Calcula la distancia de Hamming entre dos cadenas de texto.

  Args:
      cadena1 (str): La primera cadena de texto.
      cadena2 (str): La segunda cadena de texto.

  Returns:
      int: La distancia de Hamming entre las dos cadenas.
  """
  # Comprobamos si las cadenas tienen la misma longitud
  if len(cadena1) != len(cadena2):
    raise ValueError("Las cadenas deben tener la misma longitud")

  # Convertimos las cadenas a binario y rellenamos con ceros a la izquierda (fuera del for)
  binarios1 = [bin(ord(c))[2:].zfill(len(cadena1)) for c in cadena1]
  binarios2 = [bin(ord(c))[2:].zfill(len(cadena2)) for c in cadena2]

  # Inicializamos el contador de diferencias
  diferencias = 0

  print(binarios1)
  print(binarios2)

  # Recorremos las cadenas caracter por caracter
  for i in range(len(binarios1)):
    # Comparamos los bits correspondientes
    if binarios1[i] != binarios2[i]:
      print(f"Los bits {binarios1[i]} y {binarios2[i]} son diferentes")
      diferencias += 1

  # Devolvemos la distancia de Hamming
  return diferencias

# Ejemplo de uso
cadena1 = "holas"
cadena2 = "mundo"

distancia = distancia_hamming(cadena1, cadena2)
print(f"La distancia de Hamming entre '{cadena1}' y '{cadena2}' es: {distancia}")
