# Creazione di un set
frutta = {"mela", "banana", "arancia"}
print(f"Set iniziale: {frutta}")

# Aggiunta di elementi
frutta.add("kiwi")
print(f"Dopo aver aggiunto 'kiwi': {frutta}")

# Tentativo di aggiungere un duplicato (non avrà effetto)
frutta.add("mela")
print(f"Dopo aver aggiunto 'mela' (duplicato): {frutta}")

# Rimozione di un elemento
frutta.remove("banana")
print(f"Dopo aver rimosso 'banana': {frutta}")

# Operazioni tra insiemi
frutta_estate = {"pesca", "albicocca", "mela"}
unione = frutta.union(frutta_estate) # o frutta | frutta_estate
intersezione = frutta.intersection(frutta_estate) # o frutta & frutta_estate
differenza = frutta.difference(frutta_estate) # o frutta - frutta_estate

print(f"Frutta estiva: {frutta_estate}")
print(f"Unione: {unione}")
print(f"Intersezione: {intersezione}")
print(f"Differenza: {differenza}")