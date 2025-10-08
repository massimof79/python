# Classifica torneo calcio a 5

def calcola_classifica(partite):
    """
    Calcola la classifica dalle partite.
    Ogni partita: (squadra1, squadra2, gol1, gol2)
    Ritorna: lista di (squadra, punti, diff_reti, gol_fatti)
    """
    squadre = {}
    
    # Inizializziamo ogni squadra
    for s1, s2, _, _ in partite:
        if s1 not in squadre:
            squadre[s1] = {"punti": 0, "gol_fatti": 0, "gol_subiti": 0}
        if s2 not in squadre:
            squadre[s2] = {"punti": 0, "gol_fatti": 0, "gol_subiti": 0}
    
    # Elaboriamo ogni partita
    for s1, s2, g1, g2 in partite:
        # Aggiorniamo gol
        squadre[s1]["gol_fatti"] += g1
        squadre[s1]["gol_subiti"] += g2
        squadre[s2]["gol_fatti"] += g2
        squadre[s2]["gol_subiti"] += g1
        
        # Assegniamo punti (3 vittoria, 1 pareggio, 0 sconfitta)
        if g1 > g2:
            squadre[s1]["punti"] += 3
        elif g2 > g1:
            squadre[s2]["punti"] += 3
        else:
            squadre[s1]["punti"] += 1
            squadre[s2]["punti"] += 1
    
    # Creiamo la lista di classificazione
    classifica = []
    for nome, dati in squadre.items():
        diff_reti = dati["gol_fatti"] - dati["gol_subiti"]
        classifica.append((nome, dati["punti"], diff_reti, dati["gol_fatti"]))
    
    # Ordiniamo per: punti, differenza reti, gol fatti, nome
    classifica.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
    
    return classifica

def stampa_partite(partite):
    """Stampa le partite."""
    print("=== PARTITE ===")
    for s1, s2, g1, g2 in partite:
        print(f"{s1} - {s2} : {g1}-{g2}")
    print()

def stampa_classifica(classifica):
    """Stampa la classifica formattata."""
    print("=== CLASSIFICA ===")
    for i, (squadra, punti, diff, gol) in enumerate(classifica, 1):
        print(f"{i}. {squadra:8} | Punti: {punti:2} | Diff: {diff:+3} | Gol: {gol}")

# MAIN
partite = [
    ("3A", "4B", 2, 1),
    ("2C", "3A", 0, 0),
    ("4B", "2C", 3, 2),
    ("1D", "3A", 1, 4),
    ("4B", "1D", 2, 2),
    ("2C", "1D", 1, 3)
]

stampa_partite(partite)
classifica = calcola_classifica(partite)
stampa_classifica(classifica)