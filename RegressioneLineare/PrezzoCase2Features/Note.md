##########################
Random
np.random.seed(7)

appartiene alla libreria NumPy e serve a inizializzare il generatore di numeri casuali con un valore specifico chiamato seed (seme).

Per comprenderla bene occorre chiarire un punto: i numeri casuali generati da un computer non sono realmente casuali, ma sono prodotti da un algoritmo deterministico chiamato pseudo-random number generator. Questo algoritmo genera una sequenza di numeri che sembra casuale, ma che dipende completamente dal valore iniziale da cui parte, cioè il seed.

Quando si scrive:

np.random.seed(7)

si sta dicendo a NumPy:

“inizia la sequenza pseudo-casuale usando il valore iniziale 7”.


##########################

