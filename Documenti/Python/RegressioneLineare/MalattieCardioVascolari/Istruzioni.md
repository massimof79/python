Il reparto di prevenzione di un ospedale specializzato nella cura delle malattie dell'apparato circolatorio ha chiesto di sviluppare un sistema di per identificare pazienti a rischio di malattie cardiovascolari basandosi su parametri clinici standard.

L'obiettivo è creare un modello di classificazione binaria il quale, dato un paziente con i suoi parametri vitali e clinici, sia in grado di predire se è il paziente può sviluppare malattie cardiovascolari.

I parametri clinici dei pazienti sono
(eta - age): Età del paziente, espressa in anni. Variabile numerica continua; rappresenta un importante fattore di rischio cardiovascolare
(genere - sex: 1 = M, 0 = F): Sesso biologico del paziente. Variabile binaria, dove 1 indica maschio e 0 indica femmina.
(cp: 0–3): Tipo di dolore toracico (chest pain). Variabile categorica discreta:
            0 = angina tipica
            1 = angina atipicaì
            2 = dolore non anginoso
            3 = asintomatico
(rest bps): Pressione arteriosa a riposo, espressa in mmHg, misurata al momento del ricovero. Variabile numerica continua.
(chol): Livello di colesterolo sierico totale, espresso in mg/dl. Variabile numerica continua.
(fbs: >120 mg/dl): Glicemia a digiuno. Variabile binaria:
                    1 = glicemia a digiuno superiore a 120 mg/dl
                    0 = glicemia a digiuno inferiore o uguale a 120 mg/dl
(rest ecg: 0–2): Risultato dell’elettrocardiogramma a riposo. Variabile categorica:
                    0 = normale
                    1 = presenza di anomalie dell’onda ST-T
                    2 = ipertrofia ventricolare sinistra probabile o certa
(thala ch): Frequenza cardiaca massima raggiunta durante il test da sforzo. Variabile numerica continua, espressa in battiti al minuto
(exang: 1 = sì, 0 = no): Presenza di angina indotta dall’esercizio fisico. Variabile binaria.
(old peak): Depressione del tratto ST indotta dall’esercizio rispetto alla condizione di riposo. Variabile numerica continua; indica la gravità dell’ischemia miocardica durante lo sforzo.

            1. Scegliere il modello più opportuno per la predizione richiesta. 
            2. Addestrare il modello con dataset pubblico
            3. Implementare un programma python per predire tramite parametri di test se un paziente è a rischio di sviluppare malattie cardiovascolari.
