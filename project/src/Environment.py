import numpy as np

class Environment:

    def __init__(self, prices, prob_matrix, alphas, feature_1_dist, feature_2_dist):

        self.prices = prices #the four prices for each product, they could be fixed for whole project?
        #check if the prices are in increasing order? è scritto nella consegna

        self.items = np.arange(0,5,1) #the five items, fare da 1 a 5 come dice nella consegna?

        self.prob_matrix = prob_matrix #the graph probabilities, 3x5x5 matrix (3x since there will be one graph for each class of users)
        #self.margins = margins?? The margins associated to the prices are known but do we have to consider them? higher price -> higher margin?

        self.alphas = alphas #3x6 (3 class of users -> 3 sets of alpha)

        self.feature_1_dist = feature_1_dist #0.4
        self.feature_2_dist = feature_2_dist #0.8

        self.reservation_prices #????
        #devono essere fissi per ogni classe di utenti? o tipo delle medie con deviazione
        #da cui per ogni singolo utente ricavarsi il reservation price???????


    '''Computes the mapping between the user feature and the 
    corresponding class index. 
    Useful for access the right probabilities ecc...
    feature_1 == 0 and feature_2 == 0 -> class 0
    feature_1 == 1 and feature_2 == 1 -> class 1
    (feature_1 == 1 and feature_2 == 0) OR (feature_1 == 0 and feature_2) == 1 -> class 2 
    '''
    def user_class_mapping(self, feature_1, feature_2):
        if feature_1 == 0 and feature_2 == 0:
            return 0
        elif feature_1 == 1 and feature_2 == 1:
            return 1
        else:
            return 2

    #il corrispondente del metodo 'round'
    def purchase(self, item, price, user_class, reservation_price):
        #controlla se il reservation price dell'utente supera il prezzo dell'item
        if price < reservation_price:
            yes_or_no = np.random.binomial(1, self.probabilities[user_class][price])

    #qui immagino che ci debba essere un mapping fisso
    #1:2 con ad ogni primary una coppia di secondary da spawnare
    def get_secondary(self, primary):
        #es se il primary è l'item 0 ritorno i secondary 2 e 4
        #ovviamente anche l'ordine è importante per la questione dell'ordine di visualizzazione
        if primary == 0:
            return np.array([2,4])

    def click_on_secondary(self, user_class, bought_items):
        #usando le probabilità del grafo ci dice
        # la prossima mossa dell'utente (click su uno dei due secodnary o uscita direi)
        #l'uscita deve restituire -1 per l'uscita dal ciclo
        return 999






class Simulator:

    def __init__(self, days, users):
        self.days = days
        self.users = users #quanti giri fare dentro al grafo ogni giorno
        self.e = Environment(...)

        #self.badit del pricing
        #self.algoritmo per la social influence

    def run_simulation(self):

        for day in self.days:
            #cambiare prezzi in base ai bandit
            prices = 'array di prezzi'

            for user in self.users:
                #retrieve the user features -> user class
                feature_1 = np.random.choice([0,1],p=[1-self.e.feature_1_dist, self.e.feature_1_dist])
                feature_2 = np.random.choice([0,1],p=[1-self.e.feature_2_dist, self.e.feature_2_dist])
                user_class = self.e.user_class_mapping(feature_1, feature_2)

                #item di partenza usando gli alpha
                webpages = np.concatenate((np.array([-1]), self.e.items), axis=0)
                starting_point = np.random.choice(webpages, p=self.e.alphas[user_class])

                #maschera per salvarci quali items abbiamo già comprato
                bought_items = np.zeros(self.e.items.shape[0])

                #se l'utente non è andato dalla concorrenza
                if starting_point != -1:

                    primary = starting_point

                    #mi sembra sensato mettere il primary a -1 quando l'utente vuole uscire così da fermare il ciclo
                    while primary != -1:
                        #modellare con una coda con piazzamenti a random in testa
                        #per il discorso dei path multipli

                        #ci interessa solo se l'utente compra o anche quanti oggetti compra?
                        if self.e.purchase(primary, prices[user_class][primary], user_class):

                            bought_items[primary] = 1

                            #salvarsi dati per bandit del pricing
                            #per la social influence è imporatante salvarsi anche se l'utente ha comprato
                            #un certo oggetto?

                            secondary = self.e.get_secondary(primary)

                            primary = self.e.click_on_secondary(user_class, bought_items)

                            #salvarsi i dati per l'algoritmo della social influence

                            #TENERE CONTO DEI PATH MULTIPLI

                        else:
                            #l'utente non ha comprato quindi esce dal ciclo
                            #dobbiamo salvarci qualche dato per l'aggiornamento a fine giornata?
                            primary = -1

            #GIORNATA FINITA
            #aggiornare social influence
            #aggiornare algoritmo pricing



