import numpy as np

class Number_of_sold_items_estimator:

    def __init__(self, n_items, n_classes):
        self.values = np.ones((n_classes,n_items,2))
        self.collected_sold_items = [[[] for i in range(n_items)] for i in range(n_classes)]

    def update(self, item, sold_items, user_class=-1):
        if user_class == -1:
            # store the value
            for user_class in range(self.values.shape[0]):
                self.collected_sold_items[user_class][item].append(sold_items)

            #update the means
            for user_class in range(self.values.shape[0]):
                self.values[user_class][item][0] = np.mean(self.collected_sold_items[user_class][item])

        else:
            #TODO testare il funzionamento con le classi
            self.collected_sold_items[user_class][item].append(sold_items)
            self.values[user_class][item][0] = np.mean(self.collected_sold_items[user_class][item])
