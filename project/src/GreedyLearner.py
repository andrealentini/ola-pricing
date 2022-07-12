from Learner import *
import numpy as np

class Greedy_Learner(Learner):

    def __init__(self, prices, conversion_rates, n_items_to_buy_distr):
        super().__init__(prices)
        self.n_items = prices.shape[0]
        self.n_prices = prices.shape[1]

        self.solution = None

        # TODO: assert on sorting of prices

        # aggregated prices and conversion rates (mean of the three classes)
        self.prices = prices    # 5x4 matrix
        self.conversion_rates = conversion_rates    # 5x4 matrix
        self.n_items_to_buy_distr = n_items_to_buy_distr # 5x2 matrix
    
    def compute_margin(self, candidate):
        margin = 0
        for item, price in enumerate(candidate):
            margin += self.conversion_rates[item, price] * self.prices[item, price] * self.n_items_to_buy_distr[item, 0]
        return margin
    
    def pull_prices(self):
        if self.solution is not None:
            return self.solution
        
        optimal = np.array([0] * self.n_items)
        optimal_cumulative_expected_margin = self.compute_margin(optimal)

        while True:
            cumulative_expected_margins = []
            
            for idx, element in enumerate(optimal):
                # if assignment is not feasible it leads to zero increment
                if element == self.n_prices - 1:
                    cumulative_expected_margins.append(0)
                    continue
                
                # assign the following price to product
                candidate = np.array(optimal)
                candidate[idx] = element + 1

                # evaluate the increment and add to the candidates
                margin_increment = max(0, self.compute_margin(candidate) - optimal_cumulative_expected_margin)
                cumulative_expected_margins.append(margin_increment)

            # i-th increment corresponds to assigning next price to i-th product
            candidate_product = np.argmax(cumulative_expected_margins)

            # if highest increment is zero then stop
            if cumulative_expected_margins[candidate_product] == 0:
                break
            
            optimal[candidate_product] += 1
            optimal_cumulative_expected_margin += cumulative_expected_margins[candidate_product]
        
        self.solution = optimal
        return self.solution

    def update(self, pulled_arms, rewards):
        pass