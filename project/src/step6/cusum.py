import numpy as np  

class CUSUM:
    def __init__(self, M, eps, h):
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, samples):
        self.t += 1
        votes = []
        
        # generalization of cusum with multiple rewards
        for sample in samples:
            if self.t <= self.M:
                self.reference += sample / self.M
                votes.append(0)
            else:
                s_plus = (sample - self.reference) - self.eps
                s_minus = -(sample - self.reference) - self.eps
                self.g_plus = max(0, self.g_plus + s_plus)
                self.g_minus = max(0, self.g_minus + s_minus)
                detection = self.g_plus > self.h or self.g_minus > self.h
                votes.append(int(detection))

        # majority voting
        return sum(1 for elem in votes if elem == 1) >= len(votes) / 2
    
    def reset(self):
        self.t = 0
        self.g_minus = 0
        self.g_plus = 0