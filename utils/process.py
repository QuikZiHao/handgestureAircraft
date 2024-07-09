import numpy as np
import torch.nn

class Proceed:
    def __init__(self, func_amt:int, remember_stamp:int=100, proceed_score:float= 10.0):
        self.func_amt = func_amt
        self.remember_stamp = remember_stamp
        self.proceed_score = proceed_score
        self.stamp = 0
        self.conf_array = self.create_array()
        self.flag = True
        self.cooldown = 0
    
    def create_array(self) ->np.ndarray:
        temp = np.zeros(self.func_amt, dtype=float)
        return temp
    
    def step(self, class_idx:int, score:float) ->int:
        if self.flag == False:
            print('cooldown')    
            self.cooldown += 1
            if self.cooldown > 20:
                self.flag = True
                self.cooldown = 0
            return -1
        else:
            self.stamp += 1
            if self.stamp == self.remember_stamp:
                self.initial()
                return -1
            self.conf_array[class_idx] += score
            return self.check(class_idx)
    
    def initial(self):
        self.stamp = 0
        self.conf_array.fill(0.0)

    def check(self, class_idx:int) ->int:
        if self.conf_array[class_idx] >= self.proceed_score:
            self.initial()
            self.flag = False
            return class_idx
        return -1     
        