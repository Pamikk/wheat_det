import numpy as np

class tryexp:
    name = 'fff'
    def __init__(self,nn):
        self.a = nn
    def pp(self):
        print(self.a)
        print(tryexp.name)
tmp = tryexp('fafsse')
print(tmp.name)
tmp.pp()