from typing import Union, List
import numpy as np

def h2(p):
    if p == 0 or p == 1:
        h = 0
    else:
        h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    return h

def p1(a: str, p: float) -> Union[float, str]:

    if (p < 0 or p > 1) or (a != 'CBS' and a != 'CBA'):
        return 'Date de intrare invalide!'

    if a == 'CBS':
        return 1 - h2(p)
    elif a == 'CBA':
        return 1 - p

#print(p1('CBA', -3))

def p2(lp: List[float]) -> Union[List[float], str]:

    l = []

    for x in lp:
        if x < 0 or x > 1:
            return 'Date de intrare invalide!'
        elif x == 0:
            l.append(np.log2(3) + (1 - x) * np.log2(1 - x))
        elif x == 1:
            l.append(np.log2(3) + x * np.log2(x/2))
        else:
            l.append(np.log2(3) + (1 - x) * np.log2(1 - x) + x * np.log2(x/2))

    return l

# np.log2(3) + (1 - p) * np.log2(1 - p) + p * np.log2(p/2)

#print(' '.join(F'{x:.6f}' for x in p2(np.linspace(0.0, 1.0, 10))))

def p3(p: float) -> List[float]:
    return [h2((1 / 2) * p) - p * h2(1 / 2), np.log2(5 / 4), 2 / 5]

def p4(Px: float, N: float, W: float) -> float:
    return W * np.log2(1 + Px / N)

print(p4(1, 2, 3))
