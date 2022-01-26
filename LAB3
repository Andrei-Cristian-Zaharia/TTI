import heapq
from typing import List

import numpy as np


class Node:
    def __init__ (self, p, s):
        self.left = None
        self.middle = None
        self.right = None
        self.prob = p
        self.symbol = s

    def __lt__(self, other) :
        return self.prob < other.prob

    def __repr__(self) :
        return "Node({},{},{})".format(repr([self.prob, self.symbol]), ..., repr(self.left), repr(self.right))

def HuffmanTree(SP) :
    pq = []

    for symbol, prob in SP . items():
        pq.append(Node(prob, symbol))
        heapq.heapify(pq)

    while len(pq) > 1:
        n1 = heapq.heappop(pq)
        n2 = heapq.heappop(pq)

        top = Node(n1.prob+n2.prob, n1.symbol+n2.symbol)

        top.left = n1
        top.right = n2
        heapq.heappush(pq, top)
    return pq

def HuffmanTree3(SP) :
    pq = []

    for symbol, prob in SP . items():
        pq.append(Node(prob, symbol))
        heapq.heapify(pq)

    while len(pq) > 1:
        n1 = heapq.heappop(pq)
        n2 = heapq.heappop(pq)
        n3 = heapq.heappop(pq)

        top = Node(n1.prob+n2.prob + n3.prob, n1.symbol+n2.symbol+n3.symbol)

        top.middle = n3
        top.left = n1
        top.right = n2
        heapq.heappush(pq, top)
    return pq

def encode(dic_code, root, code):
    if root.left is None and root.right is None:
        dic_code[root.symbol] = code
    else:
        encode(dic_code, root.left, code+'0')
        encode(dic_code, root.right, code+'1')

def encode3(dic_code, root, code):
    if root.left is None and root.right is None and root.middle is None:
        dic_code[root.symbol] = code
    else:
        encode3(dic_code, root.left, code+'0')
        encode3(dic_code, root.middle, code+'1')
        encode3(dic_code, root.right, code + '2')

def h2(p):
    h = 0
    for x in p:
        if x == 0 or x == 1:
            h += 0
        else:
            h += -x * np.log2(x)

    return h

def p1(S: List[str], P: List[float]) -> List[float]:

    SP = dict((S[i], P[i]) for i in range(len(S)))
    dic_code = dict((S[i], '') for i in range(len(S)))

    PQ = HuffmanTree(SP)
    encode(dic_code=dic_code, root=PQ[0], code='')

    lungimea_medie = 0

    for i in range(len(S)):
        l = len(dic_code[S[i]])
        p = P[i]
        lungimea_medie += p * l

    l_min = h2(P) / np.log2(2)

    eficienta = l_min/lungimea_medie

    K = 0

    for i in range(len(S)):
        l = len(dic_code[S[i]])
        K += 2 ** -l

    return [lungimea_medie, eficienta, K]

def p2(S: List[str], P: List[float]) -> float:

    if (len(S) - 3) % 2 == 1:
        S.append("S")
        P.append(0)

    SP = dict((S[i], P[i]) for i in range(len(S)))
    dic_code = dict((S[i], '') for i in range(len(S)))

    PQ = HuffmanTree3(SP)
    encode3(dic_code=dic_code, root=PQ[0], code='')

    lungimea_medie = 0

    for i in range(len(S)):
        l = len(dic_code[S[i]])
        p = P[i]
        lungimea_medie += p * l

    return lungimea_medie

def prelucreaza_text(text: str) -> {}:

    dict = {}

    for x in text:
        if x in dict:
            dict[x] += 1
        else:
            dict[x] = 1

    for x in dict:
        dict[x] = dict[x] / len(text)

    return dict

def p3(text: str) -> List[int]:

    SP = prelucreaza_text(text)

    dic_code = dict((x, '') for x in SP)

    PQ = HuffmanTree(SP)
    encode(dic_code=dic_code, root=PQ[0], code='')

    l = 0

    for x in dic_code:
        l += len(dic_code[x]) * SP[x] * len(text)

    s = len(text) * np.ceil(np.log2(len(SP)))

    return [int(l), int(s)]

S3 = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
P3 = [0.63, 0.12, 0.02, 0.06, 0.09, 0.04, 0.04]

S2 = ['S1', 'S2']
P2 = [0.50, 0.50]

S1 = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
P1 = [0.25, 0.20, 0.20, 0.10, 0.10, 0.10, 0.05]

S = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
P = [0.30, 0.10, 0.05, 0.25, 0.20, 0.10]

#print(' '.join(F'{v:.3f}' for v in p1(S,P)))
#print(F'{p2(S3,P3):.3f}')

print(p3('Ana are mere. Mihai Stanciu nu suge pula așa de mult. Buri intră în depresie.'))
