import numpy as np

def bin2dec(a):
    b = 0

    for p in range(len(a)):
        b += a[-p - 1] * (2 ** p)

    return np.int(b)

def grad(p):
    p = np.poly1d(np.flipud(p))
    return p.order


def X(m):
    X = np.zeros(m + 1, dtype=int)
    X[m] = 1
    return X

class GF2m:
    def __init__(self, g):
        self.g = g
        self.m = grad(g)
        self.n = np.power(2, self.m) - 1
        self.k = self.n - self.m
        # print('n:', self.n)
        self.p = self.adunare_polinoame(X(self.n), X(0))
        (self.h, _) = self.divizare_polinoame(self.p, self.g)

    def adunare_polinoame(self, a, b):
        s = np.mod(np.flipud(np.polyadd(np.flipud(a), np.flipud(b))), 2)
        return s.astype(int)

    def inmultire_polinoame(self, a, b):
        p = np.mod(np.flipud(np.polymul(np.flipud(a), np.flipud(b))), 2)
        if grad(p) > self.n - 1:
            for i in range(self.n, grad(p) + 1):
                p[i - self.n] = np.mod(p[i - self.n] + p[i], 2)
                p[i] = 0
        p = p[0: grad(p) + 1]
        return p

    def divizare_polinoame(self, a, b):
        (cat, rest) = np.polydiv(np.flipud(a), np.flipud(b))
        cat = np.mod(np.flipud(cat), 2)
        rest = np.mod(np.flipud(rest), 2)
        return cat.astype(int), rest.astype(int)


def g1_encode(text: str) -> np.ndarray:
    g = np.array([1, 1, 0, 1])
    gf2m = GF2m(g)

    v = []
    cuvinte = []

    separator = " "

    for x in text:
        ascii_code = ord(x)
        integer_value = ascii_code - ord('A')
        c = format(integer_value, "b").zfill(4)

        value = separator.join(c)
        v.append(np.flipud(list(map(int, value.split()))))

    for line in v:
        xi = gf2m.inmultire_polinoame(line, [0, 0, 0, 1])
        (_, control) = gf2m.divizare_polinoame(xi, g)
        cuvant = gf2m.adunare_polinoame(control, xi)

        cuvant = gf2m.adunare_polinoame(cuvant, [0,0,0,0,0,0,0])

        cuvinte.append(cuvant)

    return np.array(cuvinte)

def g1_decode(code_matrix: np.ndarray) -> str:
    g = np.array([1, 1, 0, 1])
    gf2m = GF2m(g)

    result = ''

    ver = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 1, 0],
             [0, 1, 1],
             [1, 1, 1],
             [1, 0, 1]]

    for line in code_matrix:
        (_, rest) = gf2m.divizare_polinoame(line, g)
        rest = gf2m.adunare_polinoame(rest, [0,0,0])
        pos = 0

        for line_ver in ver:
            if (line_ver == list(rest)):
                line[pos] = not line[pos]
            else:
                pos += 1

        result += chr(bin2dec(np.flipud(line[3:7])) + ord('A'))

    return result




temp_vec = g1_encode('ABCDEFGHIJKLMNOP')

temp_vec[3, 4] = not temp_vec[3, 4]


print(g1_decode(temp_vec))

