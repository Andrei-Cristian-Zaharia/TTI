import numpy as np

class Hamming31:

    @staticmethod
    def encode(text: str) -> np.ndarray:
        return np.asanyarray([3 * [int(bit)] for bit in text])

    @staticmethod
    def decode(code_matrix: np.ndarray) -> str:

        def decode_code(code: np.ndarray) -> str:
            return '0' if np.count_nonzero(code != [0, 0, 0]) <= 1 else '1'

        return ''.join(np.apply_along_axis(decode_code, 1, code_matrix))


def h31_encode(text: str) -> np.ndarray:
    return Hamming31.encode(text)


def h31_decode(code_matrix: np.ndarray) -> str:
    return Hamming31.decode(code_matrix)

class Hamming84:

    @staticmethod
    def encode(text: str) -> np.ndarray:

        def to_number(char: str) -> int:
            return ord(char) - ord('A')

        def dec_to_bin(dec: int) -> np.ndarray:
            return np.asanyarray(list(np.binary_repr(dec).zfill(4))).astype(np.int8)

        def encode_char(char: str) -> np.ndarray:
            i3, i5, i6, i7 = dec_to_bin(to_number(char))
            c4 = i5 ^ i6 ^ i7
            c2 = i3 ^ i6 ^ i7
            c1 = i3 ^ i5 ^ i7
            c8 = i3 ^ i5 ^ i6
            return np.asanyarray([c1, c2, i3, c4, i5, i6, i7, c8])

        return np.asanyarray([encode_char(char) for char in text])

    @staticmethod
    def decode(code_matrix: np.ndarray) -> str:

        def to_char(number: int) -> str:
            return chr(number + ord('A'))

        def bin_to_dec(bin: np.ndarray) -> int:
            return int(''.join(map(str, bin)), 2)

        def decode_char(code: np.ndarray) -> str:
            H = np.array([[0, 0, 0, 1, 1, 1, 1, 0],
                          [0, 1, 1, 0, 0, 1, 1, 0],
                          [1, 0, 1, 0, 1, 0, 1, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1]])

            syndrome = (H @ code.T) % 2
            if np.count_nonzero(syndrome) == 0:
                return to_char(bin_to_dec(np.asanyarray([code[2],  code[4], code[5], code[6]])))
            elif syndrome[3] == 0:
                return '*'
            else:
                error_pos = bin_to_dec(syndrome[:3]) - 1
                correct_code = code.copy()
                correct_code[error_pos] = (code[error_pos] + 1) % 2
                return to_char(bin_to_dec(np.asanyarray([correct_code[2], correct_code[4], correct_code[5], correct_code[6]])))

        return ''.join((decode_char(code) for code in code_matrix))


def h84_encode(text: str) -> np.ndarray:
    return Hamming84.encode(text)


def h84_decode(code_matrix: np.ndarray) -> str:
    return Hamming84.decode(code_matrix)
