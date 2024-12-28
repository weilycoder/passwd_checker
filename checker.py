from __future__ import annotations

import math, itertools

from collections import deque, defaultdict

from typing import Generator, cast


class PatternID:
    LOWER_ALPHA = "L"
    UPPER_ALPHA = "U"
    DIGIT = "D"
    SPECIAL = "S"
    HIGH = "H"
    OTHER = "X"

    PINYIN = "P"
    POPULAR = "W"
    ADJACENCY = "A"
    REPETITION = "R"
    NUMBER = "N"
    DIFF_SEQ = "C"

    ALL = "LUDSHXPWARNC"


NOT_CHAR = 0x110000
DIGITS = "0123456789"
LOWER_CASE = "abcdefghijklmnopqrstuvwxyz"
UPPER_CASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ASCII_SPECIAL_CHARS = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
HIGH_ANSI_CHARS = "".join(
    map(
        chr,
        itertools.chain(
            range(0x00A1, 0x00AD),
            range(0x00AE, 0x0100),
        ),
    )
)

ASCII_LEET_MAP = {
    "4": "a",
    "@": "a",
    "?": "a",
    "^": "a",
    "8": "b",
    "(": "c",
    "{": "c",
    "[": "c",
    "<": "c",
    "3": "e",
    "&": "e",
    "6": "g",
    "9": "g",
    "#": "h",
    "1": "i",
    "!": "i",
    "|": "i",
    "0": "o",
    "*": "o",
    "$": "s",
    "5": "s",
    "+": "t",
    "7": "t",
    "%": "x",
    "2": "z",
}

HIGH_LEET_MAP = {
    "\u00AA": "a",
    "\u00DF": "b",
    "\u00A2": "c",
    "\u00A9": "c",
    "\u00C7": "c",
    "\u00E7": "c",
    "\u00D0": "d",
    "\u00F0": "d",
    "\u20AC": "e",
    "\u00A3": "e",
    "\u00A1": "i",
    "\u00A6": "i",
    "\u00D1": "n",
    "\u00F1": "n",
    "\u00A4": "o",
    "\u00B0": "o",
    "\u00D8": "o",
    "\u00F8": "o",
    "\u00AE": "r",
    "\u00A7": "s",
    "\u00B5": "u",
    "\u00D7": "x",
    "\u00A5": "y",
    "\u00DD": "y",
    "\u00FD": "y",
    "\u00FF": "y",
}


def add_type_tag[T](tag: str, data: list[tuple[T, int, int, float]]):
    return [(tag, *x) for x in data]


def decode_phonetic_alpha(ch: str):
    c = ord(ch)
    if 0xC0 <= c <= 0xC6 or 0xE0 <= c <= 0xE6:
        return "a"
    if 0xC8 <= c <= 0xCB or 0xE8 <= c <= 0xEB:
        return "e"
    if 0xCC <= c <= 0xCF or 0xEC <= c <= 0xEF:
        return "i"
    if 0xD2 <= c <= 0xD6 or 0xF2 <= c <= 0xF6:
        return "o"
    if 0xD9 <= c <= 0xDC or 0xF9 <= c <= 0xFC:
        return "u"
    return ch


def decode_leet_char(ch: str):
    assert len(ch) == 1
    if ch in ASCII_LEET_MAP:
        return ASCII_LEET_MAP[ch]
    if ch in HIGH_LEET_MAP:
        return HIGH_LEET_MAP[ch]
    return decode_phonetic_alpha(ch)


def decode_leet(s: str):
    return "".join(map(decode_leet_char, s))


def hamming_dist(x: str, y: str, part: slice[int, int, int] = slice(None)):
    assert len(x) == len(y)
    return sum(map(lambda a, b: a != b, x[part], y[part]))


def log_comb(n: int, k: int):
    assert k <= n
    ret = 0
    k = min(k, n - k)
    for i in range(n, n - k, -1):
        ret += math.log(i)
    for i in range(1, k + 1):
        ret -= math.log(i)
    return ret


class Trie[T]:
    def __init__(self, fail: Trie | None = None):
        self.fail = self if fail is None else fail
        self.__locked = False
        self.children: dict[str, Trie] = {}
        self.end_of_word = ""
        self.other: T = cast(T, None)

    @property
    def locked(self):
        return self.__locked

    def lock(self):
        assert not self.locked
        self.__locked = True

    def build(self):
        self.lock()
        queue = deque(self.children.values())
        while queue:
            u = queue.popleft()
            for k in u.children:
                u[k].fail = u.fail[k]
                queue.append(u[k])

    def query(self, word: str, *, limit: int = 5):
        assert self.locked
        words: Trie[tuple[int, int]] = Trie()
        it = self
        for i in range(len(word)):
            c = word[i]
            it = it[c]
            l = len(it.end_of_word)
            if l >= limit:
                words.insert(it.end_of_word, (i - l + 1, l))
        return list(words.get_words1(only_leaf=True))

    def insert(self, word: str, other: T = None):
        assert not self.locked
        it = self
        for c in word:
            if c not in it.children:
                it.children[c] = Trie(self)
            it = it.children[c]
        it.end_of_word = word
        it.other = other

    def get_words(self, only_leaf: bool = False) -> Generator[str]:
        if self.end_of_word and (not only_leaf or not self.children):
            yield self.end_of_word
        for _, v in self.children.items():
            yield from v.get_words(only_leaf=only_leaf)

    def get_words1(self, only_leaf: bool = False) -> Generator[T]:
        if self.end_of_word and (not only_leaf or not self.children):
            yield self.other
        for _, v in self.children.items():
            yield from v.get_words1(only_leaf=only_leaf)

    def get_words2(self, only_leaf: bool = False) -> Generator[tuple[str, T]]:
        if self.end_of_word and (not only_leaf or not self.children):
            yield self.end_of_word, self.other
        for _, v in self.children.items():
            yield from v.get_words2(only_leaf=only_leaf)

    def __getitem__(self, key: str) -> Trie:
        if key not in self.children:
            if self == self.fail:
                return self
            self.children[key] = self.fail[key]
        return self.children[key]


class CharType:
    def __init__(self, alphabet: str | int, consecutive: bool = False):
        self.alphabet = self.ch_range = None
        if isinstance(alphabet, int):
            self.ch_cnt = alphabet
        else:
            self.alphabet = alphabet
            if consecutive:
                self.ch_range = range(ord(self.alphabet[0]), ord(self.alphabet[-1]) + 1)
                self.alphabet = "".join(map(chr, self.ch_range))
            self.ch_cnt = len(self.alphabet)
        self.ch_size = math.log(self.ch_cnt)

    @property
    def consecutive(self):
        return self.ch_range is not None

    def __contains__(self, ch: str):
        if self.ch_range is not None:
            return ord(ch) in self.ch_range
        assert self.alphabet is not None
        return ch in self.alphabet


class CharTypesTable:
    SPECIAL_LEN = len(ASCII_SPECIAL_CHARS)
    HIGH_LEN = len(HIGH_ANSI_CHARS)

    char_type: list[tuple[str, CharType]] = [
        (PatternID.DIGIT, CharType(DIGITS, True)),
        (PatternID.LOWER_ALPHA, CharType(LOWER_CASE, True)),
        (PatternID.UPPER_ALPHA, CharType(UPPER_CASE, True)),
        (PatternID.SPECIAL, CharType(ASCII_SPECIAL_CHARS)),
        (PatternID.HIGH, CharType(HIGH_ANSI_CHARS)),
        (PatternID.OTHER, CharType(NOT_CHAR - 26 * 2 - 10 - SPECIAL_LEN - HIGH_LEN)),
    ]

    @classmethod
    def get_char_type(cls, ch: str):
        for tp in cls.char_type[:-1]:
            if ch in tp[1]:
                return tp
        return cls.char_type[-1]

    @classmethod
    def get_type(cls, tp: str):
        for x in cls.char_type:
            if x[0] == tp:
                return x
        return cls.char_type[-1]


class EntropyEncoder:
    def __init__(
        self,
        alphabetType: str,
        baseWeight: float,
        charWeight: float,
        occExclThreshold: int,
    ):
        self.strAlph = alphabetType
        self.baseWeight = baseWeight
        self.charWeight = charWeight
        self.occExclThreshold = occExclThreshold

        self.dHisto: defaultdict[str, int] = defaultdict(int)

    def reset(self):
        self.dHisto.clear()

    def write(self, ch: str):
        assert ch in self.strAlph
        self.dHisto[ch] += 1

    def get_output_size(self):
        tot_weight = self.baseWeight * len(self.strAlph)
        for u in self.dHisto.values():
            if u > self.occExclThreshold:
                tot_weight += (u - self.occExclThreshold) * self.charWeight

        size = 0.0
        for u in self.dHisto.values():
            weight = self.baseWeight
            if u > self.occExclThreshold:
                weight += (u - self.occExclThreshold) * self.charWeight
            size -= u * math.log(weight / tot_weight)

        return size


class EntropyEncoderTable:
    def __init__(self):
        self.encs: dict[str, EntropyEncoder] = {}

    def addEncoder(self, typeId: str, enc: EntropyEncoder):
        self.encs[typeId] = enc

    def reset(self):
        for enc in self.encs.values():
            enc.reset()

    def write(self, typeId: str, ch: str):
        ec = self.encs.get(typeId)
        if ec is None:
            return False
        ec.write(ch)
        return True

    def get_output_size(self):
        size = 0.0
        for enc in self.encs.values():
            size += enc.get_output_size()
        return size


def computePathCost(
    path: tuple[tuple[str, int | None, int, int, float], ...],
    passwd: str,
    ecPattern: EntropyEncoder,
    mcData: EntropyEncoderTable,
):
    mcData.reset()
    ecPattern.reset()
    for pi in path:
        ecPattern.write(pi[0])
    cost = ecPattern.get_output_size()
    for pi in path:
        if pi[4] is math.nan:
            cost += CharTypesTable.get_type(pi[0])[1].ch_size
        else:
            ch = passwd[pi[2]]
            if not mcData.write(pi[0], ch):
                cost += pi[4]
    cost += mcData.get_output_size()
    return cost


class Check_Popular:
    def __init__(self, table: str | list[str]):
        self.table = Trie()
        self.len_cnt: defaultdict[int, int] = defaultdict(int)
        if isinstance(table, str):
            with open(table, "r", encoding="utf-8") as file:
                for line in file:
                    word = line.strip()
                    self.table.insert(word)
                    self.len_cnt[len(word)] += 1
        else:
            for word in table:
                self.table.insert(word)
                self.len_cnt[len(word)] += 1
        self.table.build()

    def check_popular(self, passwd: str, *, limit: int = 5, leet_cost: float = 1.5):
        ret: list[tuple[None, int, int, float]] = []
        leet_pwd = decode_leet(passwd.lower())
        for pos, length in self.table.query(leet_pwd, limit=limit):
            dist = hamming_dist(passwd, leet_pwd, slice(pos, pos + length))
            cost = math.log(self.len_cnt[length])
            cost += log_comb(len(passwd), length)
            cost += dist * leet_cost
            ret.append((None, pos, length, cost))
        return ret


class Check_Adjacency:
    def __init__(self, adj_path: str | list[tuple[str, str]]):
        self.alpha: set[str] = set()
        self.adj: set[frozenset[str]] = set()

        if isinstance(adj_path, str):
            with open(adj_path, "r", encoding="utf-8") as file:
                for line in file:
                    x, y = line.split()
                    self.alpha.update(x, y)
                    self.adj.add(frozenset((x, y)))
        else:
            for x, y in adj_path:
                self.alpha.update(x, y)
                self.adj.add(frozenset((x, y)))

        self.avg_edges = len(self.adj) / len(self.alpha)
        self.edge_size = math.log(self.avg_edges)

    def check_adj(self, passwd: str, *, limit: int = 3):
        cur = ""
        words: list[tuple[None, int, int, float]] = []
        for i in range(1, len(passwd)):
            x, y = passwd[i - 1], passwd[i]
            if frozenset((x.lower(), y.lower())) in self.adj:
                if not cur:
                    cur = x
                cur += y
            else:
                if len(cur) >= limit:
                    cost = self.edge_size * len(cur)
                    words.append((None, i - len(cur), len(cur), cost))
                cur = ""
        if len(cur) >= limit:
            cost = self.edge_size * len(cur)
            words.append((None, len(passwd) - len(cur), len(cur), cost))
        return words


class Checker:
    def __init__(
        self,
        *,
        adj_path: str | None = None,
        pinyin_path: str | None = None,
        popular_path: str | None = None,
    ):
        self.adj = Check_Adjacency(adj_path) if adj_path is not None else None
        self.pinyin = Check_Popular(pinyin_path) if pinyin_path is not None else None
        self.popular = Check_Popular(popular_path) if popular_path is not None else None

    def check_adj(self, passwd: str, *, limit: int = 3):
        if self.adj is None:
            return cast(list[tuple[None, int, int, float]], [])
        return self.adj.check_adj(passwd, limit=limit)

    def check_pinyin(self, passwd: str, *, limit: int = 3, leet_cost: float = 1.5):
        if self.pinyin is None:
            return cast(list[tuple[None, int, int, float]], [])
        return self.pinyin.check_popular(passwd, limit=limit, leet_cost=leet_cost)

    def check_popular(self, passwd: str, *, limit: int = 3, leet_cost: float = 1.5):
        if self.popular is None:
            return cast(list[tuple[None, int, int, float]], [])
        return self.popular.check_popular(passwd, limit=limit, leet_cost=leet_cost)

    def check_repetitions(self, passwd: str, *, limit: int = 3):
        n = len(passwd)
        v = [ord(c) for c in passwd]
        ret: list[tuple[int, int, int, float]] = []
        chErased = 0x110000

        def erase_part(i: int, length: int):
            nonlocal v, chErased
            for j in range(i, i + length):
                v[j] = chErased
                chErased += 1

        for m in range(n // 2, limit - 1, -1):
            for i in range(n - m * 2 + 1):
                found = False
                for j in range(i + m, n - m + 1):
                    if v[i : i + m] == v[j : j + m]:
                        erase_part(j, m)
                        cost = math.log(i + 1) + math.log(m)
                        ret.append((i, j, m, cost))
                        erase_part(j, m)
                        found = True
                if found:
                    erase_part(i, m)

        return ret

    def check_number(self, passwd: str, *, limit: int = 3):
        cur: list[int] = []
        ret: list[tuple[None, int, int, float]] = []
        v = [ord(c) for c in passwd]
        v.append(NOT_CHAR)
        for i in range(len(passwd) + 1):
            ch = v[i]
            if 0x30 <= ch <= 0x39:
                cur.append(ch)
            else:
                if len(cur) >= limit:
                    s = "".join(map(chr, cur))
                    cost = math.log(len(s) - len(s.lstrip("0")) + 1)
                    if int(s):
                        cost += math.log(int(s))
                    ret.append((None, i - len(s), len(s), cost))
                cur.clear()
        return ret

    def check_diff_seq(self, passwd: str, *, limit: int = 3):
        v = [ord(c) for c in passwd]
        v.append(NOT_CHAR)
        d, p = NOT_CHAR, 0
        ret: list[tuple[None, int, int, float]] = []
        for i in range(1, len(v)):
            cur = v[i] - v[i - 1]
            if cur == d:
                continue
            if i - p >= limit:
                ct = CharTypesTable.get_char_type(passwd[p])[1]
                cost = ct.ch_size + math.log(i - p - 1)
                ret.append((None, p, i - p, cost))
            d, p = cur, i - 1
        return ret

    def check(self, passwd: str, *, base: float | None = 2):
        ret: list[list[tuple[str, int | None, int, int, float]]] = []

        for i in range(len(passwd)):
            tp, _ = CharTypesTable.get_char_type(passwd[i])
            ret.append([(tp, None, i, 1, math.nan)])
        for data in add_type_tag(PatternID.ADJACENCY, self.check_adj(passwd)):
            ret[data[2]].append(data)
        for data in add_type_tag(PatternID.PINYIN, self.check_pinyin(passwd)):
            ret[data[2]].append(data)
        for data in add_type_tag(PatternID.POPULAR, self.check_popular(passwd)):
            ret[data[2]].append(data)
        for data in add_type_tag(PatternID.REPETITION, self.check_repetitions(passwd)):
            ret[data[2]].append(data)
        for data in add_type_tag(PatternID.NUMBER, self.check_number(passwd)):
            ret[data[2]].append(data)
        for data in add_type_tag(PatternID.DIFF_SEQ, self.check_diff_seq(passwd)):
            ret[data[2]].append(data)

        ecPattern = EntropyEncoder(PatternID.ALL, 0.0, 1.0, 0)
        mcData = EntropyEncoderTable()

        for tp in CharTypesTable.char_type[:-1]:
            assert tp[1].alphabet is not None
            uw = math.sqrt(tp[1].ch_cnt)
            mcData.addEncoder(tp[0], EntropyEncoder(tp[1].alphabet, 1.0, uw, 1))

        cost = math.inf
        rec: list[tuple[int, tuple[tuple[str, int | None, int, int, float], ...]]] = []
        rec.append((0, tuple()))

        while len(rec) > 0:
            s = rec.pop()
            if s[0] >= len(passwd):
                cur = computePathCost(s[1], passwd, ecPattern, mcData)
                if cur < cost:
                    cost = cur
            else:
                subs = ret[s[0]]
                for pi in subs[::-1]:
                    rec.append((s[0] + pi[3], (*s[1], pi)))

        if base is not None:
            cost /= math.log(base)

        return cost


if __name__ == "__main__":
    checker = Checker(
        adj_path="near.txt",
        pinyin_path="pinyin.txt",
        popular_path="popular.txt",
    )
    print(checker.check(input()))
