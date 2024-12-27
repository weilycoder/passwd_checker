from __future__ import annotations

import math, itertools

from collections import deque, defaultdict

from typing import Generator, cast


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


class Check_Popular:
    def __init__(self, table_path: str):
        self.table = Trie()
        self.len_cnt: defaultdict[int, int] = defaultdict(int)
        with open(table_path, "r", encoding="utf-8") as file:
            for line in file:
                word = line.strip()
                self.table.insert(word)
                self.len_cnt[len(word)] += 1
        self.table.build()

    def check_popular(self, passwd: str, *, limit: int = 5, leet_cost: float = 1.5):
        ret = []
        leet_pwd = decode_leet(passwd.lower())
        for pos, length in self.table.query(leet_pwd, limit=limit):
            dist = hamming_dist(passwd, leet_pwd, slice(pos, pos + length))
            cost = math.log(self.len_cnt[length])
            cost += log_comb(len(passwd), length)
            cost += dist * leet_cost
            ret.append((pos, length, cost))
        return ret


class Checker:
    def __init__(
        self,
        *,
        near_path: str | None = None,
        pinyin_path: str | None = None,
    ):
        self.near: set[frozenset[str]] = set()
        if near_path is not None:
            with open(near_path, "r", encoding="utf-8") as f:
                for line in f:
                    x, y = line.split()
                    self.near.add(frozenset((x, y)))

        self.pinyin = Check_Popular(pinyin_path) if pinyin_path is not None else None

    def check_pinyin(self, passwd: str, *, limit: int = 5, leet_cost: float = 1.5):
        if self.pinyin is None:
            return []
        return self.pinyin.check_popular(passwd, limit=limit, leet_cost=leet_cost)

    def check_near(self, passwd: str, *, limit: int = 3):
        cur = ""
        words: list[tuple[str, int]] = []
        for i in range(1, len(passwd)):
            x, y = passwd[i - 1], passwd[i]
            if frozenset((x.lower(), y.lower())) in self.near:
                if not cur:
                    cur = x
                cur += y
            else:
                if len(cur) >= limit:
                    words.append((cur, i - len(cur)))
                cur = ""
        if len(cur) >= limit:
            words.append((cur, len(passwd) - len(cur)))
        return words



if __name__ == "__main__":
    checker = Checker(
        near_path="near.txt",
        pinyin_path="pinyin.txt",
    )
    # tests
    print(checker.check_pinyin("woaini"))
    print(checker.check_pinyin("woaini123"))
    print(checker.check_pinyin("woaishanghaidaxue"))
    print(checker.check_near("qazwsx"))
    print(checker.check_near("1q2w3e4r5t6y7u8i9o0p"))
