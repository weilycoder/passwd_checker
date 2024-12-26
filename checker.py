from __future__ import annotations

import collections


class Trie:
    def __init__(self, fail: Trie | None = None):
        self.fail = self if fail is None else fail
        self.__locked = False
        self.children: dict[str, Trie] = {}
        self.end_of_word = ""

    @property
    def locked(self):
        return self.__locked

    def lock(self):
        assert not self.locked
        self.__locked = True

    def build(self):
        self.lock()
        queue = collections.deque(self.children.values())
        while queue:
            u = queue.popleft()
            for k in u.children:
                u[k].fail = u.fail[k]
                queue.append(u[k])

    def query(self, word: str):
        assert self.locked
        res = ""
        it = self
        for c in word:
            it = it[c]
            if len(it.end_of_word) > len(res):
                res = it.end_of_word
        return res

    def insert(self, word: str):
        assert not self.locked
        it = self
        for c in word:
            if c not in it.children:
                it.children[c] = Trie(self)
            it = it.children[c]
        it.end_of_word = word

    def __getitem__(self, key: str) -> Trie:
        if key not in self.children:
            if self == self.fail:
                return self
            self.children[key] = self.fail[key]
        return self.children[key]


class Checker:
    def __init__(self, pinyin_path: str):
        self.trie = Trie()
        with open(pinyin_path, "r", encoding="utf-8") as f:
            for line in f:
                self.trie.insert(line.strip())
        self.trie.build()

    def check_pinyin(self, passwd: str, *, limit: int = 5):
        word = self.trie.query(passwd)
        if len(word) >= limit:
            return word


if __name__ == "__main__":
    checker = Checker("pinyin.txt")
    # tests
    print(checker.check_pinyin("woaini"))
    print(checker.check_pinyin("woaini123"))
    print(checker.check_pinyin("woaishanghaidaxue"))
