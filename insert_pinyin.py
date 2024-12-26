# The script is used to insert new pinyin into the pinyin.txt file.

import sys

data = set()

with open("pinyin.txt", "r", encoding="utf-8") as file:
    for line in file:
        data.add(line.strip())

try:
    while True:
        word = input("insert? ")
        if word in data:
            print("Warning: already exists.")
        else:
            data.add(word)
except EOFError:
    pass
except KeyboardInterrupt:
    print("Interrupted.")
    sys.exit(1)

with open("pinyin.txt", "w", encoding="utf-8") as file:
    print(*sorted(data), sep="\n", file=file)

print("done.")

sys.exit(0)
