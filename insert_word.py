# The script is used to insert new pinyin into the popular.txt file.

import sys


def insert(path: str):
    data = set()

    with open(path, "r", encoding="utf-8") as file:
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

    with open(path, "w", encoding="utf-8") as file:
        print(*sorted(data), sep="\n", file=file)

    print("done.")

    sys.exit(0)
