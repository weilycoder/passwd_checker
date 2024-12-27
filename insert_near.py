import sys

edge: set[frozenset[str]] = set()


def add(x: str, y: str, *, warn: bool = True) -> None:
    x, y = x.lower(), y.lower()
    if warn and (x, y) in edge:
        print(f"Warning: {(x, y)} is already exist.")
    edge.add(frozenset((x, y)))


with open("near.txt", "r", encoding="utf-8") as file:
    for line in file:
        x, y = line.split()
        add(x, y, warn=False)

try:
    while True:
        try:
            table = list(input("insert? "))
        except ValueError:
            continue
        else:
            for i in range(1, len(table)):
                x, y = table[i - 1], table[i]
                if x > y:
                    x, y = y, x
                if (x, y) in edge:
                    print(f"Warning: {(x, y)} is already exist.")
                else:
                    add(x, y)
except EOFError:
    pass
except KeyboardInterrupt:
    print("Interrupted.")
    sys.exit(1)

edge_list = sorted(edge)

with open("near.txt", "w", encoding="utf-8") as file:
    for x, y in edge_list:
        print(x, y, file=file)

print("done.")

sys.exit(0)
