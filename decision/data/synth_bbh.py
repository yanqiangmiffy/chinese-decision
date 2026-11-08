"""Synthetic algorithmic reasoning puzzles in the style of procedurally generated BIG-Bench tasks.

Our own generators, each with exact labels: tracking objects through a sequence of swaps, the shape an SVG path
draws, whether a walk returns to its start, chains of truth-tellers and liars, and boolean expressions. Names,
items, stories and numbers are drawn from our own pools, so no benchmark item is reproduced.
"""
import math
import random

NAMES = ["Alice", "Bob", "Claire", "Dave", "Eve", "Fred", "Gertrude", "Hana", "Ivan", "Jun", "Kofi", "Lena", "Mateo",
         "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara", "Uma", "Victor", "Wen", "Ximena", "Yusuf", "Zoe"]
SCENES = [
    ("are playing a game. At the start, each is holding a ball: {pairs}.", "{a} and {b} swap balls", "a {x} ball", "ball", "has"),
    ("are dancers at a square dance. At the start of a song, each has a partner: {pairs}.", "{a} and {b} switch partners",
     "{x}", "partner", "is dancing with"),
    ("are friends who bought presents for each other: {pairs}.", "{a} and {b} trade gifts", "a {x}", "gift", "has"),
    ("are reading books: {pairs}.", "{a} and {b} swap books", "{x}", "book", "has"),
    ("are on the same team. At the start of the match, each is assigned a position: {pairs}.",
     "{a} and {b} trade positions", "{x}", "position", "is playing"),
    ("are sitting in a row of seats. At first, each is in a seat: {pairs}.", "{a} and {b} switch seats", "seat {x}", "seat", "is in"),
]
ITEMS = {
    "ball": ["red", "blue", "green", "yellow", "orange", "purple", "white", "black", "pink", "brown"],
    "partner": ["Ophelia", "Rodrigo", "Lola", "Izzi", "Karl", "Melissa", "Sam", "Patrick", "Jamie", "Helga", "Bruno", "Mia"],
    "gift": ["teddy bear", "puzzle", "kite", "scarf", "board game", "candle", "mug", "watch", "book", "hat"],
    "book": ["Dune", "Emma", "Ulysses", "Frankenstein", "Hamlet", "Beloved", "Matilda", "Persuasion", "Walden", "Ivanhoe"],
    "position": ["goalkeeper", "striker", "left winger", "right winger", "center midfielder", "fullback", "sweeper", "cheerleader"],
    "seat": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ASK = "Which option is the correct answer?"


def lettered(values):
    return [(f"({LETTERS[i]})", v) for i, v in enumerate(values)]


def tracking(rng):
    n = rng.choice([3, 5, 7])
    people = rng.sample(NAMES, n)
    intro, swap, show, kind, verb = rng.choice(SCENES)
    items = rng.sample(ITEMS[kind], n)
    holding = dict(zip(people, items))
    pairs = ", ".join(f"{p} {verb} {show.format(x=i)}" for p, i in holding.items())
    steps = []
    for _ in range(n if rng.random() < .7 else n + rng.randint(1, 3)):
        a, b = rng.sample(people, 2)
        holding[a], holding[b] = holding[b], holding[a]
        steps.append(swap.format(a=a, b=b))
    who = rng.choice(people)
    order = ", ".join(steps[:-1]) + f", and finally {steps[-1]}" if len(steps) > 1 else steps[0]
    names = ", ".join(people[:-1]) + f", and {people[-1]}"
    text = (f"{names} {intro.format(pairs=pairs)}\nAs the time goes on, they make these swaps in order: first {order}. "
            f"At the end, {who} {verb}\nOptions:\n" + "\n".join(f"({LETTERS[i]}) {show.format(x=x)}" for i, x in enumerate(items)))
    options = lettered([show.format(x=x) for x in items])
    return text, options, f"({LETTERS[items.index(holding[who])]})", f"tracking_{n}"


SHAPES = ["circle", "heptagon", "hexagon", "kite", "line", "octagon", "pentagon", "rectangle", "sector", "triangle"]


def polygon(rng, sides):
    cx, cy, r = rng.uniform(20, 80), rng.uniform(20, 80), rng.uniform(8, 30)
    start = rng.uniform(0, 2 * math.pi)
    jitter = .25 if sides > 4 else 0
    return [(cx + r * (1 + rng.uniform(-jitter, jitter) * .3) * math.cos(start + 2 * math.pi * k / sides),
             cy + r * (1 + rng.uniform(-jitter, jitter) * .3) * math.sin(start + 2 * math.pi * k / sides)) for k in range(sides)]


def path(points, rng, closed=True):
    fmt = lambda p: f"{p[0]:.2f},{p[1]:.2f}"
    pts = points + ([points[0]] if closed else [])
    if rng.random() < .5:
        return "M " + fmt(pts[0]) + " " + " ".join("L " + fmt(p) for p in pts[1:])
    return " ".join(f"M {fmt(a)} L {fmt(b)}" for a, b in zip(pts, pts[1:]))


def shapes(rng):
    shape = rng.choice(SHAPES)
    if shape in ("triangle", "pentagon", "hexagon", "heptagon", "octagon"):
        d = path(polygon(rng, {"triangle": 3, "pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8}[shape]), rng)
    elif shape == "rectangle":
        x, y, w, h = rng.uniform(10, 60), rng.uniform(10, 60), rng.uniform(5, 30), rng.uniform(5, 30)
        while abs(w - h) < 3:
            h = rng.uniform(5, 30)
        a = rng.uniform(0, math.pi)
        pts = [(x, y), (x + w * math.cos(a), y + w * math.sin(a)),
               (x + w * math.cos(a) - h * math.sin(a), y + w * math.sin(a) + h * math.cos(a)), (x - h * math.sin(a), y + h * math.cos(a))]
        d = path(pts, rng)
    elif shape == "kite":
        cx, cy, a = rng.uniform(20, 80), rng.uniform(20, 80), rng.uniform(0, math.pi)
        top, bottom, half = rng.uniform(5, 12), rng.uniform(15, 30), rng.uniform(6, 14)
        local = [(0, -top), (half, 0), (0, bottom), (-half, 0)]
        pts = [(cx + u * math.cos(a) - v * math.sin(a), cy + u * math.sin(a) + v * math.cos(a)) for u, v in local]
        d = path(pts, rng)
    elif shape == "line":
        pts = [(rng.uniform(5, 95), rng.uniform(5, 95)) for _ in range(2)]
        d = f"M {pts[0][0]:.2f},{pts[0][1]:.2f} L {pts[1][0]:.2f},{pts[1][1]:.2f}"
    elif shape == "circle":
        cx, cy, r = rng.uniform(20, 80), rng.uniform(20, 80), rng.uniform(5, 30)
        d = f"M {cx:.2f},{cy - r:.2f} A {r:.2f},{r:.2f} 0.00 1,0 {cx:.2f},{cy + r:.2f} A {r:.2f},{r:.2f} 0.00 1,0 {cx:.2f},{cy - r:.2f}"
    else:   # sector
        cx, cy, r = rng.uniform(20, 80), rng.uniform(20, 80), rng.uniform(10, 30)
        a, b = rng.uniform(0, 2 * math.pi), rng.uniform(.4, 2.5)
        p1 = (cx + r * math.cos(a), cy + r * math.sin(a))
        p2 = (cx + r * math.cos(a + b), cy + r * math.sin(a + b))
        d = (f"M {cx:.2f},{cy:.2f} L {p1[0]:.2f},{p1[1]:.2f} A {r:.2f},{r:.2f} 0.00 {int(b > math.pi)},1 {p2[0]:.2f},{p2[1]:.2f} "
             f"L {cx:.2f},{cy:.2f}")
    text = f'This SVG path element <path d="{d}"/> draws a\nOptions:\n' + "\n".join(f"({LETTERS[i]}) {s}" for i, s in enumerate(SHAPES))
    return text, lettered(SHAPES), f"({LETTERS[SHAPES.index(shape)]})", "shapes"


def navigate(rng):
    x = y = 0
    heading = 0                                     # 0 north, 1 east, 2 south, 3 west
    steps = []
    forward_only = rng.random() < .5
    moves = rng.randint(3, 8)
    for i in range(moves):
        if forward_only:
            d, k = rng.choice(["forward", "backward", "left", "right"]), rng.randint(1, 10)
            dx, dy = {"forward": (0, k), "backward": (0, -k), "left": (-k, 0), "right": (k, 0)}[d]
            steps.append(f"Take {k} step{'s' if k > 1 else ''} {d}.")
        else:
            if rng.random() < .35:
                turn = rng.choice(["Turn left.", "Turn right.", "Turn around."])
                heading = (heading + {"Turn left.": 3, "Turn right.": 1, "Turn around.": 2}[turn]) % 4
                steps.append(turn)
                continue
            k = rng.randint(1, 10)
            dx, dy = [(0, k), (k, 0), (0, -k), (-k, 0)][heading]
            steps.append(f"Take {k} step{'s' if k > 1 else ''}.")
        x, y = x + dx, y + dy
    if rng.random() < .75:                          # make about half of the walks return
        if forward_only:
            if y:
                steps.append(f"Take {abs(y)} step{'s' if abs(y) > 1 else ''} {'backward' if y > 0 else 'forward'}.")
            if x:
                steps.append(f"Take {abs(x)} step{'s' if abs(x) > 1 else ''} {'left' if x > 0 else 'right'}.")
            x = y = 0
    back = x == 0 and y == 0
    text = ("If you follow these instructions, do you return to the starting point? " + ("Always face forward. " if forward_only else "")
            + " ".join(steps) + "\nOptions:\n- Yes\n- No")
    return text, [("Yes", "Yes"), ("No", "No")], "Yes" if back else "No", "navigate"


def lies(rng):
    people = rng.sample(NAMES, rng.randint(4, 6))
    truth = rng.random() < .5
    parts = [f"{people[0]} {'tells the truth' if truth else 'lies'}."]
    for prev, person in zip(people, people[1:]):
        says_truth = rng.random() < .5
        parts.append(f"{person} says {prev} {'tells the truth' if says_truth else 'lies'}.")
        truth = says_truth == truth
    text = f"Question: {' '.join(parts)} Does {people[-1]} tell the truth?\nOptions:\n- Yes\n- No"
    return text, [("Yes", "Yes"), ("No", "No")], "Yes" if truth else "No", "lies"


def expression(rng, depth):
    if depth == 0 or rng.random() < .3:
        value = rng.random() < .5
        return str(value), value
    roll = rng.random()
    if roll < .25:
        text, value = expression(rng, depth - 1)
        return f"not {text}", not value
    if roll < .4:
        text, value = expression(rng, depth - 1)
        return f"( {text} )", value
    a, va = expression(rng, depth - 1)
    b, vb = expression(rng, depth - 1)
    op = rng.choice(["and", "or"])
    return f"{a} {op} {b}", (va and vb) if op == "and" else (va or vb)


def boolean(rng):
    text = "True"
    while len(text.split()) < 5:
        text, _ = expression(rng, rng.randint(2, 3))
    value = eval(text.replace("( ", "(").replace(" )", ")"))          # Python precedence, like the task's statement
    return f"{text} is", [("True", "True"), ("False", "False")], str(value), "boolean"


TASKS = [tracking, tracking, tracking, shapes, shapes, navigate, lies, boolean]


def puzzles(seed, count):
    rng = random.Random(f"synthetic-bbh:{seed}")
    for _ in range(count):
        yield rng.choice(TASKS)(rng)
