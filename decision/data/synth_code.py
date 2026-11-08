"""Synthetic Python output prediction: short random functions, an input, and the value `f(input)` returns.

Our own generator: a function body is a few statements drawn from string, list, number and dict idioms
(loops, comprehensions, slicing, counting, sorting, conditionals), run on a random input. Distractors are the
outputs of the same function with one statement changed (an off-by-one, a flipped comparison, a different
method), so they look like the slips a reader makes. Labels come from executing the code.
"""
import random
import string

WORDS = ["apple", "data", "level", "python", "zebra", "moon", "a", "abc", "hello world", "x-y-z", "racecar", "Mississippi",
         "banana", "Code", "TEST", "map", "  pad  ", "k9", "ab12cd", "noon", "flow", "cat dog", "eel"]


def rand_str(rng):
    if rng.random() < .5:
        return rng.choice(WORDS)
    return "".join(rng.choice(string.ascii_lowercase[:6] + " -1") for _ in range(rng.randint(1, 9)))


def rand_list(rng):
    return [rng.randint(-5, 12) for _ in range(rng.randint(0, 7))]


# statement templates: (input kind, [variants]); variant 0 is the real one, the others make distractors.
STR_OPS = [
    ["s = s.upper()", "s = s.lower()", "s = s.title()"],
    ["s = s[::-1]", "s = s[1:]", "s = s[:-1]"],
    ["s = s.replace('a', 'o')", "s = s.replace('o', 'a')", "s = s.replace('a', '')"],
    ["s = s.strip()", "s = s.lstrip()", "s = s.rstrip()"],
    ["s = s[1::2]", "s = s[::2]", "s = s[2:]"],
    ["s = ''.join(c for c in s if c not in 'aeiou')", "s = ''.join(c for c in s if c in 'aeiou')", "s = ''.join(c for c in s if c != 'a')"],
    ["s = s + s[0] if s else s", "s = s[0] + s if s else s", "s = s + s[-1] if s else s"],
    ["s = s.swapcase()", "s = s.upper()", "s = s.capitalize()"],
    ["s = '-'.join(s.split())", "s = ''.join(s.split())", "s = '_'.join(s.split())"],
    ["s = s.center(len(s) + 2, '*')", "s = s.ljust(len(s) + 2, '*')", "s = s.rjust(len(s) + 2, '*')"],
]
STR_ENDS = [
    ["return s", "return s[::-1]", "return s.upper()"],
    ["return len(s)", "return len(s) - 1", "return len(s) + 1"],
    ["return s.count('a')", "return s.count('e')", "return s.count('a') + 1"],
    ["return s.find('a')", "return s.rfind('a')", "return s.index('a') if 'a' in s else 0"],
    ["return [s[i:i+2] for i in range(0, len(s), 2)]", "return [s[i:i+2] for i in range(0, len(s), 3)]", "return [s[i:i+3] for i in range(0, len(s), 3)]"],
    ["return s.isalpha()", "return s.isdigit()", "return not s.isalpha()"],
    ["return {c: s.count(c) for c in sorted(set(s))[:3]}", "return {c: s.count(c) for c in sorted(set(s))[:2]}", "return {c: 1 for c in sorted(set(s))[:3]}"],
]
LIST_OPS = [
    ["nums = [n * 2 for n in nums]", "nums = [n + 2 for n in nums]", "nums = [n * n for n in nums]"],
    ["nums = [n for n in nums if n % 2 == 0]", "nums = [n for n in nums if n % 2 == 1]", "nums = [n for n in nums if n > 0]"],
    ["nums = sorted(nums)", "nums = sorted(nums, reverse=True)", "nums = nums[::-1]"],
    ["nums = nums[1:]", "nums = nums[:-1]", "nums = nums[2:]"],
    ["nums.append(len(nums))", "nums.insert(0, len(nums))", "nums.append(0)"],
    ["nums = list(dict.fromkeys(nums))", "nums = sorted(set(nums))", "nums = nums"],
    ["nums = [n - 1 if n > 3 else n + 1 for n in nums]", "nums = [n + 1 if n > 3 else n - 1 for n in nums]", "nums = [n - 1 if n >= 3 else n + 1 for n in nums]"],
    ["nums = [abs(n) for n in nums]", "nums = [-n for n in nums]", "nums = [max(n, 0) for n in nums]"],
    ["if len(nums) > 2:\n        nums[0], nums[-1] = nums[-1], nums[0]", "if len(nums) > 2:\n        nums[0], nums[1] = nums[1], nums[0]", "if len(nums) > 3:\n        nums[0], nums[-1] = nums[-1], nums[0]"],
    ["for i in range(len(nums) - 1):\n        if nums[i] > nums[i + 1]:\n            nums[i] = nums[i + 1]",
     "for i in range(len(nums) - 1):\n        if nums[i] < nums[i + 1]:\n            nums[i] = nums[i + 1]",
     "for i in range(1, len(nums)):\n        if nums[i] > nums[i - 1]:\n            nums[i] = nums[i - 1]"],
]
LIST_ENDS = [
    ["return nums", "return nums[::-1]", "return sorted(nums)"],
    ["return sum(nums)", "return sum(nums) - 1", "return max(nums, default=0)"],
    ["return len(nums)", "return len(set(nums))", "return len(nums) + 1"],
    ["return [(nums.count(n), n) for n in sorted(set(nums))]", "return [(n, nums.count(n)) for n in sorted(set(nums))]", "return [(1, n) for n in sorted(set(nums))]"],
    ["return nums[len(nums) // 2] if nums else None", "return nums[0] if nums else None", "return nums[-1] if nums else None"],
    ["return {n: i for i, n in enumerate(nums)}", "return {i: n for i, n in enumerate(nums)}", "return {n: 0 for n in nums}"],
    ["return [sum(nums[:i + 1]) for i in range(len(nums))]", "return [sum(nums[:i]) for i in range(len(nums))]", "return [sum(nums[i:]) for i in range(len(nums))]"],
]


def build(kind, picks, ends, end, arg):
    lines = [f"def f({arg}):"]
    for template, variant in picks:
        lines.append("    " + template[variant])          # continuation lines carry their own indent
    lines.append("    " + ends[end[0]][end[1]])
    return "\n".join(lines)


def run(code, value):
    scope = {}
    exec(code, {"__builtins__": {"len": len, "sorted": sorted, "set": set, "dict": dict, "list": list, "range": range,
                                 "sum": sum, "max": max, "abs": abs, "enumerate": enumerate}}, scope)
    return scope["f"](value)


def example(rng):
    kind = rng.choice(["str", "list"])
    ops, ends, arg = (STR_OPS, STR_ENDS, "s") if kind == "str" else (LIST_OPS, LIST_ENDS, "nums")
    chosen = rng.sample(range(len(ops)), rng.randint(1, 3))
    end = rng.randrange(len(ends))
    value = rand_str(rng) if kind == "str" else rand_list(rng)
    real = build(kind, [(ops[i], 0) for i in chosen], ends, (end, 0), arg)
    try:
        answer = repr(run(real, value if kind == "str" else list(value)))
    except Exception:
        return None
    outs = {answer}
    variants = [(j, v) for j in range(len(chosen) + 1) for v in (1, 2)]
    rng.shuffle(variants)
    for j, v in variants:
        picks = [(ops[i], v if k == j else 0) for k, i in enumerate(chosen)]
        code = build(kind, picks, ends, (end, v if j == len(chosen) else 0), arg)
        try:
            outs.add(repr(run(code, value if kind == "str" else list(value))))
        except Exception:
            continue
        if len(outs) == 4:
            break
    if len(outs) < 3:
        return None
    options = sorted(outs, key=lambda _: rng.random())
    return {"code": real, "input": repr(value), "task": "Python output selection"}, options, answer


def examples(seed, count):
    rng = random.Random(f"synthetic-code:{seed}")
    made = 0
    while made < count:
        out = example(rng)
        if out:
            made += 1
            yield out
