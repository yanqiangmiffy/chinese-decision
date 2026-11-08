"""Synthetic symbolic chord recognition: a short pop-style arrangement and the chord sounding at a target beat.

Our own generator (no song from any dataset): a key, a chord progression drawn from common pop patterns
(with sevenths, suspensions and borrowed chords now and then), an accompaniment (block chords, arpeggios,
broken patterns, a bass line on the root or fifth) and a melody of chord tones with passing and neighbour
notes. The answer is the chord's pitch-class set in the 13-quality vocabulary used for symbolic chord
recognition (enharmonic equivalents grouped), or NoChord where the arrangement rests.
"""
import random

ROOTS = "C C# D D# E F F# G G# A A# B".split()
TEMPLATES = {"M": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
             "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10), "half-dim7": (0, 3, 6, 10), "dim7": (0, 3, 6, 9),
             "mMaj7": (0, 3, 7, 11), "aug7": (0, 4, 8, 10)}
ALIASES = {}
for _r in range(12):
    for _name, _ints in TEMPLATES.items():
        ALIASES.setdefault(tuple(sorted((_r + x) % 12 for x in _ints)), []).append(f"{ROOTS[_r]}{_name}")
VOCAB = {pc: " / ".join(names) for pc, names in ALIASES.items()}
VOCAB["NoChord"] = "NoChord"
VOCAB["Other"] = "Other"

MAJOR = [0, 2, 4, 5, 7, 9, 11]
# scale degree -> (semitones above tonic, quality) in a major key; minor keys use the relative major's chords
DEGREES = {"I": (0, "M"), "ii": (2, "m"), "iii": (4, "m"), "IV": (5, "M"), "V": (7, "M"), "vi": (9, "m"), "vii": (11, "dim"),
           "bVII": (10, "M"), "iv": (5, "m"), "bVI": (8, "M"), "II": (2, "M"), "III": (4, "M")}
PROGRESSIONS = [["I", "V", "vi", "IV"], ["vi", "IV", "I", "V"], ["I", "vi", "IV", "V"], ["IV", "V", "iii", "vi"],
                ["ii", "V", "I", "vi"], ["I", "IV", "V", "IV"], ["vi", "V", "IV", "V"], ["I", "iii", "IV", "V"],
                ["IV", "I", "V", "vi"], ["I", "bVII", "IV", "I"], ["vi", "ii", "V", "I"], ["I", "V", "IV", "iv"],
                ["IV", "V", "I", "vi", "ii", "V", "I", "I"], ["I", "II", "IV", "I"], ["vi", "bVI", "bVII", "I"],
                ["I", "III", "vi", "IV"], ["ii", "iii", "IV", "V"]]
SEVENTH = {"M": ["maj7", "7"], "m": ["m7"], "dim": ["half-dim7", "dim7"]}


def chord_pcs(root, quality):
    return tuple(sorted((root + x) % 12 for x in TEMPLATES[quality]))


def label(pcs):
    return VOCAB.get(tuple(sorted(set(pcs))), "Other") if pcs else "NoChord"


def progression(rng, tonic, bars):
    pattern = rng.choice(PROGRESSIONS)
    chords = []
    for i in range(bars * 2):
        degree = pattern[i % len(pattern)]
        step, quality = DEGREES[degree]
        if rng.random() < .18 and quality in SEVENTH:
            quality = rng.choice(SEVENTH[quality])
        elif degree == "V" and rng.random() < .25:
            quality = "7"
        elif quality == "M" and rng.random() < .05:
            quality = rng.choice(["sus2", "sus4"])
        elif rng.random() < .01:
            quality = rng.choice(["aug", "mMaj7", "aug7", "dim7"])
        chords.append(((tonic + step) % 12, quality))
    return chords


def arrange(rng):
    """Notes [(pitch, start, end)] and the annotated chord spans [(start, end, pcs)] in beats."""
    tonic = rng.randrange(12)
    span = rng.choice([2, 4, 4, 4, 8])                 # beats per chord
    total = 32 + rng.choice([0, 8, 16])
    chords = progression(rng, tonic, total // span)[: total // span]
    notes, spans = [], []
    style = rng.choice(["block", "arpeggio", "broken", "pad", "alberti"])
    register = rng.choice([48, 52, 55])
    rest_from = rng.choice([None, None, None, rng.randrange(len(chords))])
    for i, (root, quality) in enumerate(chords):
        start = i * span
        if rest_from is not None and i == rest_from:
            spans.append((start, start + span, ()))
            continue
        pcs = chord_pcs(root, quality)
        spans.append((start, start + span, pcs))
        tones = [register + ((root + x) % 12) for x in TEMPLATES[quality]]
        tones = sorted(t if t >= register else t + 12 for t in tones)
        if rng.random() < .5:                           # inversion
            tones = tones[1:] + [tones[0] + 12]
        bass = 36 + root if root < 8 else 24 + root
        for b in range(0, span, 2 if rng.random() < .5 else 1):
            pitch = bass if (b // 2) % 2 == 0 or quality in ("sus2", "sus4") else bass + 7
            notes.append((pitch, start + b, start + b + rng.choice([.5, 1, 1, 2])))
        if style in ("block", "pad"):
            step = span if style == "pad" else rng.choice([1, 2])
            for b in range(0, span, step):
                for t in tones:
                    notes.append((t, start + b, start + b + step * rng.choice([.5, .9, 1])))
        else:
            order = tones + tones[-2:0:-1] if style == "arpeggio" else ([tones[0], tones[-1], tones[1 % len(tones)], tones[-1]] if style == "alberti" else tones[::-1] + tones)
            step = rng.choice([.5, .5, 1])
            k = 0
            b = 0.0
            while b < span:
                notes.append((order[k % len(order)], start + b, start + b + step * rng.choice([1, 1, 1.5])))
                k += 1
                b += step
        # melody: chord tones on strong beats, scale tones between
        scale = [(tonic + x) % 12 for x in MAJOR]
        b = 0.0
        while b < span:
            length = rng.choice([.5, .5, 1, 1, 1.5, 2])
            strong = abs(b - round(b)) < 1e-9 and int(b) % 2 == 0
            pc = rng.choice(pcs) if strong or rng.random() < .55 else rng.choice(scale)
            notes.append((72 + pc - (12 if pc > 7 and rng.random() < .5 else 0), start + b, start + b + length))
            b += length
            if rng.random() < .15:
                b += .5                                  # a rest
    return notes, spans, tonic


def example(rng):
    notes, spans, tonic = arrange(rng)
    end = max(e for _, e, _ in spans)
    grid = [t for t in range(2, int(end) - 1, 4)]
    target = rng.choice(grid)
    pcs = next((p for s, e, p in spans if s <= target < e), ())
    # the annotated chord is the chord tones; a rest bar has no chord
    context = [{"pitch": int(p), "pitch_class": int(p) % 12, "start_beat": float(s), "end_beat": float(e)}
               for p, s, e in sorted(notes, key=lambda n: (n[1], n[0])) if s < target + 4 and e > target - 4]
    key = ROOTS[tonic] if rng.random() < .7 else ROOTS[(tonic + 9) % 12] + "m"
    public = {"ticks_per_beat": rng.choice([220, 480, 960]),
              "key_signatures": [{"beat": 0.0, "key": key}] if rng.random() < .5 else [],
              "time_signatures": [{"beat": 0.0, "numerator": 4, "denominator": 4}] if rng.random() < .6 else []}
    if not pcs:   # a resting arrangement: nothing sounds at the target beat
        context = [n for n in context if not (n["start_beat"] <= target < n["end_beat"])]
    return {"target_time_beats": target, "context_notes": context, "public_context": public}, label(pcs)


def examples(seed, count):
    rng = random.Random(f"synthetic-chords:{seed}")
    for _ in range(count):
        yield example(rng)
