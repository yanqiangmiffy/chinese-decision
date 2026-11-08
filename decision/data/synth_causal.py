"""Synthetic causal-inference questions over small binary causal graphs (association, intervention, counterfactual).

Our own generator for the question kinds of causal-ladder benchmarks: marginal and conditional probabilities,
average treatment effects, effects of treatment on the treated, natural direct/indirect effects, backdoor
adjustment, explaining away, collider bias and deterministic counterfactuals. Variables and stories are our own
(see VARIABLES); every number is computed exactly from a random binary causal model, so labels are exact.
"""
import itertools
import random

PREAMBLE = ("Imagine a self-contained, hypothetical world with only the following conditions, and without any "
            "unmentioned factors or causal relationships: ")

# name, event when 1, event when 0, group when 1, group when 0
VARIABLES = [
    ("exercise", "regular exercise", "no regular exercise", "people who exercise regularly", "people who do not exercise"),
    ("sleep", "enough sleep", "too little sleep", "people who sleep enough", "people who sleep too little"),
    ("diet", "a healthy diet", "an unhealthy diet", "people with a healthy diet", "people with an unhealthy diet"),
    ("stress", "high stress", "low stress", "people under high stress", "people under low stress"),
    ("blood pressure", "high blood pressure", "normal blood pressure", "people with high blood pressure", "people with normal blood pressure"),
    ("heart disease", "heart disease", "no heart disease", "people with heart disease", "people without heart disease"),
    ("vaccination", "being vaccinated", "not being vaccinated", "vaccinated people", "unvaccinated people"),
    ("infection", "infection", "no infection", "infected people", "uninfected people"),
    ("fever", "fever", "no fever", "people with fever", "people without fever"),
    ("medication", "taking the medication", "not taking the medication", "patients taking the medication", "patients not taking the medication"),
    ("recovery", "recovery", "no recovery", "recovered patients", "patients who did not recover"),
    ("tutoring", "receiving tutoring", "no tutoring", "students with tutoring", "students without tutoring"),
    ("study time", "long study hours", "short study hours", "students who study long hours", "students who study briefly"),
    ("exam score", "a high exam score", "a low exam score", "students with high exam scores", "students with low exam scores"),
    ("scholarship", "winning a scholarship", "no scholarship", "scholarship winners", "students without a scholarship"),
    ("parental income", "high parental income", "low parental income", "children of high-income parents", "children of low-income parents"),
    ("rain", "rain", "no rain", "rainy days", "dry days"),
    ("traffic", "heavy traffic", "light traffic", "days with heavy traffic", "days with light traffic"),
    ("lateness", "arriving late", "arriving on time", "late arrivals", "on-time arrivals"),
    ("alarm", "the alarm ringing", "the alarm staying silent", "mornings when the alarm rings", "mornings when the alarm stays silent"),
    ("advertising", "a large ad budget", "a small ad budget", "firms with large ad budgets", "firms with small ad budgets"),
    ("sales", "high sales", "low sales", "firms with high sales", "firms with low sales"),
    ("price cut", "a price cut", "no price cut", "stores that cut prices", "stores that keep prices"),
    ("profit", "high profit", "low profit", "profitable stores", "unprofitable stores"),
    ("training program", "joining the training program", "skipping the training program", "workers in the program", "workers not in the program"),
    ("skill", "high skill", "low skill", "skilled workers", "unskilled workers"),
    ("promotion", "getting promoted", "not getting promoted", "promoted workers", "workers who were not promoted"),
    ("wage", "a high wage", "a low wage", "high earners", "low earners"),
    ("fertilizer", "using fertilizer", "no fertilizer", "fields with fertilizer", "fields without fertilizer"),
    ("soil quality", "rich soil", "poor soil", "fields with rich soil", "fields with poor soil"),
    ("harvest", "a large harvest", "a small harvest", "fields with large harvests", "fields with small harvests"),
    ("irrigation", "irrigation", "no irrigation", "irrigated fields", "unirrigated fields"),
    ("pollution", "high pollution", "low pollution", "polluted cities", "clean cities"),
    ("asthma", "asthma", "no asthma", "people with asthma", "people without asthma"),
    ("smoking", "smoking", "not smoking", "smokers", "nonsmokers"),
    ("tar deposit", "high tar deposit", "low tar deposit", "people with high tar deposit", "people with low tar deposit"),
    ("lung cancer", "lung cancer", "no lung cancer", "people with lung cancer", "people without lung cancer"),
    ("gene", "carrying the gene", "not carrying the gene", "gene carriers", "non-carriers"),
    ("coffee", "drinking coffee", "not drinking coffee", "coffee drinkers", "people who skip coffee"),
    ("alertness", "high alertness", "low alertness", "alert people", "drowsy people"),
    ("accident", "having an accident", "no accident", "drivers with accidents", "drivers without accidents"),
    ("seat belt", "wearing a seat belt", "no seat belt", "belted drivers", "unbelted drivers"),
    ("injury", "serious injury", "minor or no injury", "seriously injured people", "people without serious injury"),
    ("talent", "talent", "no talent", "talented people", "untalented people"),
    ("effort", "great effort", "little effort", "hard workers", "people putting in little effort"),
    ("fame", "being famous", "not being famous", "famous people", "people who are not famous"),
    ("luck", "good luck", "bad luck", "lucky people", "unlucky people"),
    ("success", "success", "failure", "successful people", "unsuccessful people"),
    ("music lessons", "taking music lessons", "no music lessons", "children with music lessons", "children without music lessons"),
    ("attention span", "a long attention span", "a short attention span", "children with long attention spans", "children with short attention spans"),
    ("reading", "reading daily", "not reading daily", "daily readers", "people who rarely read"),
    ("vocabulary", "a large vocabulary", "a small vocabulary", "people with a large vocabulary", "people with a small vocabulary"),
    ("sunlight", "plenty of sunlight", "little sunlight", "sunny regions", "dark regions"),
    ("vitamin D", "high vitamin D", "low vitamin D", "people with high vitamin D", "people with low vitamin D"),
    ("bone density", "high bone density", "low bone density", "people with dense bones", "people with fragile bones"),
    ("policy", "the new policy", "the old policy", "cities with the new policy", "cities with the old policy"),
    ("crime", "high crime", "low crime", "high-crime cities", "low-crime cities"),
    ("street lights", "bright street lights", "dim street lights", "streets with bright lights", "streets with dim lights"),
    ("water quality", "clean water", "dirty water", "villages with clean water", "villages with dirty water"),
    ("cholera", "cholera", "no cholera", "people with cholera", "people without cholera"),
    ("software update", "installing the update", "skipping the update", "updated devices", "outdated devices"),
    ("crash", "a crash", "no crash", "devices that crash", "devices that run stably"),
    ("battery life", "long battery life", "short battery life", "devices with long battery life", "devices with short battery life"),
    ("insurance", "having insurance", "no insurance", "insured people", "uninsured people"),
    ("hospital visit", "visiting a hospital", "not visiting a hospital", "people who visit hospitals", "people who stay away from hospitals"),
]

GRAPHS = {   # node -> parents; X treatment, Y outcome; U is unobserved
    "chain": {"X": [], "M": ["X"], "Y": ["M"]},
    "fork": {"X": [], "V2": [], "Y": ["X", "V2"]},
    "confounding": {"V1": [], "X": ["V1"], "Y": ["V1", "X"]},
    "mediation": {"X": [], "M": ["X"], "Y": ["X", "M"]},
    "collision": {"X": [], "Y": [], "V3": ["X", "Y"]},
    "diamond": {"X": [], "M": ["X"], "V3": ["X"], "Y": ["M", "V3"]},
    "diamondcut": {"V1": [], "X": ["V1"], "V3": ["V1"], "Y": ["X", "V3"]},
    "arrowhead": {"X": [], "V2": [], "M": ["X", "V2"], "Y": ["X", "M", "V2"]},
    "frontdoor": {"U": [], "X": ["U"], "M": ["X"], "Y": ["M", "U"]},
    "IV": {"U": [], "V2": [], "X": ["V2", "U"], "Y": ["X", "U"]},
}
QUERIES = {
    "chain": ["marginal", "correlation", "ate", "ett", "nie", "backadj", "det"],
    "fork": ["marginal", "correlation", "ate", "ett", "det"],
    "confounding": ["marginal", "correlation", "ate", "ett", "backadj", "det"],
    "mediation": ["marginal", "correlation", "ate", "ett", "nde", "nie", "backadj", "det"],
    "collision": ["marginal", "correlation", "exp_away", "collider_bias", "backadj"],
    "diamond": ["marginal", "correlation", "ate", "ett", "nie", "backadj", "det"],
    "diamondcut": ["marginal", "correlation", "ate", "ett", "backadj", "det"],
    "arrowhead": ["marginal", "correlation", "ate", "ett", "nde", "nie", "det"],
    "frontdoor": ["marginal", "correlation", "ate", "ett", "nie"],
    "IV": ["marginal", "correlation", "ate"],
}


class Model:
    def __init__(self, graph, rng):
        self.parents = GRAPHS[graph]
        self.order = list(self.parents)                       # parents always listed first
        self.cpt = {}
        for node, ps in self.parents.items():
            for values in itertools.product((0, 1), repeat=len(ps)):
                self.cpt[node, values] = round(rng.uniform(.02, .98), 2)

    def joint(self, do=None):
        do = do or {}
        for values in itertools.product((0, 1), repeat=len(self.order)):
            world = dict(zip(self.order, values))
            p = 1.0
            for node in self.order:
                if node in do:
                    if world[node] != do[node]:
                        p = 0
                        break
                    continue
                q = self.cpt[node, tuple(world[x] for x in self.parents[node])]
                p *= q if world[node] else 1 - q
            if p:
                yield world, p

    def prob(self, event, given=None, do=None):
        given = given or {}
        num = den = 0.0
        for world, p in self.joint(do):
            if all(world[k] == v for k, v in given.items()):
                den += p
                if all(world[k] == v for k, v in event.items()):
                    num += p
        return num / den if den else 0.0

    def descendants(self, node):
        out, frontier = set(), [node]
        while frontier:
            n = frontier.pop()
            for child, ps in self.parents.items():
                if n in ps and child not in out:
                    out.add(child)
                    frontier.append(child)
        return out

    def ett(self):
        """E[Y_{X=0} | X=1] - E[Y | X=1]: non-descendants of X keep their distribution given X=1."""
        keep = [n for n in self.order if n not in self.descendants("X") and n != "X"]
        total = 0.0
        for values in itertools.product((0, 1), repeat=len(keep)):
            n = dict(zip(keep, values))
            w = self.prob(n, {"X": 1})
            if w:
                total += w * (self.prob({"Y": 1}, do={"X": 0, **n}) - self.prob({"Y": 1}, do={"X": 1, **n}))
        return total

    def natural(self):
        """(NDE, NIE) through mediator M, for graphs where M is the only mediator."""
        keep = [n for n in self.order if n not in self.descendants("X") and n != "X"]
        nde = nie = 0.0
        for values in itertools.product((0, 1), repeat=len(keep)):
            n = dict(zip(keep, values))
            pn = self.prob(n)
            for m in (0, 1):
                pm0 = self.prob({"M": m}, do={"X": 0, **n})
                pm1 = self.prob({"M": m}, do={"X": 1, **n})
                y1 = self.prob({"Y": 1}, do={"X": 1, "M": m, **n})
                y0 = self.prob({"Y": 1}, do={"X": 0, "M": m, **n})
                nde += pn * pm0 * (y1 - y0)
                nie += pn * y0 * (pm1 - pm0)
        return nde, nie


def pct(p):
    return f"{round(100 * p)}%"


class Story:
    def __init__(self, graph, rng):
        names = [n for n in GRAPHS[graph] if n != "U"]
        picked = rng.sample(VARIABLES, len(names))
        self.var = dict(zip(names, picked))

    def event(self, node, value):
        return self.var[node][1 if value else 2]

    def group(self, node, value):
        return self.var[node][3 if value else 4]

    def name(self, node):
        return self.var[node][0]

    def background(self, parents):
        effects = {}
        for child, ps in parents.items():
            for p in ps:
                if p != "U" and child != "U":
                    effects.setdefault(p, []).append(child)
        sentences = []
        for cause, kids in effects.items():
            names = [self.name(k) for k in kids]
            joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            sentences.append(f"{self.name(cause)[0].upper() + self.name(cause)[1:]} has a direct effect on {joined}.")
        if "U" in parents:
            hit = [c for c, ps in parents.items() if "U" in ps]
            sentences.append(f"Unobserved factors have a direct effect on {self.name(hit[0])} and {self.name(hit[1])}.")
        return PREAMBLE + " ".join(sentences)

    def conditional(self, model, target, given):
        """'For <groups>, the probability of <target> is p%.' for every setting of `given`."""
        lines = []
        for values in itertools.product((0, 1), repeat=len(given)):
            g = dict(zip(given, values))
            p = model.prob({target: 1}, g)
            groups = " and ".join(self.group(k, v) for k, v in g.items())
            lines.append(f"For {groups}, the probability of {self.event(target, 1)} is {pct(p)}.")
        return lines


def yes_no(rng, positive_is_true):
    """Phrase direction and gold: returns (word for direction, gold 'yes'/'no')."""
    ask_up = rng.random() < .5
    return ask_up, ("yes" if ask_up == positive_is_true else "no")


def make_question(graph, query, rng):
    model, story = Model(graph, rng), Story(graph, rng)
    bg = story.background(model.parents)
    x1, y1 = story.event("X", 1), story.event("Y", 1)
    if query == "marginal":
        p = model.prob({"Y": 1})
        if abs(p - .5) < .02:
            return None
        info = [f"The overall probability of {x1} is {pct(model.prob({'X': 1}))}."] + story.conditional(model, "Y", ["X"])
        more = rng.random() < .5
        q = f"Is {y1} {'more' if more else 'less'} likely than {story.event('Y', 0)} overall?"
        gold = "yes" if (p > .5) == more else "no"
    elif query == "correlation":
        a, b = model.prob({"Y": 1}, {"X": 1}), model.prob({"Y": 1}, {"X": 0})
        if abs(a - b) < .01:
            return None
        info = [f"The overall probability of {x1} is {pct(model.prob({'X': 1}))}.",
                f"The probability of {story.event('X', 0)} and {y1} is {pct(model.prob({'X': 0, 'Y': 1}))}.",
                f"The probability of {x1} and {y1} is {pct(model.prob({'X': 1, 'Y': 1}))}."]
        up, gold = yes_no(rng, a > b)
        q = f"Is the chance of {y1} {'larger' if up else 'smaller'} when observing {x1}?"
    elif query in ("ate", "ett"):
        ate = model.prob({"Y": 1}, do={"X": 1}) - model.prob({"Y": 1}, do={"X": 0})
        effect = ate if query == "ate" else -model.ett()          # treated: Y_1 - Y_0
        if abs(effect) < .01:
            return None
        if graph in ("confounding", "diamondcut"):
            z = "V1" if graph == "confounding" else "V3"
            info = story.conditional(model, "Y", [z, "X"]) + [f"The overall probability of {story.event(z, 1)} is {pct(model.prob({z: 1}))}."]
            if query == "ett":
                info += story.conditional(model, z, ["X"])
        elif graph == "IV":
            info = story.conditional(model, "Y", ["V2"]) + story.conditional(model, "X", ["V2"])
            iv = (model.prob({"Y": 1}, {"V2": 1}) - model.prob({"Y": 1}, {"V2": 0})) / (
                model.prob({"X": 1}, {"V2": 1}) - model.prob({"X": 1}, {"V2": 0}) or 1e-9)
            effect = iv                                            # what the given numbers identify
            if abs(effect) < .01:
                return None
        elif graph == "frontdoor":
            info = story.conditional(model, "M", ["X"]) + story.conditional(model, "Y", ["X", "M"]) + [
                f"The overall probability of {x1} is {pct(model.prob({'X': 1}))}."]
        else:
            info = story.conditional(model, "Y", ["X"])
        up, gold = yes_no(rng, effect > 0)
        if query == "ate":
            q = f"Will {x1} {'increase' if up else 'decrease'} the chance of {y1}?"
        else:
            # would Y be more likely for the treated had they not been treated?  more likely <=> effect < 0
            up, gold = yes_no(rng, effect < 0)
            q = (f"For {story.group('X', 1)}, would it be {'more' if up else 'less'} likely to see {y1} "
                 f"if they had {story.event('X', 0)} instead?")
    elif query in ("nde", "nie"):
        nde, nie = model.natural()
        if graph in ("chain", "frontdoor", "diamond"):
            nie = model.prob({"Y": 1}, do={"X": 1}) - model.prob({"Y": 1}, do={"X": 0})
        effect = nde if query == "nde" else nie
        if abs(effect) < .01:
            return None
        given = ["X", "M"] + (["V2"] if graph == "arrowhead" else [])
        info = story.conditional(model, "Y", given if graph not in ("chain", "frontdoor", "diamond") else
                                 (["M"] if graph == "chain" else ["X", "M"] if graph == "frontdoor" else ["M", "V3"]))
        if graph == "diamond":
            info += story.conditional(model, "M", ["X"]) + story.conditional(model, "V3", ["X"])
        else:
            info += story.conditional(model, "M", ["X"] + (["V2"] if graph == "arrowhead" else []))
        if graph == "arrowhead":
            info.append(f"The overall probability of {story.event('V2', 1)} is {pct(model.prob({'V2': 1}))}.")
        if graph == "frontdoor":
            info.append(f"The overall probability of {x1} is {pct(model.prob({'X': 1}))}.")
        up, gold = yes_no(rng, effect > 0)
        word = "positively" if up else "negatively"
        if query == "nie":
            via = story.name("M") if graph != "diamond" else f"{story.name('M')} and {story.name('V3')}"
            q = f"Does {story.name('X')} {word} affect {story.name('Y')} through {via}?"
        else:
            q = f"If we disregard the mediation effect through {story.name('M')}, would {story.name('X')} {word} affect {story.name('Y')}?"
    elif query == "backadj":
        z = {"chain": "M", "confounding": "V1", "mediation": "M", "collision": "V3", "diamond": "M", "diamondcut": "V1"}[graph]
        valid = graph in ("confounding", "diamondcut")
        case_by_case = f"We look at how {story.name('X')} correlates with {story.name('Y')} case by case according to {story.name(z)}."
        general = f"We look directly at how {story.name('X')} correlates with {story.name('Y')} in general."
        first_is_case = rng.random() < .5
        m1, m2 = (case_by_case, general) if first_is_case else (general, case_by_case)
        info = [f"Method 1: {m1} Method 2: {m2}"]
        q = f"To understand how {story.name('X')} affects {story.name('Y')}, is it more correct to use the Method 1 than Method 2?"
        gold = "yes" if valid == first_is_case else "no"
    elif query == "exp_away":
        a, b = model.prob({"Y": 1}, {"X": 1, "V3": 1}), model.prob({"Y": 1}, {"X": 0, "V3": 1})
        if abs(a - b) < .01:
            return None
        info = [f"The overall probability of {x1} is {pct(model.prob({'X': 1}))}."] + story.conditional(model, "Y", ["X", "V3"])
        up, gold = yes_no(rng, a > b)
        q = f"If we look at {story.group('V3', 1)}, does the chance of {y1} {'increase' if up else 'decrease'} when {x1}?"
    elif query == "collider_bias":
        r = round(rng.uniform(-.6, .6), 2) or .1
        info = [f"For {story.group('V3', 1)}, the correlation between {x1} and {y1} is {r}."]
        affects = rng.random() < .5
        q = (f"If we look at {story.group('V3', 1)}, does it mean that {x1} {'affects' if affects else 'does not affect'} {y1}?")
        gold = "no" if affects else "yes"
    else:   # deterministic counterfactual
        return det_question(graph, story, rng)
    return f"{bg}\n\n{' '.join(info)}\n\n{q}", gold, query


def det_question(graph, story, rng):
    parents = {k: [p for p in v if p != "U"] for k, v in GRAPHS[graph].items() if k != "U"}
    funcs, facts = {}, []
    for node, ps in parents.items():
        if not ps:
            continue
        neg = {p: rng.random() < .3 for p in ps}
        op = rng.choice(["or", "and"]) if len(ps) > 1 else "id"
        funcs[node] = (ps, neg, op)
        parts = [story.event(p, 0 if neg[p] else 1) for p in ps]
        facts.append(f"{f' {op} '.join(parts)} causes {story.event(node, 1)}." if op != "id" else
                     f"{parts[0].capitalize()} causes {story.event(node, 1)}.")
    roots = [n for n, ps in parents.items() if not ps and n != "X"]
    world = {r: rng.random() < .5 for r in roots}
    facts += [f"We observed {story.event(r, int(v))}." for r, v in world.items()]

    def run(x, force=True):
        w = dict(world, X=x)
        for node in parents:
            if node in funcs and not (force and node == "X"):
                ps, neg, op = funcs[node]
                vals = [w[p] != neg[p] for p in ps]
                w[node] = any(vals) if op == "or" else all(vals) if op == "and" else vals[0]
        return w

    x = run(rng.random() < .5, force=False)["X"]                        # the factual X

    target = rng.random() < .5
    outcome = run(not x)["Y"]
    q = (f"Would {story.event('Y', int(target))} occur if {story.event('X', int(not x))} instead of {story.event('X', int(x))}?")
    facts = [f[0].upper() + f[1:] for f in facts]
    return f"{story.background(GRAPHS[graph])}\n\nWe know that {' '.join(facts)}\n\n{q}", "yes" if outcome == target else "no", "det"


def questions(seed, count):
    rng = random.Random(f"synthetic-causal:{seed}")
    made = 0
    while made < count:
        graph = rng.choice(list(GRAPHS))
        query = rng.choice(QUERIES[graph])
        out = make_question(graph, query, rng)
        if out:
            made += 1
            yield (*out, graph)
