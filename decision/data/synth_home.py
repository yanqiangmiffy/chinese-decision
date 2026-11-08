"""Synthetic smart-home requests: how a household assistant should handle a request, and what changes.

Our own generator for the decisions the Decision Index's Home-Appliance task asks about (handling status, the
justified device targets, the immediate outcome, a listed state, and properties right after handling). The
homes, device names, rooms, policies and request wordings here are drawn from much larger random pools than
the benchmark's fixed households, and no benchmark row is produced: every home is random.

Handling rules (one consistent household model):
  * an ambiguous or unknown device or room -> ask a clarifying question (no justified targets)
  * a time in the future -> schedule; a "whenever"/"if" condition -> save an automation rule
  * a value outside the device's range, or a property the device does not have -> reject as unsupported
  * unlocking while the home is in away mode, or touching an unavailable device when the request is atomic
    -> block under household policy (with best-effort + allow_partial, available parts run instead)
  * nothing would change -> make no change because already satisfied
  * a question about state -> answer a state query
  * otherwise execute now; ordered steps apply in order (the last write wins), and only real changes count
"""
import copy
import hashlib
import json
import random

ROOMS = ["kitchen", "living room", "den", "main bedroom", "bedroom", "guest room", "nursery", "office", "study",
         "garage", "patio", "porch", "deck", "hallway", "dining room", "laundry room", "basement", "attic", "sunroom",
         "kids room", "library", "gym", "workshop", "mudroom", "family room", "lounge", "pantry", "loft",
         "back bedroom", "front hall", "playroom", "media room", "conservatory", "utility room", "balcony"]
LIGHT_NAMES = ["Floor Lamp", "Desk Lamp", "Bedside Lamp", "Ceiling Light", "Pendant Lights", "Strip Lights",
               "Wall Sconce", "Night Light", "Chandelier", "Track Lights", "Vanity Light", "Porch Light",
               "Accent Lamp", "Arc Lamp", "Globe Light", "Lantern", "Spotlight", "Mood Light", "Recessed Lights",
               "String Lights", "Table Lamp", "Uplight", "Cabinet Lights", "Shelf Lights"]
APPLIANCES = [("Coffee Maker", 900), ("Toaster", 850), ("Space Heater", 1500), ("Electric Blanket", 90),
              ("Fan", 45), ("Humidifier", 35), ("Dehumidifier", 300), ("Rice Cooker", 650), ("Slow Cooker", 250),
              ("Dishwasher", 1800), ("Washing Machine", 500), ("Dryer", 2400), ("Aquarium Pump", 12),
              ("Fountain", 25), ("Printer", 30), ("Monitor", 40), ("Game Console", 160), ("TV", 120),
              ("Soundbar", 30), ("Air Fryer", 1400), ("Towel Warmer", 100), ("Fridge Plug", 150),
              ("Iron", 1100), ("Hair Dryer", 1600), ("Electric Grill", 1700), ("Water Heater", 3000),
              ("Router", 10), ("Charger Dock", 20), ("Mini Fridge", 85), ("Heat Lamp", 250), ("Treadmill", 600),
              ("Pool Pump", 1100), ("Bread Maker", 550), ("Sewing Machine", 70), ("Fish Tank Light", 15)]
SPEAKERS = ["Smart Speaker", "Radio", "Ceiling Speaker", "Stereo", "Portable Speaker", "Sound System"]
THERMOSTATS = ["Thermostat", "Climate Panel", "Heat Pump", "Radiator Valve", "AC Unit"]
LOCKS = ["Front Door", "Back Door", "Garage Door", "Gate", "Side Door", "Patio Door", "Shed Door", "Cellar Door"]
UNKNOWN = ["dehumidifier", "robot vacuum", "projector", "window blinds", "ceiling fan", "sprinkler", "doorbell camera",
           "electric fireplace", "wine cooler", "bread maker", "air conditioner", "garage opener", "pool heater",
           "massage chair", "foot spa", "karaoke machine"]
ALIASES = ["back room", "upstairs room", "the kids' area", "the quiet room", "the far room", "the spare room",
           "downstairs room", "the side room"]
GROUP_NAMES = ["evening", "morning", "bedtime", "movie", "away", "dinner", "relax", "work", "party", "night"]
STATUS = {"execute": "execute now", "execute_partial": "execute available parts now", "schedule": "schedule for later",
          "rule": "save an automation rule", "clarify": "ask a clarifying question",
          "no_change": "make no change because already satisfied", "unsupported": "reject as unsupported",
          "blocked": "block under household policy", "query": "answer a state query"}
SCHEDULE = "a future schedule is added; device properties do not change yet"
RULE = "an automation rule is saved; device properties do not change immediately"
NOTHING = "no device property changes"
YES, NO = "yes, this device is in the justified target set", "no, this device is not in the justified target set"
NA = "not applicable or not listed"


def slug(text):
    return "_".join(text.lower().replace("'", "").split())


def lower(name):
    return name.lower()


def make_home(rng):
    rooms = rng.sample(ROOMS, rng.randint(3, 7))
    devices, used = [], set()

    def add(device):
        base = f"{slug(device['room'])}_{device.pop('slot')}"
        did, n = base, 2
        while did in used:
            did, n = f"{base}_{n}", n + 1
        used.add(did)
        devices.append({"id": did, **device})

    for room in rooms:
        for _ in range(rng.choice([1, 1, 2])):
            add({"slot": "light", "name": rng.choice(LIGHT_NAMES), "room": room, "kind": "light",
                 "available": True, "power": rng.random() < .5, "brightness": rng.choice(range(10, 101, 5)),
                 "watts": rng.randint(4, 40)})
        if rng.random() < .8:
            name, watts = rng.choice(APPLIANCES)
            add({"slot": "plug", "name": name, "room": room, "kind": "appliance", "available": True,
                 "power": rng.random() < .5, "watts": watts})
    for kind, names, count in (("thermostat", THERMOSTATS, rng.choice([1, 1, 2])), ("speaker", SPEAKERS, rng.choice([1, 2])),
                               ("lock", LOCKS, rng.choice([1, 1, 2]))):
        for _ in range(count):
            room = rng.choice(rooms)
            if kind == "thermostat":
                low = rng.choice([10, 12, 15, 16, 17, 18])
                high = rng.choice([24, 25, 26, 28, 30])
                add({"slot": "thermostat", "name": rng.choice(names), "room": room, "kind": kind, "available": True,
                     "power": rng.random() < .8, "temperature": rng.randint(low, high), "min_temperature": low,
                     "max_temperature": high, "watts": rng.randint(2, 6)})
            elif kind == "speaker":
                add({"slot": "speaker", "name": rng.choice(names), "room": room, "kind": kind, "available": True,
                     "power": rng.random() < .5, "volume": rng.choice(range(0, 101, 5)), "watts": rng.randint(5, 60)})
            else:
                add({"slot": "lock", "name": rng.choice(names), "room": room, "kind": kind, "available": True,
                     "locked": rng.random() < .6, "watts": 1})
    rng.shuffle(devices)
    groups = {}
    for name in rng.sample(GROUP_NAMES, rng.randint(1, 3)):
        members = [d["id"] for d in rng.sample(devices, min(len(devices), rng.randint(2, 5)))]
        groups[name] = members
    if rng.random() < .5:
        floor = rng.sample(rooms, max(1, len(rooms) // 2))
        groups["downstairs" if rng.random() < .5 else "upstairs"] = [d["id"] for d in devices if d["room"] in floor]
    aliases = {}
    if len(rooms) >= 3 and rng.random() < .8:
        aliases[rng.choice(ALIASES)] = rng.sample(rooms, 2)
    hour, minute = rng.randint(5, 22), rng.choice([0, 10, 15, 20, 30, 40, 45])
    home = {"now": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{hour:02d}:{minute:02d}:00{rng.choice(['-07:00', '-05:00', '+00:00', '+01:00', '+08:00'])}",
            "outside_temperature": rng.randint(-5, 35), "rooms": rooms, "room_aliases": aliases, "devices": devices,
            "groups": groups, "device_context": rng.choice(devices)["id"],
            "policy": {"multi_action": rng.choice(["atomic", "atomic", "ordered", "best_effort"]), "conflict": "last_wins",
                       "allow_partial": rng.random() < .3, "security_mode": rng.choice(["home", "home", "away"]),
                       "away_unlock": "blocked", "unavailable_action": "blocked"},
            "schedules": [], "rules": []}
    if rng.random() < .25:
        d = rng.choice(devices)
        prop = "locked" if d["kind"] == "lock" else "power"
        home["schedules"].append({"at": f"{rng.randint(0, 23):02d}:{rng.choice(['00', '30'])}", "ops": [[d["id"], prop, rng.random() < .5]]})
    if rng.random() < .25:
        d = rng.choice(devices)
        prop = "locked" if d["kind"] == "lock" else "power"
        home["rules"].append({"condition": ["outside_temperature", rng.choice(["below", "above"]), rng.randint(0, 30)],
                              "ops": [[d["id"], prop, rng.random() < .5]]})
    for d in rng.sample(devices, rng.choice([0, 0, 0, 1])):
        d["available"] = False
    return home


def dev(home, did):
    return next(d for d in home["devices"] if d["id"] == did)


def unique_named(home, kinds=None):
    """Devices whose lower-case name is unique in the home (so naming them is unambiguous)."""
    counts = {}
    for d in home["devices"]:
        counts[lower(d["name"])] = counts.get(lower(d["name"]), 0) + 1
    return [d for d in home["devices"] if counts[lower(d["name"])] == 1 and (kinds is None or d["kind"] in kinds)]


def on_off(value):
    return "on" if value else "off"


def say(rng, *options):
    return rng.choice(options)


# --- scenarios: each returns (spec, request) or None when the home does not support it -----------------------

def s_toggle(home, rng):
    pool = unique_named(home, ("light", "appliance", "speaker"))
    if not pool:
        return None
    d = rng.choice(pool)
    value = rng.random() < .5
    n = lower(d["name"])
    request = say(rng, f"Turn {on_off(value)} the {n}.", f"Switch the {n} {on_off(value)}.", f"Please put the {n} {on_off(value)}.",
                  f"Can you turn the {n} {on_off(value)}?", f"{'Start' if value else 'Stop'} the {n}.",
                  f"I want the {n} {'running' if value else 'off'}.", f"{n.capitalize()} {on_off(value)}, please.",
                  f"Could you {'power up' if value else 'power down'} the {n}?")
    return {"kind": "execute", "ops": [[d["id"], "power", value]]}, request


def s_room_lights(home, rng):
    room = rng.choice(home["rooms"])
    lights = [d for d in home["devices"] if d["room"] == room and d["kind"] == "light"]
    if not lights:
        return None
    value = rng.random() < .5
    except_one = len(lights) >= 2 and rng.random() < .6 and len({lower(d["name"]) for d in lights}) == len(lights)
    if except_one:
        keep = rng.choice(lights)
        ops = [[d["id"], "power", value] for d in lights if d["id"] != keep["id"]]
        request = say(rng, f"Turn {on_off(value)} all the lights in the {room} except the {lower(keep['name'])}.",
                      f"Switch {on_off(value)} every light in {room} but leave the {lower(keep['name'])} alone.")
    else:
        ops = [[d["id"], "power", value] for d in lights]
        request = say(rng, f"Turn {on_off(value)} the lights in the {room}.", f"All {room} lights {on_off(value)}.",
                      f"Switch every light in {room} {on_off(value)}.")
    return {"kind": "execute", "ops": ops}, request


def s_group(home, rng):
    name = rng.choice(list(home["groups"]))
    members = [m for m in home["groups"][name] if "power" in dev(home, m)]
    if not members:
        return None
    value = rng.random() < .5
    except_one = len(members) >= 3 and rng.random() < .5
    if except_one:
        candidates = [d for d in unique_named(home) if d["id"] in members]
        if not candidates:
            return None
        keep = rng.choice(candidates)
        ops = [[m, "power", value] for m in members if m != keep["id"]]
        request = say(rng, f"Turn {on_off(value)} the {name} group, but leave the {lower(keep['name'])} as it is.",
                      f"{'Start' if value else 'Shut down'} the {name} group except the {lower(keep['name'])}.")
    else:
        ops = [[m, "power", value] for m in members]
        request = say(rng, f"Turn the {name} group {on_off(value)}.", f"{'Activate' if value else 'Deactivate'} the {name} group.",
                      f"{'Start' if value else 'Stop'} everything in the {name} group, please.")
    return {"kind": "execute", "ops": ops}, request


def s_duplicate(home, rng):
    names = {}
    for d in home["devices"]:
        names.setdefault(lower(d["name"]), []).append(d)
    dup = [v for v in names.values() if len(v) >= 2]
    if not dup:
        # make one: rename a second light to match the first
        lights = [d for d in home["devices"] if d["kind"] == "light"]
        if len(lights) < 2:
            return None
        a, b = rng.sample(lights, 2)
        b["name"] = a["name"]
        dup = [[a, b]]
    group = rng.choice(dup)
    n = lower(group[0]["name"])
    prop = "locked" if group[0]["kind"] == "lock" else "power"
    verb = say(rng, f"Turn on the {n}.", f"Switch off the {n}.", f"Toggle the {n}.") if prop == "power" else f"Lock the {n}."
    return {"kind": "clarify", "candidates": [d["id"] for d in group]}, verb


def s_alias(home, rng):
    if not home["room_aliases"]:
        return None
    alias, rooms = next(iter(home["room_aliases"].items()))
    lights = [d for d in home["devices"] if d["room"] in rooms and d["kind"] == "light"]
    if len(lights) < 2:
        return None
    value = rng.random() < .5
    return ({"kind": "clarify", "candidates": [d["id"] for d in lights]},
            say(rng, f"Turn {on_off(value)} the light in the {alias}.", f"Can you switch the {alias} light {on_off(value)}?"))


def s_missing(home, rng):
    names = {lower(d["name"]) for d in home["devices"]}
    thing = rng.choice([u for u in UNKNOWN if u not in names])
    room = rng.choice(home["rooms"])
    return {"kind": "clarify", "candidates": []}, say(rng, f"Start the {thing} in the {room}.", f"Turn on the {thing}.",
                                                       f"Switch off the {thing} in {room}.")


def s_already(home, rng):
    pool = unique_named(home, ("light", "appliance", "speaker", "lock"))
    if not pool:
        return None
    d = rng.choice(pool)
    n = lower(d["name"])
    if d["kind"] == "lock":
        v = d["locked"]
        request = say(rng, f"Make sure the {n} is {'locked' if v else 'unlocked'}.", f"{'Lock' if v else 'Unlock'} the {n}.")
        return {"kind": "no_change", "ops": [[d["id"], "locked", v]]}, request
    v = d["power"]
    return ({"kind": "no_change", "ops": [[d["id"], "power", v]]},
            say(rng, f"Leave the {n} {on_off(v)}.", f"Keep the {n} {on_off(v)}.", f"Turn the {n} {on_off(v)}.",
                f"Make sure the {n} is {on_off(v)}."))


def s_thermostat(home, rng):
    pool = [d for d in unique_named(home, ("thermostat",))]
    if not pool:
        return None
    d = rng.choice(pool)
    n = lower(d["name"])
    roll = rng.random()
    if roll < .45:
        value = rng.choice([d["min_temperature"] - rng.randint(1, 6), d["max_temperature"] + rng.randint(1, 6)])
        return ({"kind": "unsupported", "ops": [[d["id"], "temperature", value]], "reason": "outside supported range"},
                say(rng, f"Set the {n} to {value}°C.", f"Put the {n} at {value} degrees."))
    if roll < .65:
        value = d["temperature"]
        return {"kind": "no_change", "ops": [[d["id"], "temperature", value]]}, f"Set the {n} to {value}°C."
    value = rng.choice([t for t in range(d["min_temperature"], d["max_temperature"] + 1) if t != d["temperature"]])
    delta = value - d["temperature"]
    request = say(rng, f"Set the {n} to {value}°C.", f"Change the {n} to {value} degrees.",
                  f"{'Raise' if delta > 0 else 'Lower'} the {n} by {abs(delta)} degrees.")
    return {"kind": "execute", "ops": [[d["id"], "temperature", value]]}, request


def s_relative(home, rng):
    pool = unique_named(home, ("light", "speaker"))
    if not pool:
        return None
    d = rng.choice(pool)
    n = lower(d["name"])
    prop = "brightness" if d["kind"] == "light" else "volume"
    step = rng.choice([5, 10, 15, 20, 25, 30])
    up = rng.random() < .5
    value = max(0, min(100, d[prop] + (step if up else -step)))
    word = ("brighter", "dimmer") if prop == "brightness" else ("louder", "quieter")
    request = say(rng, f"Make the {n} {step} points {word[0] if up else word[1]}.",
                  f"{'Increase' if up else 'Decrease'} the {n} {prop} by {step}.")
    if rng.random() < .3:
        value = rng.choice(range(0, 101, 5))
        request = f"Set the {n} {prop} to {value}%." if prop == "brightness" else f"Set the {n} volume to {value}."
    return {"kind": "execute", "ops": [[d["id"], prop, value]]}, request


def s_unsupported_property(home, rng):
    pool = unique_named(home, ("appliance", "lock"))
    if not pool:
        return None
    d = rng.choice(pool)
    n = lower(d["name"])
    prop, value, text = rng.choice([("brightness", 50, f"Dim the {n} to 50%."), ("volume", 30, f"Set the {n} volume to 30."),
                                    ("temperature", 22, f"Set the {n} to 22°C.")])
    return {"kind": "unsupported", "ops": [[d["id"], prop, value]], "reason": "device has no such property"}, text


def s_two_devices(home, rng):
    pool = unique_named(home, ("light", "appliance", "speaker"))
    if len(pool) < 2:
        return None
    a, b = rng.sample(pool, 2)
    value = rng.random() < .5
    best = rng.random() < .5
    if rng.random() < (.75 if best else .4):
        b["available"] = False
    if best:
        home["policy"]["multi_action"] = "best_effort"
        home["policy"]["allow_partial"] = True
        tail = say(rng, " Do whatever is available.", " Do what you can.", " Skip anything that's offline.")
        kind = "execute_partial"
    else:
        home["policy"]["multi_action"] = "atomic"
        tail = say(rng, "", " Both or neither.", "")
        kind = "execute"
    request = say(rng, f"Turn {on_off(value)} the {lower(a['name'])} and the {lower(b['name'])}.",
                  f"Switch both the {lower(a['name'])} and the {lower(b['name'])} {on_off(value)}.") + tail
    return {"kind": kind, "ops": [[a["id"], "power", value], [b["id"], "power", value]]}, request


def s_unavailable_single(home, rng):
    pool = unique_named(home, ("light", "appliance", "speaker"))
    if not pool:
        return None
    d = rng.choice(pool)
    d["available"] = False
    value = not d["power"]
    return {"kind": "execute", "ops": [[d["id"], "power", value]]}, f"Turn {on_off(value)} the {lower(d['name'])}."


def s_schedule(home, rng):
    pool = unique_named(home, ("light", "appliance", "speaker", "lock"))
    if not pool:
        return None
    d = rng.choice(pool)
    now = int(home["now"][11:13])
    hour = (now + rng.randint(1, 6)) % 24
    minute = rng.choice([0, 15, 30, 45])
    at = f"{hour:02d}:{minute:02d}"
    spoken = f"{(hour % 12) or 12}:{minute:02d} {'pm' if hour >= 12 else 'am'}"
    if d["kind"] == "lock":
        value = rng.random() < .5
        op, verb = [d["id"], "locked", value], ("lock" if value else "unlock")
    else:
        value = rng.random() < .5
        op, verb = [d["id"], "power", value], f"turn {on_off(value)}"
    request = say(rng, f"At {spoken}, {verb} the {lower(d['name'])}.", f"{verb.capitalize()} the {lower(d['name'])} at {spoken}.",
                  f"Schedule the {lower(d['name'])} to {verb} at {at}.")
    if rng.random() < .3:
        mins = rng.choice([10, 20, 30, 45, 60, 90])
        request = f"In {mins} minutes, {verb} the {lower(d['name'])}."
        total = int(home["now"][11:13]) * 60 + int(home["now"][14:16]) + mins
        at = f"{(total // 60) % 24:02d}:{total % 60:02d}"
    return {"kind": "schedule", "at": at, "ops": [op]}, request


def s_rule(home, rng):
    pool = unique_named(home, ("light", "appliance", "speaker", "thermostat"))
    if not pool:
        return None
    d = rng.choice(pool)
    direction = rng.choice(["below", "above"])
    limit = rng.randint(0, 30)
    if d["kind"] == "thermostat":
        value = rng.randint(d["min_temperature"], d["max_temperature"])
        op, act = [d["id"], "temperature", value], f"set the {lower(d['name'])} to {value}°C"
    else:
        value = rng.random() < .5
        op, act = [d["id"], "power", value], f"turn {on_off(value)} the {lower(d['name'])}"
    request = say(rng, f"Whenever it gets {direction} {limit}°C outside, {act}.", f"If the outside temperature goes {direction} {limit}°C, {act}.",
                  f"From now on, {act} whenever it's {direction} {limit} degrees outside.")
    return {"kind": "rule", "condition": ["outside_temperature", direction, limit], "ops": [op]}, request


def s_power_filter(home, rng):
    room = rng.choice(home["rooms"])
    apps = [d for d in home["devices"] if d["room"] == room and d["kind"] == "appliance"]
    if not apps:
        return None
    limit = rng.choice([50, 100, 200, 500, 1000])
    above = rng.random() < .6
    hits = [d for d in apps if (d["watts"] > limit if above else d["watts"] < limit)]
    if not hits:
        return None
    value = rng.random() < .3
    request = f"In the {room}, {'turn on' if value else 'shut down'} appliances rated {'over' if above else 'under'} {limit} watts."
    return {"kind": "execute", "ops": [[d["id"], "power", value] for d in hits]}, request


def s_pronoun(home, rng):
    d = dev(home, home["device_context"])
    if "power" not in d:
        return None
    value = rng.random() < .5
    return ({"kind": "execute", "ops": [[d["id"], "power", value]]},
            say(rng, f"Turn it {on_off(value)} too.", f"Now switch that {on_off(value)}.", f"And turn it {on_off(value)}.",
                f"Put that one {on_off(value)} as well."))


def s_sequence(home, rng):
    pool = unique_named(home, ("light", "appliance", "speaker"))
    if not pool:
        return None
    d = rng.choice(pool)
    n = lower(d["name"])
    home["policy"]["multi_action"] = "ordered"
    if d["kind"] == "light" and rng.random() < .6:
        level = rng.choice(range(10, 101, 10))
        return ({"kind": "execute", "ops": [[d["id"], "power", False], [d["id"], "brightness", level], [d["id"], "power", True]]},
                f"Switch off the {n}, set it to {level}%, then switch it back on.")
    first = rng.random() < .5
    return ({"kind": "execute", "ops": [[d["id"], "power", first], [d["id"], "power", not first]]},
            say(rng, f"Turn the {n} {on_off(first)}, then turn it {on_off(not first)} again.",
                f"Switch the {n} {on_off(first)} and then back {on_off(not first)}."))


def s_lock(home, rng):
    pool = unique_named(home, ("lock",))
    if not pool:
        return None
    d = rng.choice(pool)
    value = rng.random() < .5
    n = lower(d["name"])
    if not value and rng.random() < .6:
        home["policy"]["security_mode"] = "away"
    request = say(rng, f"{'Lock' if value else 'Unlock'} the {n}.", f"Please {'lock' if value else 'unlock'} the {n}.",
                  f"Can you {'secure' if value else 'open the lock on'} the {n}?")
    if not value and home["policy"]["security_mode"] == "away":
        return {"kind": "blocked", "ops": [[d["id"], "locked", False]], "reason": "away mode blocks unlock"}, request
    return {"kind": "execute", "ops": [[d["id"], "locked", value]]}, request


def s_query(home, rng):
    pool = unique_named(home)
    if not pool:
        return None
    d = rng.choice(pool)
    n = lower(d["name"])
    prop = {"thermostat": "temperature", "speaker": "volume", "lock": "locked", "light": "brightness"}.get(d["kind"], "power")
    if d["kind"] in ("light", "speaker") and rng.random() < .5:
        prop = "power"
    request = {"temperature": f"What's the {n} set to right now?", "volume": f"How loud is the {n}?",
               "locked": f"Is the {n} locked?", "brightness": f"How bright is the {n} at the moment?",
               "power": say(rng, f"Is the {n} on?", f"Is the {n} running right now?")}[prop]
    return {"kind": "query", "target": d["id"], "property": prop}, request


SCENARIOS = [s_toggle, s_toggle, s_room_lights, s_group, s_duplicate, s_alias, s_missing, s_already, s_thermostat,
             s_relative, s_unsupported_property, s_two_devices, s_two_devices, s_unavailable_single, s_schedule, s_rule,
             s_power_filter, s_pronoun, s_sequence, s_lock, s_lock, s_query]


def resolve(home, spec):
    """The handling status, justified targets, ordered real changes and the home right after handling."""
    final = copy.deepcopy(home)
    status, changed = spec["kind"], []
    if status in ("execute", "execute_partial"):
        skip = status == "execute_partial" and final["policy"]["allow_partial"]
        if any(not dev(final, did)["available"] for did, _, _ in spec["ops"]) and not skip:
            status = "blocked"
        else:
            for did, prop, value in spec["ops"]:
                d = dev(final, did)
                if not d["available"]:
                    continue
                if prop not in d:
                    status = "unsupported"
                    changed, final = [], copy.deepcopy(home)
                    break
                old, d[prop] = d.get(prop), value
                if old != value:
                    changed.append([did, prop, value])
            if status in ("execute", "execute_partial") and not changed:
                status = "no_change"
            elif status == "execute_partial" and all(dev(home, did)["available"] for did, _, _ in spec["ops"]):
                status = "execute"
        if status == "blocked":
            final = copy.deepcopy(home)
    elif status == "schedule":
        final["schedules"].append({"at": spec["at"], "ops": spec["ops"]})
    elif status == "rule":
        final["rules"].append({"condition": spec["condition"], "ops": spec["ops"]})
    targets = sorted({x[0] for x in spec.get("ops", [])} | set(spec.get("candidates", [])) |
                     ({spec["target"]} if "target" in spec else set()))
    return {"status": status, "targets": targets, "changed": changed, "final": final}


def outcome_text(result):
    if result["status"] == "schedule":
        return SCHEDULE
    if result["status"] == "rule":
        return RULE
    if not result["changed"]:
        return NOTHING
    return "; ".join(f"{d}.{p} becomes {json.dumps(v)}" for d, p, v in result["changed"])


def questions_for(home, spec, result, rng):
    """(role, instructions, candidate texts, gold text) for one request."""
    out = [("resolution", "How should the assistant handle the request?", list(STATUS.values()), STATUS[result["status"]])]
    justified = set() if result["status"] == "clarify" else set(result["targets"])
    for d in home["devices"]:
        out.append((f"target:{d['id']}", f"Is {d['id']} in the justified device target set for this request?", [YES, NO],
                    YES if d["id"] in justified else NO))
    outcome = outcome_text(result)
    distract = [NOTHING, SCHEDULE, RULE]
    if result["targets"]:
        t = rng.choice(result["targets"])
        prop = "locked" if dev(home, t)["kind"] == "lock" else "power"
        distract += [f"{t}.{prop} becomes true", f"{t}.{prop} becomes false"]
    if result["changed"] and len(result["changed"]) > 1:
        distract.append("; ".join(f"{d}.{p} becomes {json.dumps(v)}" for d, p, v in result["changed"][:-1]))
    out.append(("outcome", "What is the concrete immediate outcome after applying the correct handling?",
                list(dict.fromkeys([outcome] + distract)), outcome))
    probe = rng.choice(home["devices"])
    prop = rng.choice([p for p in ("power", "locked", "temperature", "brightness", "volume") if p in probe])
    value = probe[prop]
    other = json.dumps(not value) if isinstance(value, bool) else value + rng.choice([-5, -1, 1, 5])
    vals = [f"{probe['id']}.{prop} is {json.dumps(value)}", f"{probe['id']}.{prop} is {other}", "the property is not listed"]
    out.append(("state_read", f"Before handling the request, what is the listed {prop} state of {probe['name']} in {probe['room']}?", vals, vals[0]))
    ops = spec.get("ops", [])
    if ops:
        for did, prop, value in rng.sample(ops, min(2, len(ops))):
            after = dev(result["final"], did).get(prop)
            before = dev(home, did).get(prop)
            gold = NA if after is None else json.dumps(after)
            vals = list(dict.fromkeys([gold, json.dumps(value), NA] + ([json.dumps(before)] if before is not None else [])))
            out.append((f"post:{did}", f"Immediately after correct handling, what is {prop} for {did}?", vals, gold))
    return out


def requests(seed, count):
    """`count` random requests as (state, [(role, instructions, candidates, gold)], meta)."""
    rng = random.Random(f"synthetic-home:{seed}")
    made = 0
    while made < count:
        home = make_home(rng)
        scenario = rng.choice(SCENARIOS)
        drafted = scenario(home, rng)
        if not drafted:
            continue
        spec, request = drafted
        result = resolve(home, spec)
        state = dict(home, request=request)
        made += 1
        yield state, questions_for(home, spec, result, rng), {"scenario": scenario.__name__[2:], "status": result["status"]}


def key(namespace, text):
    return "k_" + hashlib.sha1(f"{namespace}:{text}".encode()).hexdigest()[:8]
