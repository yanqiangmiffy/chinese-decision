"""Relation sources: DuIE and CMeIE (entities + triples), FinRE and SanWen (given entity pairs).

DuIE and CMeIE feed both heads:
- span rows, one per subject: each type is "<subject>的<predicate>" and its spans are the objects,
  so the span head has to read the type text to know what to extract;
- pair decisions: for a (subject, object) pair, which schema predicate holds, or 无关系. Negatives
  are other ordered pairs of the sentence's entities whose types allow some predicate.
FinRE and SanWen give the pair and its relation only, so they feed the decision head.
"""
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .common import (NONE_RELATION, Raw, Source, Views, carve, choice_case, entity, locate, sample_candidates,
                     span_row, type_entry)

PAIR_INSTRUCTIONS = "根据文本，判断“{s}”（主体）与“{o}”（客体）之间是下列哪种关系；文本没有表达其中任何一种时选择“无关系”。"
NONE_DESCRIPTION = "文本没有表达以上任何一种关系"
DUIE_PREDICATES = {"object@inWork": "所属作品", "object@onDate": "获奖日期", "object@period": "获奖届次",
                   "object@inArea": "所在地区"}
DUIE_TYPES = {"Date": "日期", "Number": "数值", "Text": "文本"}

# FinRE: active and passive ("被X") readings; {h}/{t} are the head and tail entities of the line.
FINRE_SYMMETRIC = {"合作": "“{h}”与“{t}”合作", "竞争": "“{h}”与“{t}”存在竞争", "纠纷": "“{h}”与“{t}”存在纠纷",
                   "交易": "“{h}”与“{t}”进行交易", "签约": "“{h}”与“{t}”签约", "合并": "“{h}”与“{t}”合并",
                   "重组": "“{h}”与“{t}”重组", "合资": "“{h}”与“{t}”合资", "商讨": "“{h}”与“{t}”商讨合作或交易",
                   "订单": "“{h}”与“{t}”之间有订单往来", "自己": "“{h}”与“{t}”指同一家公司或机构（全称、简称或别称）"}
FINRE_DIRECTED = {"拥有": ("“{h}”拥有“{t}”", "“{h}”为“{t}”所拥有"), "持股": ("“{h}”持有“{t}”的股份", "“{h}”的股份被“{t}”持有"),
                  "分析": ("“{h}”分析评价“{t}”", "“{h}”被“{t}”分析评价"), "收购": ("“{h}”收购“{t}”", "“{h}”被“{t}”收购"),
                  "拟收购": ("“{h}”拟收购“{t}”", "“{h}”拟被“{t}”收购"), "减持": ("“{h}”减持“{t}”的股份", "“{h}”的股份被“{t}”减持"),
                  "增持": ("“{h}”增持“{t}”的股份", "“{h}”的股份被“{t}”增持"), "入股": ("“{h}”入股“{t}”", "“{h}”被“{t}”入股"),
                  "注资": ("“{h}”向“{t}”注资", "“{h}”获得“{t}”注资"), "帮助": ("“{h}”帮助“{t}”", "“{h}”得到“{t}”帮助"),
                  "成立": ("“{h}”成立了“{t}”", "“{h}”由“{t}”成立"), "买资": ("“{h}”购买“{t}”的资产", "“{h}”的资产被“{t}”购买"),
                  "发行": ("“{h}”发行“{t}”", "“{h}”由“{t}”发行"), "转让": ("“{h}”向“{t}”转让股权或资产", "“{h}”获得“{t}”转让的股权或资产"),
                  "欠款": ("“{h}”欠“{t}”款项", "“{h}”被“{t}”欠款"), "借壳": ("“{h}”借“{t}”的壳上市", "“{h}”被“{t}”借壳上市")}
FINRE_UNKNOWN = "文本没有表达两者之间的明确关系"
SANWEN = {"Create": ("创作", "一方创作、制作或建造了另一方"), "Use": ("使用", "一方使用另一方"),
          "Near": ("邻近", "两者在空间位置上相邻或靠近"), "Social": ("社会关系", "两人之间有亲属以外的社会关系，如朋友、师生、邻里"),
          "Located": ("位于", "一方位于另一方所在的地点或范围内"), "Ownership": ("拥有", "一方拥有或占有另一方"),
          "General-Special": ("一般与特殊", "一方是另一方的泛称或上位类别"), "Family": ("亲属", "两者之间有亲属关系"),
          "Part-Whole": ("部分与整体", "一方是另一方的组成部分"), "unknown": (NONE_RELATION, FINRE_UNKNOWN)}


def contiguous(indices):
    return indices and list(indices) == list(range(indices[0], indices[-1] + 1))


class TripleSource(Source):
    """Shared views of DuIE and CMeIE. Payload: entities [(start, end, labels)] and triples [(s, p, o)] by entity index."""

    task = "re"

    def __init__(self, name, dataset, schema, predicate_names, type_names, domain="general", license=""):
        self.name, self.dataset, self.domain, self.license = name, dataset, domain, license
        self.schema = schema                         # (subject type, predicate) -> {object types}
        self.predicate_names, self.type_names = predicate_names, type_names
        self.valid = defaultdict(set)                # (subject type, object type) -> {predicates}
        self.by_subject = defaultdict(set)           # subject type -> {predicates}
        for (subject_type, predicate), object_types in schema.items():
            self.by_subject[subject_type].add(predicate)
            for object_type in object_types:
                self.valid[(subject_type, object_type)].add(predicate)

    def pname(self, predicate):
        return self.predicate_names.get(predicate, predicate)

    def tname(self, label):
        return self.type_names.get(label, label)

    def predicates(self, subject_labels, object_labels=None):
        if object_labels is None:
            return set().union(*(self.by_subject[s] for s in subject_labels))
        return set().union(*(self.valid[(s, o)] for s in subject_labels for o in object_labels))

    def views(self, raw, rng, options):
        text, (entities, triples) = raw.text, raw.payload
        views = Views(structured={
            "id": f"{self.name}/{raw.split}/{raw.uid}", "source": self.name, "text": text,
            "entities": [{"start": s, "end": e, "text": text[s:e], "labels": sorted(labels)} for s, e, labels in entities],
            "relations": [{"subject": s, "predicate": p, "object": o} for s, p, o in triples]})
        stem = f"re_{self.name}"
        # Span rows: one per subject surface; its spans are the objects of each predicate.
        by_surface = defaultdict(list)
        for s, p, o in triples:
            by_surface[text[entities[s][0]:entities[s][1]]].append((s, p, o))
        for index, (surface, items) in enumerate(sorted(by_surface.items())):
            if len(surface) > options["max_subject_chars"]:
                continue
            labels = set().union(*(entities[s][2] for s, _, _ in items))
            predicates = sorted(self.predicates(labels) | {p for _, p, _ in items})
            types = []
            for predicate in predicates:
                objects = sorted(set().union(*(self.schema.get((label, predicate), set()) for label in labels)))
                types.append(type_entry(predicate, f"{surface}的{self.pname(predicate)}",
                                        "或".join(self.tname(t) for t in objects)))
            spans = {(entities[o][0], entities[o][1], p) for _, p, o in items}
            views.spans.append((stem, span_row(f"{stem}/{raw.split}/{raw.uid}#{index}", stem, text, types,
                                               [entity(*span) for span in sorted(spans)], self.meta(raw, "re_object"))))
        # Pair decisions: annotated pairs, then as many type-compatible unrelated pairs at most.
        related = defaultdict(set)
        for s, p, o in triples:
            related[(s, o)].add(p)
        pairs = [(pair, sorted(ps)) for pair, ps in sorted(related.items())]
        unrelated = [(a, b) for a in range(len(entities)) for b in range(len(entities))
                     if a != b and (a, b) not in related and entities[a][:2] != entities[b][:2]
                     and self.predicates(entities[a][2], entities[b][2])]
        rng.shuffle(unrelated)
        quota = max(1, round(len(pairs) * options["negative_pairs"]))
        pairs += [(pair, []) for pair in unrelated[:quota]]
        for index, ((a, b), gold) in enumerate(pairs):
            s_text, o_text = text[entities[a][0]:entities[a][1]], text[entities[b][0]:entities[b][1]]
            if s_text == o_text:
                continue
            pool = sorted(self.predicates(entities[a][2], entities[b][2]) | set(gold))
            keys = sample_candidates(gold, pool, rng, limit=15)
            criteria = {self.pname(p): f"“{o_text}”是“{s_text}”的{self.pname(p)}" for p in keys}
            criteria[NONE_RELATION] = NONE_DESCRIPTION
            views.decisions.append(choice_case(
                f"re_pair/{self.name}/{raw.split}/{raw.uid}#{index}", {"文本": text, "主体": s_text, "客体": o_text},
                PAIR_INSTRUCTIONS.format(s=s_text, o=o_text), criteria,
                [self.pname(p) for p in gold] or NONE_RELATION, self.meta(raw, "re_pair", negative=not gold)))
        return views


class DuIE(TripleSource):
    """DuIE (HF processed version, 54 type-predicate combinations): validation is our test, 3% of train is dev."""

    def __init__(self, root):
        self.root = Path(root)
        schema = defaultdict(set)
        for row in self.rows("train"):
            for rel in row["rels"]:
                schema[(rel["s"]["label"], rel["p"])].add(rel["o"]["label"])
        super().__init__("duie", "DuIE-HF", dict(schema), DUIE_PREDICATES, DUIE_TYPES, license="DuIE（百度，研究用途）")

    def rows(self, split):
        path, = sorted((self.root / "DuIE-HF/data").glob(f"{split}-*.parquet"))
        return pq.read_table(path).to_pylist()

    def records(self):
        for file_split, split in (("train", None), ("validation", "test")):
            for index, row in enumerate(self.rows(file_split)):
                text, spans, triples = row["text"], {}, []
                for rel in row["rels"]:
                    ends = []
                    for side in (rel["s"], rel["o"]):
                        start, end = side["indices"][0], side["indices"][-1] + 1
                        if not contiguous(side["indices"]) or text[start:end] != side["text"]:
                            break
                        spans.setdefault((start, end), set()).add(side["label"])
                        ends.append((start, end))
                    if len(ends) == 2:
                        triples.append((ends[0], rel["p"], ends[1]))
                if not triples:
                    continue
                order = sorted(spans)
                position = {span: i for i, span in enumerate(order)}
                entities = [(s, e, spans[(s, e)]) for s, e in order]
                triples = sorted({(position[s], p, position[o]) for s, p, o in triples})
                yield Raw(f"{file_split}-{index}", split or carve(text, dev=0.03), text, (entities, triples))


class CMeIE(TripleSource):
    """CMeIE: subjects and objects are strings, placed by `locate`; dev is our test (test has no labels)."""

    def __init__(self, root):
        self.root = Path(root)
        schema = defaultdict(set)
        for line in open(self.root / "CMeIE/53_schemas.jsonl", encoding="utf-8"):
            item = json.loads(line)
            schema[(item["subject_type"], item["predicate"])].add(item["object_type"])
        super().__init__("cmeie", "CMeIE", dict(schema), {}, {}, domain="medical", license="CBLUE（研究用途）")

    def records(self):
        for file_split, split in (("train", None), ("dev", "test")):
            for index, line in enumerate(open(self.root / f"CMeIE/CMeIE_{file_split}.jsonl", encoding="utf-8")):
                row = json.loads(line)
                text, mentions, wanted = row["text"], set(), []
                for spo in row["spo_list"]:
                    s, o = spo["subject"], spo["object"]["@value"]
                    s_type, o_type = spo["subject_type"], spo["object_type"]["@value"]
                    if not s or not o or s == o:
                        continue
                    mentions |= {(s_type, s), (o_type, o)}
                    wanted.append((s, s_type, spo["predicate"], o, o_type))
                spans, _ = locate(text, mentions)
                by_surface = defaultdict(list)
                for e in spans:
                    by_surface[(e["label"], text[e["start"]:e["end"]])].append((e["start"], e["end"]))
                merged = defaultdict(set)
                for e in spans:
                    merged[(e["start"], e["end"])].add(e["label"])
                order = sorted(merged)
                position = {span: i for i, span in enumerate(order)}
                triples = set()
                for s, s_type, predicate, o, o_type in wanted:
                    # Every occurrence of a subject relates to every occurrence of its object.
                    for s_span in by_surface.get((s_type, s), []):
                        for o_span in by_surface.get((o_type, o), []):
                            triples.add((position[s_span], predicate, position[o_span]))
                if not triples:
                    continue
                entities = [(s, e, merged[(s, e)]) for s, e in order]
                yield Raw(f"{file_split}-{index}", split or carve(text, dev=0.03), text, (entities, sorted(triples)))


class PairSource(Source):
    """FinRE / SanWen lines `head<TAB>tail<TAB>relation<TAB>text`, grouped by text. Payload: [(head, tail, labels)]."""

    def __init__(self, name, dataset, folder, domain, license=""):
        self.name, self.dataset, self.folder, self.domain, self.license = name, dataset, Path(folder), domain, license

    def records(self):
        for file_split, split in (("train", "train"), ("valid", "dev"), ("test", "test")):
            grouped = defaultdict(lambda: defaultdict(set))
            for line in open(self.folder / f"{file_split}.txt", encoding="utf-8"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4 or not parts[0] or not parts[1] or parts[0] == parts[1]:
                    continue
                head, tail, relation, text = parts[0], parts[1], parts[2], "\t".join(parts[3:])
                grouped[text][(head, tail)].add(relation)
            for index, (text, pairs) in enumerate(grouped.items()):
                payload = []
                for (head, tail), labels in sorted(pairs.items()):
                    known = labels - {"unknown"}
                    payload.append((head, tail, sorted(known or labels)))
                yield Raw(f"{file_split}-{index}", split, text, payload)

    def criteria(self, head, tail, gold, rng):
        raise NotImplementedError

    def views(self, raw, rng, options):
        views = Views()
        for index, (head, tail, labels) in enumerate(raw.payload):
            criteria, gold = self.criteria(head, tail, labels, rng)
            views.decisions.append(choice_case(
                f"re_pair/{self.name}/{raw.split}/{raw.uid}#{index}", {"文本": raw.text, "实体1": head, "实体2": tail},
                f"根据文本，判断“{head}”与“{tail}”之间是下列哪种关系。", criteria, gold,
                self.meta(raw, "re_pair", negative=gold == [NONE_RELATION])))
        return views


class FinRE(PairSource):
    """FinRE (44 labels): the gold labels, the converse reading, 无关系 and random others, at most 16."""

    def __init__(self, root):
        super().__init__("finre", "FinRE", Path(root) / "FinRE/data/FinRE", "finance", "Chinese_NRE（MIT）")
        self.labels = list(FINRE_SYMMETRIC) + list(FINRE_DIRECTED) + ["被" + label for label in FINRE_DIRECTED]

    def describe(self, label, head, tail):
        if label in FINRE_SYMMETRIC:
            return FINRE_SYMMETRIC[label].format(h=head, t=tail)
        if label in FINRE_DIRECTED:
            return FINRE_DIRECTED[label][0].format(h=head, t=tail)
        return FINRE_DIRECTED[label[1:]][1].format(h=head, t=tail)

    def criteria(self, head, tail, labels, rng):
        gold = [label for label in labels if label != "unknown"]
        converse = [label[1:] if label.startswith("被") else "被" + label for label in gold]
        keys = sample_candidates(gold, self.labels, rng, limit=15, first=converse)
        criteria = {label: self.describe(label, head, tail) for label in keys}
        criteria[NONE_RELATION] = FINRE_UNKNOWN
        return criteria, gold or [NONE_RELATION]


class SanWen(PairSource):
    """SanWen (literature, 9 relations + unknown): all ten candidates every time."""

    def __init__(self, root):
        super().__init__("sanwen", "SanWen-RE", Path(root) / "SanWen-RE/data/SanWen", "literature", "Chinese_NRE（MIT）")

    def criteria(self, head, tail, labels, rng):
        criteria = {name: description for name, description in SANWEN.values()}
        return criteria, [SANWEN[label][0] for label in labels if label in SANWEN] or [NONE_RELATION]


def sources(root):
    return [DuIE(root), CMeIE(root), FinRE(root), SanWen(root)]
