"""Event sources: DuEE and FewFC (sentence events with triggers and arguments), ChFinAnn (document records).

DuEE and FewFC feed both heads:
- span rows, one per (sentence, event type present): the types are 触发词 and the roles of that
  event type, with the event type written in each description; arguments of every event of that type
  are marked. A quarter of the sentences also get a row for an absent event type with no spans, so
  the span head learns to stay silent when the event is not there;
- decisions: whether the sentence describes each of its event types (plus one absent type), and
  for DuEE the 9-way category of single-category sentences, for FewFC the type of single-type ones.
ChFinAnn feeds the decision head: whether an announcement reports each event type, and which
mention fills a record field (or 公告未提及), asked only where every record of that type agrees.
"""
import json
from collections import defaultdict
from pathlib import Path

from .common import (Raw, Source, Views, carve, choice_case, entity, noul_case, read_parquet, sample_candidates,
                     span_row, type_entry)

TRIGGER = "trigger"
FEWFC_TYPES = {"质押": "债务人或第三方将动产或权利（如股份）移交债权人作为债权担保",
               "股份股权转让": "公司股东将股份或股权有偿转让给他人",
               "投资": "组织或个人向某一领域或企业投放资金或实物以获得收益",
               "起诉": "依法向法院提起诉讼，请求对特定案件进行审判",
               "减持": "上市公司股东或高管卖出所持本公司股票",
               "收购": "一家公司通过产权交易取得另一家公司一定程度的控制权",
               "判决": "司法机关对审理结束的案件作出裁决",
               "签署合同": "合作双方在合同等书面文件上签字或盖章确立合作关系",
               "担保": "为促使债务人履行债务，向债权人订立担保合同",
               "中标": "招标人向经评选的投标人发出中标通知并订立合同"}
# FewFC/Event_Definition.pdf: role name and description per event type.
FEWFC_ROLES = {
    "质押": {"sub-org": ("质押公司", "发起质押的公司"), "sub-per": ("质押人", "发起质押的个人"),
           "obj-org": ("质权公司", "接受质押的公司"), "obj-per": ("质权人", "接受质押的个人"),
           "collateral": ("质押物", "质押的标的，如公司股票、大额存单"), "date": ("质押日期", "明确写出的质押日期"),
           "money": ("质押金额", "质押涉及的金额"), "number": ("质押数量", "质押的股份等数量"), "proportion": ("质押比例", "质押所占比例")},
    "股份股权转让": {"sub-org": ("转让公司", "发起股份股权转让的公司"), "sub-per": ("转让人", "发起股份股权转让的个人"),
               "obj-org": ("受让公司", "接受股份股权转让的公司"), "obj-per": ("受让人", "接受股份股权转让的个人"),
               "collateral": ("转让标的", "转让的股份、股权等"), "date": ("转让日期", "明确写出的转让日期"),
               "money": ("转让金额", "转让交易金额"), "number": ("转让数量", "转让的股份数量"), "proportion": ("转让比例", "转让所占比例"),
               "target-company": ("标的公司", "被转让股份股权所属的公司")},
    "起诉": {"sub-per": ("原告个人", "提起诉讼的个人"), "sub-org": ("原告公司", "提起诉讼的公司"),
           "obj-per": ("被告个人", "被起诉的个人"), "obj-org": ("被告公司", "被起诉的公司"), "date": ("起诉日期", "起诉的日期")},
    "投资": {"sub": ("投资方", "发起投资的组织或单位"), "obj": ("被投资方", "被投资的组织或单位"),
           "money": ("投资金额", "投资的金额"), "date": ("投资日期", "投资的日期")},
    "减持": {"sub": ("减持方", "减持股份的一方，通常为个人"), "obj": ("被减持方", "被减持股份所属的公司"),
           "title": ("减持方职务", "减持方担任的职务"), "date": ("减持日期", "减持的日期"),
           "share-per": ("占个人持股比例", "减持股份占减持方个人持股的百分比"),
           "share-org": ("占公司股份比例", "减持股份占公司股份的百分比")},
    "收购": {"sub-org": ("收购公司", "发起收购的公司"), "sub-per": ("收购人", "发起收购的个人"),
           "obj-org": ("被收购公司", "被收购的公司"), "way": ("收购方式", "如现金收购、股份收购、承担债务式收购"),
           "date": ("收购日期", "明确写出的收购日期"), "money": ("收购金额", "收购的金额"),
           "number": ("收购数量", "收购的股份数量"), "proportion": ("收购比例", "收购的股份比例")},
    "担保": {"sub-org": ("担保公司", "提供担保的公司"), "sub-per": ("担保人", "提供担保的个人"),
           "obj-org": ("被担保公司", "接受担保的公司"), "way": ("担保方式", "如保证、抵押、质押、留置、定金"),
           "amount": ("担保金额", "担保的金额"), "date": ("担保日期", "明确写出的担保日期")},
    "中标": {"sub": ("中标方", "中标的公司"), "obj": ("招标方", "招标的单位或组织"),
           "amount": ("中标金额", "中标的金额"), "date": ("中标日期", "中标的日期")},
    "签署合同": {"sub-org": ("甲方组织", "发起合同签署的组织或单位"), "sub-per": ("甲方个人", "发起合同签署的自然人"),
             "obj-org": ("乙方组织", "接受合同签署的组织或单位"), "obj-per": ("乙方个人", "接受合同签署的自然人"),
             "amount": ("成交额", "合同成交金额"), "date": ("签署日期", "签署合同的日期")},
    "判决": {"institution": ("裁判单位", "作出判决的法院"), "sub-per": ("原告个人", "原告中的个人"),
           "sub-org": ("原告公司", "原告中的公司"), "obj-per": ("被告个人", "被告中的个人"),
           "obj-org": ("被告公司", "被告中的公司"), "date": ("判决日期", "判决的日期"), "money": ("判决金额", "判决涉及的金额")},
}
CHFINANN_EVENTS = {"EquityPledge": "股权质押", "EquityFreeze": "股权冻结", "EquityRepurchase": "股份回购",
                   "EquityOverweight": "股东增持", "EquityUnderweight": "股东减持"}
CHFINANN_FIELDS = {
    "EquityPledge": {"Pledger": "质押人（出质股东）", "PledgedShares": "本次质押股数", "Pledgee": "质权人",
                     "TotalHoldingShares": "质押人持股总数", "TotalHoldingRatio": "质押人持股比例",
                     "TotalPledgedShares": "质押人累计质押股数", "StartDate": "质押起始日", "EndDate": "质押到期日",
                     "ReleasedDate": "解除质押日"},
    "EquityFreeze": {"EquityHolder": "被冻结股东", "FrozeShares": "冻结股数", "LegalInstitution": "执行冻结的司法机关",
                     "TotalHoldingShares": "该股东持股总数", "TotalHoldingRatio": "该股东持股比例", "StartDate": "冻结起始日",
                     "EndDate": "冻结到期日", "UnfrozeDate": "解冻日"},
    "EquityRepurchase": {"CompanyName": "回购公司", "HighestTradingPrice": "回购最高成交价", "LowestTradingPrice": "回购最低成交价",
                         "RepurchasedShares": "累计回购股数", "ClosingDate": "统计截止日", "RepurchaseAmount": "回购总金额"},
    "EquityOverweight": {"EquityHolder": "增持股东", "TradedShares": "增持股数", "StartDate": "增持起始日", "EndDate": "增持截止日",
                         "LaterHoldingShares": "增持后持股数", "AveragePrice": "增持均价"},
    "EquityUnderweight": {"EquityHolder": "减持股东", "TradedShares": "减持股数", "StartDate": "减持起始日", "EndDate": "减持截止日",
                          "LaterHoldingShares": "减持后持股数", "AveragePrice": "减持均价"},
}
# Mentions carry a guessed field; a question's distractors are mentions of the same kind.
FIELD_KIND = {"party": ["EquityHolder", "CompanyName", "Pledgee", "Pledger", "LegalInstitution", "StockAbbr"],
              "shares": ["PledgedShares", "TotalHoldingShares", "TotalPledgedShares", "FrozeShares", "TradedShares",
                         "RepurchasedShares", "LaterHoldingShares"],
              "ratio": ["TotalHoldingRatio", "OtherType"],
              "date": ["StartDate", "EndDate", "ReleasedDate", "UnfrozeDate", "ClosingDate"],
              "money": ["HighestTradingPrice", "LowestTradingPrice", "AveragePrice", "RepurchaseAmount"]}
KIND = {field: kind for kind, fields in FIELD_KIND.items() for field in fields}
KIND_NAMES = {"party": "公告中的主体", "shares": "股数", "ratio": "比例", "date": "日期", "money": "金额或价格"}
NOT_MENTIONED = "公告未提及"


class SentenceEvents(Source):
    """Shared views of DuEE and FewFC. Payload: [(event type, [(start, end, role)])], the trigger's role is `trigger`."""

    def __init__(self, name, dataset, roles, type_descriptions, domain, license=""):
        self.name, self.dataset, self.domain, self.license = name, dataset, domain, license
        self.roles = roles                      # event type -> {role key: (name, description)}
        self.type_descriptions = type_descriptions

    def label(self, event_type):
        return event_type

    def types_for(self, event_type):
        label = self.label(event_type)
        entries = [type_entry(TRIGGER, "触发词", f"表示“{label}”事件发生的核心词")]
        for key, (name, description) in self.roles[event_type].items():
            entries.append(type_entry(key, name, f"“{label}”事件中，{description}" if description else f"“{label}”事件的{name}"))
        return entries

    def siblings(self, event_type):
        return [t for t in self.roles if t != event_type]

    def views(self, raw, rng, options):
        text, events = raw.text, raw.payload
        views = Views(structured={"id": f"{self.name}/{raw.split}/{raw.uid}", "source": self.name, "text": text,
                                  "events": [{"type": t, "mentions": [{"start": s, "end": e, "role": r, "text": text[s:e]}
                                                                      for s, e, r in mentions]} for t, mentions in events]})
        stem = f"ee_{self.name}"
        present = sorted({t for t, _ in events})
        for index, event_type in enumerate(present):
            spans = {(s, e, r) for t, mentions in events if t == event_type for s, e, r in mentions
                     if r == TRIGGER or r in self.roles[event_type]}
            views.spans.append((stem, span_row(f"{stem}/{raw.split}/{raw.uid}#{index}", stem, text,
                                               self.types_for(event_type), [entity(*s) for s in sorted(spans)],
                                               self.meta(raw, "ee_argument", event_type=event_type))))
        absent = [t for t in self.roles if t not in present]
        negative = None
        if absent:
            near = [t for t in absent if any(t in self.siblings(p) for p in present)] or absent
            negative = rng.choice(near)
            if rng.random() < options["negative_event_rows"]:
                views.spans.append((stem, span_row(f"{stem}/{raw.split}/{raw.uid}#absent", stem, text,
                                                   self.types_for(negative), [],
                                                   self.meta(raw, "ee_argument", event_type=negative, negative=True))))
        state = {"句子": text}
        for event_type in present + ([negative] if negative else []):
            label = self.label(event_type)
            views.decisions.append(noul_case(
                f"ee_type/{self.name}/{raw.split}/{raw.uid}#{event_type}", state,
                f"这句话是否描述了“{label}”事件（{self.type_descriptions[event_type]}）？",
                event_type in present, self.meta(raw, "ee_type", event_type=event_type)))
        views.decisions.extend(self.choice_views(raw, present, state))
        return views

    def choice_views(self, raw, present, state):
        return []


class DuEE(SentenceEvents):
    """DuEE (HF processed version, 65 event types): validation is our test, 3% of train is dev."""

    def __init__(self, root):
        self.root = Path(root)
        roles = defaultdict(dict)
        for row in self.rows("train"):
            for event in row["events"]:
                for argument in event["args"]:
                    roles[event["label"]].setdefault(argument["label"], (argument["label"], ""))
        roles = {t: dict(sorted(r.items())) for t, r in sorted(roles.items())}
        self.categories = defaultdict(list)
        for event_type in roles:
            self.categories[event_type.partition("-")[0]].append(event_type.partition("-")[2])
        descriptions = {t: f"{t.partition('-')[0]}类事件" for t in roles}
        super().__init__("duee", "DuEE-HF", roles, descriptions, "news", license="DuEE（百度，研究用途）")

    def rows(self, split):
        path, = sorted((self.root / "DuEE-HF/data").glob(f"{split}-*.parquet"))
        return read_parquet(path).to_pylist()

    def label(self, event_type):
        return event_type.partition("-")[2]

    def siblings(self, event_type):
        category = event_type.partition("-")[0]
        return [t for t in self.roles if t != event_type and t.startswith(category + "-")]

    def records(self):
        for file_split, split in (("train", None), ("validation", "test")):
            for index, row in enumerate(self.rows(file_split)):
                text, events = row["text"], []
                for event in row["events"]:
                    if event["label"] not in self.roles:
                        continue
                    mentions = []
                    for item, role in [(event["trigger"], TRIGGER)] + [(a, a["label"]) for a in event["args"]]:
                        indices = item["indices"]
                        start, end = indices[0], indices[-1] + 1
                        if item.get("is_continuous", True) and text[start:end] == item["text"]:
                            mentions.append((start, end, role))
                    events.append((event["label"], mentions))
                if events:
                    yield Raw(f"{file_split}-{index}", split or carve(text, dev=0.03), text, events)

    def choice_views(self, raw, present, state):
        categories = {t.partition("-")[0] for t in present}
        if len(categories) != 1:
            return []
        criteria = {c: "包括：" + "、".join(subtypes) for c, subtypes in sorted(self.categories.items())}
        return [choice_case(f"ee_category/{self.name}/{raw.split}/{raw.uid}", state,
                            "按百度DuEE事件体系，判断这句话描述的事件属于哪一个大类。", criteria, categories.pop(),
                            self.meta(raw, "ee_category"))]


class FewFC(SentenceEvents):
    """FewFC `rearranged/` (10 financial event types): train_base + train_trans, test_base + test_trans; 3% of train is dev."""

    NO_EVENT = "无上述事件"

    def __init__(self, root):
        self.root = Path(root)
        super().__init__("fewfc", "FewFC", FEWFC_ROLES, FEWFC_TYPES, "finance", license="FewFC（研究用途）")

    def records(self):
        for file, split in (("train_base", None), ("train_trans", None), ("test_base", "test"), ("test_trans", "test")):
            for index, line in enumerate(open(self.root / f"FewFC/rearranged/{file}.json", encoding="utf-8")):
                row = json.loads(line)
                text, events = row["content"], []
                for event in row.get("events", []):
                    if event["type"] not in self.roles:
                        continue
                    mentions = [(m["span"][0], m["span"][1], m["role"]) for m in event["mentions"]
                                if text[m["span"][0]:m["span"][1]] == m["word"]]
                    events.append((event["type"], mentions))
                yield Raw(f"{file}-{row.get('id', index)}", split or carve(text, dev=0.03), text, events)

    def choice_views(self, raw, present, state):
        if len(present) > 1:
            return []
        criteria = dict(self.type_descriptions)
        criteria[self.NO_EVENT] = "这句话没有描述以上任何一类金融事件"
        return [choice_case(f"ee_category/{self.name}/{raw.split}/{raw.uid}", state,
                            "判断这句金融新闻描述的是下列哪一类事件。", criteria, present[0] if present else self.NO_EVENT,
                            self.meta(raw, "ee_category"))]


class ChFinAnn(Source):
    """ChFinAnn (Doc2EDAG, distant supervision): official train/dev/test announcements up to `max_doc_chars`."""

    quality = "distant"

    def __init__(self, root):
        self.root, self.name, self.dataset, self.domain = Path(root), "chfinann", "ChFinAnn", "finance"
        self.license = "Doc2EDAG（MIT）"

    def records(self):
        for split in ("train", "dev", "test"):
            with open(self.root / f"ChFinAnn/extracted/Data/{split}.json", encoding="utf-8") as stream:
                docs = json.load(stream)
            for doc_id, doc in docs:
                text = "\n".join(doc["sentences"])
                yield Raw(doc_id, split, text, doc)

    def views(self, raw, rng, options):
        doc, views = raw.payload, Views()
        if len(raw.text) > options["max_doc_chars"]:
            options["stats"]["chfinann:too_long"] += 1
            return views
        records = defaultdict(list)
        for _, event_type, fields in doc["recguid_eventname_eventdict_list"]:
            if event_type in CHFINANN_EVENTS:
                records[event_type].append(fields)
        state = {"公告": raw.text}
        absent = [t for t in CHFINANN_EVENTS if t not in records]
        asked = sorted(records) + ([rng.choice(absent)] if absent else [])
        for event_type in asked:
            views.decisions.append(noul_case(
                f"doc_event/{self.name}/{raw.split}/{raw.uid}#{event_type}", state,
                f"这篇上市公司公告是否披露了{CHFINANN_EVENTS[event_type]}事件？", event_type in records,
                self.meta(raw, "doc_event", event_type=event_type)))
        guesses = doc["ann_mspan2guess_field"]
        for event_type, items in sorted(records.items()):
            questions = []
            for field, field_name in CHFINANN_FIELDS[event_type].items():
                values = {item.get(field) for item in items}
                if len(values) != 1:
                    continue                    # records of this type disagree: the question would be ambiguous
                value = values.pop()
                if value is None and rng.random() >= options["not_mentioned_rate"]:
                    continue
                questions.append((field, field_name, value))
            rng.shuffle(questions)
            for field, field_name, value in questions[:options["fields_per_event"]]:
                pool = [m for m, guess in sorted(guesses.items()) if KIND.get(guess) == KIND[field]]
                if value is not None and value not in guesses:
                    continue
                keys = sample_candidates([value] if value else [], pool, rng, limit=15)
                if not keys:
                    continue
                criteria = {key: KIND_NAMES[KIND[field]] for key in keys}
                criteria[NOT_MENTIONED] = "公告没有给出这一项"
                views.decisions.append(choice_case(
                    f"doc_field/{self.name}/{raw.split}/{raw.uid}#{event_type}.{field}", state,
                    f"这篇公告披露了{CHFINANN_EVENTS[event_type]}事件。该事件的“{field_name}”是哪一项？",
                    criteria, value or NOT_MENTIONED, self.meta(raw, "doc_field", event_type=event_type, field=field)))
        return views


def sources(root):
    return [DuEE(root), FewFC(root), ChFinAnn(root)]
