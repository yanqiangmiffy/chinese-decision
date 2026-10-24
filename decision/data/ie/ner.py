"""Chinese NER sources: standalone corpora with exact offsets and the B2NERD_all subsets.

Every source uses B2NERD's standardized Chinese labels and their descriptions. Standalone corpora
map their tags onto the labels B2NERD gives the same corpus (checked against B2NERD on the training
texts: MSRA, CLUENER, Resume, Weibo, CMeEE-V2, Boson and MultiCoNER2 agree on 99%+ of entities), so
"人名" means the same everywhere and differing annotation standards show in the descriptions.

Names are made readable ("组织机构名->公司" -> "公司", "地名->其它" -> "其它地名"); notes in parentheses
move into the description. Numbered B2NERD variants without a description ("品牌名2") merge into
their base label, as B2NERD's own pruned release does.
"""
import json
import re
from pathlib import Path

from .common import (OTHER, Raw, Source, Views, bio_entities, carve, choice_case, entity, join_tokens, locate,
                     mark, read_parquet, sample_candidates, type_entry, window_rows)

B2NERD = "B2NERD/extracted/B2NERD_all/NER_zh"
TYPING_INSTRUCTIONS = "文本中用【】标出的部分是一个实体。根据各类型的定义，判断它属于哪一类。"

MSRA = {"LOC": "地名或地理政治实体", "ORG": "组织机构名", "PER": "人名"}
CLUENER = {"name": "人名", "company": "组织机构名->公司", "game": "作品名->游戏作品", "organization": "组织机构名->其它",
           "movie": "作品名->影像作品", "address": "地名->其它", "position": "职业或职位",
           "government": "组织机构名->政府机构", "scene": "地名->景点", "book": "作品名->文字作品"}
RESUME = {"NAME": "人名", "CONT": "国籍", "RACE": "民族", "TITLE": "头衔", "EDU": "教育背景", "ORG": "组织机构名",
          "PRO": "专业名", "LOC": "籍贯"}
WEIBO = {"PER.NAM": "人名或昵称", "PER.NOM": "人群泛称", "LOC.NAM": "地点(不带地理政治实体)", "LOC.NOM": "地点泛称",
         "GPE.NAM": "地理政治实体", "GPE.NOM": "地理政治实体泛称", "ORG.NAM": "组织机构名(不带地理政治实体)",
         "ORG.NOM": "组织机构泛称"}
CMEEE = {"dis": "疾病诊断", "sym": "症状或体征描述", "pro": "检查或治疗程序", "equ": "医疗设备", "dru": "药物的名称或类别",
         "ite": "医学检查项目->实验室检验项目", "bod": "身体部位或身体物质", "dep": "医院科室", "mic": "微生物名"}
BOSON = {"person_name": "人名", "location": "地名或地理政治实体(非标准)", "time": "时间相关表述",
         "company_name": "组织机构名->公司", "product_name": "产品名(非标准)", "org_name": "组织机构名->其它(非标准)"}
MULTICONER = {"Artist": "人名->艺术家", "Athlete": "人名->运动员", "Politician": "人名->政治人物", "Cleric": "人名->神职人员",
              "Scientist": "人名->科学家", "SportsManager": "人名->体育经理", "OtherPER": "人名->其它",
              "Facility": "地名->设施", "HumanSettlement": "地名->人类居住地", "Station": "地名->车站", "OtherLOC": "地名->其它",
              "VisualWork": "作品名->影像作品", "MusicalWork": "作品名->音乐作品", "WrittenWork": "作品名->文字作品",
              "ArtWork": "作品名->艺术作品", "Software": "作品名->软件", "MusicalGRP": "组织机构名->音乐团体",
              "PublicCorp": "组织机构名->公共公司", "PrivateCorp": "组织机构名->私营公司", "SportsGRP": "组织机构名->体育团队",
              "AerospaceManufacturer": "组织机构名->航空航天制造商", "CarManufacturer": "组织机构名->汽车制造商",
              "ORG": "组织机构名->其它", "Clothing": "产品名->服装", "Vehicle": "产品名->交通工具", "Food": "产品名->食品",
              "Drink": "产品名->饮品", "OtherPROD": "产品名->其它", "Medication/Vaccine": "药物名",
              "MedicalProcedure": "检查或治疗程序", "AnatomicalStructure": "解剖部位(含动植物)", "Symptom": "症状(非标准)",
              "Disease": "疾病名"}
# Not in B2NERD. WikiANN links Wikipedia anchors automatically; MultiNERD is machine-annotated Wikipedia/WikiNews.
WIKIANN = {"PER": ("人名", "维基百科文本中的人物姓名"), "ORG": ("组织机构名", "维基百科文本中的组织、机构、团体名称"),
           "LOC": ("地名", "维基百科文本中的地理位置名称，包括国家、城市、山川、建筑等")}
MULTINERD = {"PER": ("人名", "真实或虚构人物的姓名"), "ORG": ("组织机构名", "公司、政府部门、学校、团体等组织的名称"),
             "LOC": ("地名", "国家、城市、地区、山川、建筑等地理位置"), "ANIM": ("动物", "动物的种类或名称"),
             "BIO": ("微生物", "细菌、病毒等微生物的名称"), "CEL": ("天体", "行星、恒星、星系等天体的名称"),
             "DIS": ("疾病", "疾病、病症的名称"), "EVE": ("事件", "有名称的历史事件、战役、赛事、节日等"),
             "FOOD": ("食物", "食物、菜肴、饮品的名称"), "INST": ("器具", "乐器、工具等人造器具的名称"),
             "MEDIA": ("媒体作品", "书籍、电影、歌曲、报刊、电视节目等作品或媒体的名称"),
             "MYTH": ("神话人物", "神话、宗教或传说中的人物与神祇"), "PLANT": ("植物", "植物的种类或名称"),
             "TIME": ("时期", "有名称的历史时期或时代，如文艺复兴、战国时期"), "VEHI": ("交通工具", "车辆、船舶、飞机等交通工具的名称或型号")}
MULTINERD_TAGS = ["O"] + [f"{p}-{t}" for t in MULTINERD for p in "BI"]
# B2NERD subsets read as they are. Corpora that also exist standalone (Boson, CBLUE, CLUENER, MSRA, Resume,
# Weibo, MultiCoNER23) come from their own files. Left out as exact-text subsets of another subset:
# CCIR2021 and finance-sina (of zh-ontonotes), CCKS2018 (777 of its 797 texts are in CCKS2017).
B2NERD_SUBSETS = {
    "bank": "finance", "CAIL2021": "legal", "CAIL2022": "legal", "CCKS2017": "medical",
    "CCKS2019": "medical", "CCKS2020": "medical", "CCKS2021address": "general", "chineseliterature-ner": "general",
    "FNED": "general", "gaiic2022": "ecommerce", "imcs21": "medical",
    "math-high": "education", "Microbiota": "medical", "military-ner": "military", "multiconer22": "general",
    "nlpcc2018": "dialogue", "peopledaily2014": "general", "ruijinmcc": "medical", "Taobao": "ecommerce",
    "wanchuang": "medical", "youku": "general", "zh-ontonotes": "general",
}


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def readable(label):
    """Display name and the notes taken out of a B2NERD label."""
    notes = [note for note in re.findall(r"\(([^)]*)\)", label) if note != "非标准"]
    name = re.sub(r"\([^)]*\)", "", label).strip()
    if "->" in name:
        parent, leaf = name.split("->", 1)
        name = OTHER + parent if leaf == OTHER else leaf
    return name, notes


def b2nerd_types(labels, descriptions):
    """Type entries keyed by label; colliding display names fall back to the label itself."""
    entries = {}
    for label in labels:
        name, notes = readable(label)
        description = descriptions.get(label, "")
        if notes:
            description += "（" + "；".join(notes) + "）"
        entries[label] = [name, description]
    names = [name for name, _ in entries.values()]
    for label, entry in entries.items():
        if names.count(entry[0]) > 1:
            entry[0] = label
    return [type_entry(label, name, description) for label, (name, description) in entries.items()]


def merged_labels(labels, descriptions):
    """Numbered variants without a description ("品牌名2") -> their base label; other labels map to themselves."""
    merge = {}
    for label in labels:
        match = re.fullmatch(r"(.*\D)\d", label)
        if label not in descriptions and match and match.group(1) in labels:
            merge[label] = match.group(1)
    return {label: merge.get(label, label) for label in labels}


class NerSource(Source):
    """Rows carry every type of the corpus; one entity per record also becomes a typing decision."""

    def __init__(self, name, dataset, types, read, quality="gold", domain="general", license=""):
        self.name, self.dataset, self.types, self.read = name, dataset, types, read
        self.quality, self.domain, self.license = quality, domain, license
        self.names = {entry["key"]: entry["name"] for entry in types}
        self.criteria = {entry["name"]: entry["description"] or entry["name"] for entry in types}

    def records(self):
        return self.read()

    def views(self, raw, rng, options):
        entities = raw.payload
        rows, lost = window_rows(f"ner/{self.name}/{raw.split}/{raw.uid}", f"ner_{self.name}", raw.text, self.types,
                                 entities, self.meta(raw, "ner"), options["max_chars"])
        options["stats"][f"ner:{self.name}:lost_at_cuts"] += lost
        views = Views(spans=[(f"ner_{self.name}", row) for row in rows])
        candidates = [row for row in rows if row["entities"]]
        if candidates and len(self.criteria) >= 2:
            row = rng.choice(candidates)
            picked = rng.choice(row["entities"])
            labels = {e["label"] for e in row["entities"] if (e["start"], e["end"]) == (picked["start"], picked["end"])}
            gold = [self.names[label] for label in labels]
            keys = sample_candidates(gold, list(self.criteria), rng)
            views.decisions.append(choice_case(
                f"ner_typing/{self.name}/{raw.split}/{raw.uid}",
                {"文本": mark(row["text"], picked["start"], picked["end"])}, TYPING_INSTRUCTIONS,
                {key: self.criteria[key] for key in keys}, gold, self.meta(raw, "ner_typing")))
        return views


def parquet_rows(path):
    table = read_parquet(path)
    return table, table.to_pylist()


def tag_names(table):
    info = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    return info["ner_tags"]["feature"]["names"]


def bio_records(root, splits, clean=None, tags=None):
    """Character-tagged parquet/jsonl rows -> Raw records with character-offset entities.

    `splits` maps our split to (file split, glob); the split "carve" takes 3% of the file as dev.
    """
    for split, (file_split, path) in splits.items():
        for file in sorted(Path(root).glob(path)):
            if file.suffix == ".parquet":
                table, rows = parquet_rows(file)
                names = tags or tag_names(table)
            else:
                rows, names = [json.loads(line) for line in open(file, encoding="utf-8")], tags
            for index, row in enumerate(rows):
                tokens, labels = list(row["tokens"]), [names[i] for i in row["ner_tags"]]
                if clean:
                    tokens, labels = clean(tokens, labels)
                    if tokens is None:
                        continue
                text, starts = join_tokens(tokens)
                if not text.strip():
                    continue
                entities = [entity(starts[s], starts[e - 1] + len(tokens[e - 1]), label)
                            for s, e, label in bio_entities(labels)]
                yield Raw(f"{file_split}-{index}", split if split != "carve" else carve(text, dev=0.03), text, entities)


def offset_records(root, pattern, splits, offsets):
    """Rows with character-offset entities; `offsets(row)` gives (start, end, label, surface) tuples."""
    for split, file_split in splits.items():
        for file in sorted(Path(root).glob(pattern.format(file_split))):
            rows = parquet_rows(file)[1] if file.suffix == ".parquet" else read_json(file)
            for index, row in enumerate(rows):
                text = row["text"]
                entities = [entity(s, e, label) for s, e, label, surface in offsets(row)
                            if 0 <= s < e <= len(text) and text[s:e] == surface]
                assigned = split if split != "carve" else carve(text, dev=0.03)
                yield Raw(f"{file_split}-{index}", assigned, text, entities)


def mapped_types(root, subset, mapping):
    descriptions = read_json(Path(root) / B2NERD / subset / "label_description_v1.json")
    full = b2nerd_types(list(dict.fromkeys(mapping.values())), descriptions)
    by_label = {entry["key"]: entry for entry in full}
    return [type_entry(tag, by_label[label]["name"], by_label[label]["description"]) for tag, label in mapping.items()]


def strip_weibo_suffix(tokens, labels):
    """Weibo's segmentation suffixes left on Latin letters ("h1 o1 s1 t1") lose their digit."""
    return [token[0] if re.fullmatch(r"[A-Za-z]\d", token) else token for token in tokens], labels


def clean_multinerd(tokens, labels):
    """MultiNERD-zh: rows with a "None" token (lost characters) are dropped; a leading "#" marker goes."""
    if "None" in tokens:
        return None, None
    while tokens and tokens[0] == "#" and labels[0] == "O":
        tokens, labels = tokens[1:], labels[1:]
    return tokens, labels


def boson_records(root):
    """Boson's inline `{{type:text}}` markup, nested in a few lines, parsed with a stack; split by paragraph hash."""
    for index, line in enumerate(open(Path(root) / "Boson-NER/origindata.txt", encoding="utf-8")):
        line = line.rstrip("\n")
        text, stack, entities, position = [], [], [], 0
        while position < len(line):
            if line.startswith("{{", position) and ":" in line[position:]:
                colon = line.index(":", position)
                stack.append((line[position + 2:colon], sum(map(len, text))))
                position = colon + 1
            elif line.startswith("}}", position) and stack:
                label, start = stack.pop()
                surface = "".join(text)[start:]
                lead = len(surface) - len(surface.lstrip())
                end = start + len(surface.rstrip())
                if start + lead < end and label in BOSON:
                    entities.append(entity(start + lead, end, label))
                position += 2
            else:
                text.append(line[position])
                position += 1
        text = "".join(text)
        if text.strip():
            yield Raw(f"line-{index}", carve(text, dev=0.03, test=0.07), text, entities)


def b2nerd_records(root, subset):
    """B2NERD_all rows: entities are (type, name) sets, placed by `locate`; rows whose entities do not occur are dropped."""
    folder = Path(root) / B2NERD / subset
    labels = read_json(folder / "labels.json")
    merge = merged_labels(labels, read_json(folder / "label_description_v1.json"))
    files = {split: folder / f"{split}.json" for split in ("train", "dev", "test")}
    rows = {split: read_json(path) for split, path in files.items() if path.exists()}
    present = {split for split, items in rows.items() if items}
    for split in sorted(present):
        for row in rows.pop(split):
            text = row["sentence"]
            if subset.startswith("multiconer") and text.startswith("#"):
                text = text[1:]
            if not text.strip():
                continue
            spans, missing = locate(text, [(merge[e["type"]], e["name"]) for e in row["entities"]])
            if missing:
                continue
            if split == "train":
                assigned = carve(text, dev=0.03 if "dev" not in present else 0.0,
                                 test=0.05 if "test" not in present else 0.0)
            else:
                assigned = split
            yield Raw(f"{split}-{row['id']}", assigned, text, spans)


def b2nerd_source(root, subset, domain):
    folder = Path(root) / B2NERD / subset
    labels = read_json(folder / "labels.json")
    descriptions = read_json(folder / "label_description_v1.json")
    merge = merged_labels(labels, descriptions)
    types = b2nerd_types([label for label in labels if merge[label] == label], descriptions)
    name = subset.lower().replace("-", "_")
    return NerSource(name, f"B2NERD_all/{subset}", types, lambda: b2nerd_records(root, subset), domain=domain,
                     license="B2NERD（来源数据各自的许可）")


def sources(root):
    """All NER sources, standalone first."""
    root = Path(root)
    found = []

    def cluener_offsets(row):
        return [(e["start_offset"], e["end_offset"], e["label"], e["entity"]) for e in row["entities"]]

    found.append(NerSource(
        "msra", "MSRA-NER", mapped_types(root, "MSRA", MSRA),
        lambda: bio_records(root, {"carve": ("train", "MSRA-NER/data/train-*.parquet"),
                                         "test": ("test", "MSRA-NER/data/test-*.parquet")}), domain="news"))
    found.append(NerSource(
        "people_daily", "PeopleDaily-NER", mapped_types(root, "MSRA", MSRA),
        lambda: offset_records(root, "PeopleDaily-NER/data/{}-*.parquet",
                               {"train": "train", "dev": "validation", "test": "test"}, cluener_offsets),
        domain="news"))
    found.append(NerSource(
        "cluener", "CLUENER2020", mapped_types(root, "CLUENER", CLUENER),
        lambda: offset_records(root, "CLUENER2020/data/{}-*.parquet", {"carve": "train", "test": "validation"},
                               cluener_offsets)))
    found.append(NerSource(
        "resume", "Resume-NER", mapped_types(root, "resume", RESUME),
        lambda: bio_records(root, {"train": ("train", "Resume-NER/data/train-*.parquet"),
                                         "dev": ("validation", "Resume-NER/data/validation-*.parquet"),
                                         "test": ("test", "Resume-NER/data/test-*.parquet")})))
    found.append(NerSource(
        "weibo", "Weibo-NER", mapped_types(root, "weibo", WEIBO),
        lambda: bio_records(root, {"train": ("train", "Weibo-NER/data/train-*.parquet"),
                                         "dev": ("validation", "Weibo-NER/data/validation-*.parquet"),
                                         "test": ("test", "Weibo-NER/data/test-*.parquet")}, clean=strip_weibo_suffix),
        domain="social"))
    found.append(NerSource(
        "cmeee_v2", "CMeEE-V2", mapped_types(root, "CBLUE", CMEEE),
        lambda: offset_records(root, "CMeEE-V2/CMeEE-V2_{}.json", {"carve": "train", "test": "dev"},
                               lambda row: [(e["start_idx"], e["end_idx"] + 1, e["type"], e["entity"])
                                            for e in row["entities"]]), domain="medical", license="CBLUE"))
    found.append(NerSource("boson", "Boson-NER", mapped_types(root, "Boson", BOSON), lambda: boson_records(root),
                           domain="news", license="Boson 数据许可（见 Boson-NER 目录）"))
    found.append(NerSource(
        "multiconer2", "MultiCoNER2-zh", mapped_types(root, "multiconer23", MULTICONER),
        lambda: conll_records(root), license="CC BY 4.0"))
    found.append(NerSource(
        "wikiann", "WikiANN-zh", [type_entry(tag, *WIKIANN[tag]) for tag in WIKIANN],
        lambda: bio_records(root, {"train": ("train", "WikiANN-zh/zh/train-*.parquet"),
                                         "dev": ("validation", "WikiANN-zh/zh/validation-*.parquet"),
                                         "test": ("test", "WikiANN-zh/zh/test-*.parquet")}), quality="silver"))
    found.append(NerSource(
        "multinerd", "MultiNERD-zh", [type_entry(tag, *MULTINERD[tag]) for tag in MULTINERD],
        lambda: bio_records(root, {"train": ("train", "MultiNERD-zh/train/train_zh.jsonl"),
                                         "dev": ("val", "MultiNERD-zh/val/val_zh.jsonl"),
                                         "test": ("test", "MultiNERD-zh/test/test_zh.jsonl")},
                            clean=clean_multinerd, tags=MULTINERD_TAGS), quality="silver", license="CC BY-NC-SA 4.0"))
    for subset, domain in B2NERD_SUBSETS.items():
        found.append(b2nerd_source(root, subset, domain))
    return found


def conll_records(root):
    """MultiCoNER2 conll: one token per line (characters, or Latin words), `# id` headers between sentences."""
    for split, file_split in (("train", "train"), ("dev", "dev"), ("test", "test")):
        tokens, labels, index = [], [], 0
        lines = open(Path(root) / f"MultiCoNER2-zh/ZH-Chinese/zh_{file_split}.conll", encoding="utf-8").read().split("\n")
        for line in lines + ["# end"]:
            if line.startswith("# id") or line == "# end" or not line.strip():
                if tokens:
                    text, starts = join_tokens(tokens)
                    entities = [entity(starts[s], starts[e - 1] + len(tokens[e - 1]), label)
                                for s, e, label in bio_entities(labels)]
                    yield Raw(f"{file_split}-{index}", split, text, entities)
                    index += 1
                tokens, labels = [], []
                continue
            parts = line.split()
            tokens.append(parts[0])
            labels.append(parts[-1])
