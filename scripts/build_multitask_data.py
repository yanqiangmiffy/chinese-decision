"""Build a small classification + extraction mix for multi-task feasibility runs.

Sources (local copies, see G:/pretrained_models/datasets/ie_collections/README_数据集总索引.md):
- extraction: CLUENER2020 (10 fine-grained types) and PeopleDaily NER (PER/LOC/ORG), character offsets.
- classification: DuEE sentences with a single event category, as a new 9-way choice decision.
- retention: CJ-Dataset classification sources the released model was trained on (Ifeng news topics,
  ChnSentiCorp, MASSIVE intents), as replay for training and validation for forgetting checks,
  plus a random sample of other CJ general validation decisions.

Public test files with empty labels are never used: CLUENER's official validation split is our test set.
Needs pandas + pyarrow only, so it runs in any environment that reads parquet.

Output layout (`--out`):
    extraction/{source}.{train,dev,test}.jsonl  {"id", "source", "text", "types": [...], "entities": [...]}
    decision/{train,dev,test}.cases.jsonl       CJ case format, `_meta.source_family` names the source
    manifest.json
"""
import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CLUENER_TYPES = [
    ("name", "人名", "真实或虚构人物的姓名"),
    ("address", "地址", "省市区县、街道门牌等具体地点"),
    ("company", "公司", "企业、公司的名称"),
    ("government", "政府机构", "政府部门与行政机关"),
    ("organization", "组织机构", "学校、协会、球队等非政府、非公司的组织"),
    ("position", "职位", "职务与头衔"),
    ("book", "书名", "小说、杂志、教材等出版物"),
    ("game", "游戏", "电子游戏或游戏名称"),
    ("movie", "影视作品", "电影、电视剧、综艺节目"),
    ("scene", "景点", "旅游景点与地标"),
]
PEOPLE_DAILY_TYPES = [
    ("PER", "人名", "人物的姓名"),
    ("LOC", "地名", "国家、城市、山川等地理位置"),
    ("ORG", "机构名", "组织、团体、机构的名称"),
]
# Test only: none of 国籍/学历/专业/民族/职称 is a training type, so Resume measures unseen types.
RESUME_TYPES = [
    ("NAME", "人名", "人物的姓名"),
    ("CONT", "国籍", "国家或地区归属，如“中国国籍”"),
    ("RACE", "民族", "民族名称，如汉族"),
    ("EDU", "学历", "学历与学位，如本科、硕士"),
    ("PRO", "专业", "所学专业"),
    ("TITLE", "职称", "职务、职称与头衔"),
    ("ORG", "组织机构", "公司、学校、政府等机构的名称"),
    ("LOC", "地点", "籍贯等地名"),
]
RESUME_TAGS = ["O", "B-CONT", "I-CONT", "B-EDU", "I-EDU", "B-LOC", "I-LOC", "B-NAME", "I-NAME", "B-ORG", "I-ORG",
               "B-PRO", "I-PRO", "B-RACE", "I-RACE", "B-TITLE", "I-TITLE"]
DUEE_INSTRUCTIONS = "按百度DuEE事件体系，判断这句新闻描述的事件属于哪一个大类。"
CJ_RETENTION = ("ifeng_news", "chnsenticorp", "slurp_massive")


def parquet(root, pattern):
    paths = sorted(Path(root).glob(pattern))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one file for {pattern}, found {paths}")
    return pd.read_parquet(paths[0])


def type_list(types):
    return [{"key": key, "name": name, "description": description} for key, name, description in types]


def span_rows(frame, source, split, types, stats):
    """Rows with character-offset entities; entities whose text disagrees with the offsets are dropped."""
    keys = {key for key, _, _ in types}
    for index, (text, entities) in enumerate(zip(frame["text"], frame["entities"])):
        kept = []
        for entity in entities:
            start, end, label = int(entity["start_offset"]), int(entity["end_offset"]), entity["label"]
            if label not in keys or not 0 <= start < end <= len(text) or text[start:end] != entity["entity"]:
                stats[f"{source}:{split}:dropped_entities"] += 1
                continue
            kept.append({"start": start, "end": end, "label": label})
        stats[f"{source}:{split}:entities"] += len(kept)
        yield {"id": f"{source}-{split}-{index}", "source": source, "text": text,
               "types": type_list(types), "entities": sorted(kept, key=lambda e: (e["start"], e["end"]))}


def bio_rows(frame, source, split, types, tags, stats):
    """Character-tagged BIO rows as character-offset entities; an I- tag without its B- starts a new entity."""
    for index, (chars, ids) in enumerate(zip(frame["tokens"], frame["ner_tags"])):
        if any(len(c) != 1 for c in chars):
            stats[f"{source}:{split}:skipped_rows"] += 1
            continue
        entities, current = [], None
        for position, tag in enumerate([tags[i] for i in ids] + ["O"]):
            prefix, _, label = tag.partition("-")
            if current and (prefix != "I" or label != current["label"]):
                current["end"] = position
                entities.append(current)
                current = None
            if prefix == "B" or (prefix == "I" and current is None):
                current = {"start": position, "label": label}
        stats[f"{source}:{split}:entities"] += len(entities)
        yield {"id": f"{source}-{split}-{index}", "source": source, "text": "".join(chars),
               "types": type_list(types), "entities": entities}


def duee_rows(frame, split):
    """Sentences whose events all fall in one category, and the subtypes seen per category."""
    rows, subtypes = [], defaultdict(set)
    for index, (text, events) in enumerate(zip(frame["text"], frame["events"])):
        labels = [event["label"] for event in events]
        for label in labels:
            category, _, subtype = label.partition("-")
            subtypes[category].add(subtype)
        categories = {label.partition("-")[0] for label in labels}
        if len(categories) == 1:
            rows.append((f"duee-{split}-{index}", text, categories.pop()))
    return rows, subtypes


def duee_case(row_id, text, category, criteria, split):
    return {"id": row_id, "state": {"新闻句子": text},
            "questions": {"q": {"type": "choice", "instructions": DUEE_INSTRUCTIONS, "criteria": criteria}},
            "gold": {"q": category},
            "_meta": {"source_family": "duee_category", "task_id": "duee_event_category", "split": split}}


def reservoir_cj(path, families, per_family, rng, exclude=()):
    """Uniform samples per source family in one pass; a substring test skips most lines unparsed.

    The test matches the family name only, so it is valid for compact and spaced JSON alike;
    the parsed `_meta.source_family` decides.
    """
    markers = {family: f'"{family}"' for family in families}
    seen, buckets = Counter(), defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            family = next((f for f, marker in markers.items() if marker in line), None)
            if family is None:
                continue
            case = json.loads(line)
            if (case.get("_meta") or {}).get("source_family") != family or case["id"] in exclude:
                continue
            seen[family] += 1
            bucket = buckets[family]
            if len(bucket) < per_family[family]:
                bucket.append(case)
            else:
                i = rng.randrange(seen[family])
                if i < per_family[family]:
                    bucket[i] = case
    return buckets, seen


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ie-root", default="G:/pretrained_models/datasets/ie_collections")
    parser.add_argument("--cj-root", default="G:/pretrained_models/datasets/CJ-Dataset")
    parser.add_argument("--out", default="datasets/multitask-mini")
    parser.add_argument("--cluener-train", type=int, default=4000)
    parser.add_argument("--people-daily-train", type=int, default=3000)
    parser.add_argument("--people-daily-test", type=int, default=1000)
    parser.add_argument("--duee-train", type=int, default=3000)
    parser.add_argument("--dev", type=int, default=300, help="dev rows per source")
    parser.add_argument("--cj-replay", type=int, default=600, help="CJ training decisions per retention source")
    parser.add_argument("--cj-eval", type=int, default=400, help="CJ validation decisions per retention source")
    parser.add_argument("--cj-other-eval", type=int, default=600, help="other CJ general validation decisions")
    parser.add_argument("--skip-cj-train", action="store_true", help="no replay (skips the 6 GB training scan)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng, stats, out = random.Random(args.seed), Counter(), Path(args.out)
    counts = {}

    def sample(rows, n):
        rows = list(rows)
        rng.shuffle(rows)
        return rows[:n], rows[n:]

    # CLUENER: official validation is the test set; dev comes from unused training rows.
    train = list(span_rows(parquet(args.ie_root, "CLUENER2020/data/train-*.parquet"), "cluener", "train",
                           CLUENER_TYPES, stats))
    chosen, rest = sample(train, args.cluener_train)
    test = list(span_rows(parquet(args.ie_root, "CLUENER2020/data/validation-*.parquet"), "cluener", "test",
                          CLUENER_TYPES, stats))
    for split, rows in (("train", chosen), ("dev", rest[:args.dev]), ("test", test)):
        counts[f"extraction/cluener.{split}"] = write_jsonl(out / "extraction" / f"cluener.{split}.jsonl", rows)

    splits = {name: list(span_rows(parquet(args.ie_root, f"PeopleDaily-NER/data/{name}-*.parquet"), "people_daily",
                                   name, PEOPLE_DAILY_TYPES, stats)) for name in ("train", "validation", "test")}
    for split, rows in (("train", sample(splits["train"], args.people_daily_train)[0]),
                        ("dev", sample(splits["validation"], args.dev)[0]),
                        ("test", sample(splits["test"], args.people_daily_test)[0])):
        counts[f"extraction/people_daily.{split}"] = write_jsonl(
            out / "extraction" / f"people_daily.{split}.jsonl", rows)

    resume = list(bio_rows(parquet(args.ie_root, "Resume-NER/data/test-*.parquet"), "resume_zero_shot", "test",
                           RESUME_TYPES, RESUME_TAGS, stats))
    counts["extraction/resume_zero_shot.test"] = write_jsonl(out / "extraction" / "resume_zero_shot.test.jsonl", resume)

    # DuEE event categories; descriptions list the subtypes seen in training.
    duee_train, subtypes = duee_rows(parquet(args.ie_root, "DuEE-HF/data/train-*.parquet"), "train")
    duee_test, _ = duee_rows(parquet(args.ie_root, "DuEE-HF/data/validation-*.parquet"), "test")
    criteria = {category: "包括：" + "、".join(sorted(subtypes[category])) for category in sorted(subtypes)}
    chosen, rest = sample(duee_train, args.duee_train)
    decision = {"train": [duee_case(*row, criteria, "train") for row in chosen],
                "dev": [duee_case(*row, criteria, "dev") for row in rest[:args.dev]],
                "test": [duee_case(*row, criteria, "test") for row in duee_test if row[2] in criteria]}
    stats["duee:categories"] = len(criteria)

    # CJ retention: validation samples for evaluation, training samples for replay.
    validation = Path(args.cj_root) / "general" / "validation.cases.jsonl.gz"
    wanted = {family: args.cj_eval for family in CJ_RETENTION}
    buckets, seen = reservoir_cj(validation, CJ_RETENTION, wanted, rng)
    for family in CJ_RETENTION:
        decision["test"].extend(buckets[family])
        stats[f"cj_validation:{family}"] = seen[family]
    others, retention_ids = [], {case["id"] for family in CJ_RETENTION for case in buckets[family]}
    with gzip.open(validation, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            case = json.loads(line)
            if (case.get("_meta") or {}).get("source_family") not in CJ_RETENTION:
                others.append(case)
    for case in sample(others, args.cj_other_eval)[0]:
        case["_meta"]["source_family"] = "cj_other:" + case["_meta"]["source_family"]
        decision["test"].append(case)
    if not args.skip_cj_train:
        train_path = Path(args.cj_root) / "general" / "train.cases.jsonl.gz"
        buckets, seen = reservoir_cj(train_path, CJ_RETENTION, {f: args.cj_replay for f in CJ_RETENTION}, rng,
                                     exclude=retention_ids)
        for family in CJ_RETENTION:
            decision["train"].extend(buckets[family])
            stats[f"cj_train:{family}"] = seen[family]
    for split, cases in decision.items():
        rng.shuffle(cases)
        counts[f"decision/{split}"] = write_jsonl(out / "decision" / f"{split}.cases.jsonl", cases)
        for case in cases:
            stats[f"decision:{split}:{case['_meta']['source_family']}"] += 1

    manifest = {"seed": args.seed, "arguments": vars(args), "files": counts, "stats": dict(sorted(stats.items())),
                "duee_categories": criteria}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
