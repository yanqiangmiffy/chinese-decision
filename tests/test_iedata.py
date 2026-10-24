import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from decision.data.ie.common import (Raw, bio_entities, carve, choice_case, chunks, join_tokens, locate, mark,
                                    window_rows)
from decision.data.ie.event import FewFC
from decision.data.ie.ner import b2nerd_types, boson_records, merged_labels, readable
from decision.data.ie.relation import FinRE, TripleSource
from decision.processing import examples_for

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_ie_data import leakage_filter  # noqa: E402

OPTIONS = {"max_chars": 300, "max_subject_chars": 40, "negative_pairs": 1.0, "negative_event_rows": 1.0,
           "max_doc_chars": 3000, "fields_per_event": 3, "not_mentioned_rate": 1.0, "stats": Counter()}


def spans(entities):
    return [(e["start"], e["end"], e["label"]) for e in entities]


def valid(case):
    example, = examples_for(case, "general")
    return example


class TokenTests(unittest.TestCase):
    def test_bio_entities_start_a_new_entity_on_a_stray_inside_tag_or_label_change(self):
        self.assertEqual(bio_entities(["B-PER", "I-PER", "I-LOC", "O", "I-ORG", "B-ORG"]),
                         [(0, 2, "PER"), (2, 3, "LOC"), (4, 5, "ORG"), (5, 6, "ORG")])

    def test_join_tokens_keeps_spaces_between_latin_words_only(self):
        text, starts = join_tokens(["由", "saenuri", "dang", "领", "导"])
        self.assertEqual(text, "由saenuri dang领导")
        self.assertEqual(starts, [0, 1, 9, 13, 14, 15])
        self.assertEqual(join_tokens(list("Eternal"))[0], "Eternal")


class LocateTests(unittest.TestCase):
    def test_every_occurrence_outside_a_longer_mention_is_marked(self):
        entities, missing = locate("北京大学在北京，北京很大", [("LOC", "北京"), ("ORG", "北京大学")])
        self.assertEqual(spans(entities), [(0, 4, "ORG"), (5, 7, "LOC"), (8, 10, "LOC")])
        self.assertEqual(missing, [])

    def test_a_surface_only_inside_longer_mentions_is_nested_and_two_labels_share_a_span(self):
        entities, _ = locate("北京大学很好", [("LOC", "北京"), ("ORG", "北京大学"), ("学校", "北京大学")])
        self.assertEqual(spans(entities), [(0, 2, "LOC"), (0, 4, "ORG"), (0, 4, "学校")])

    def test_missing_surfaces_are_reported(self):
        self.assertEqual(locate("你好", [("PER", "张三")])[1], [("PER", "张三")])


class WindowTests(unittest.TestCase):
    def test_cuts_follow_sentence_ends_and_never_split_an_entity(self):
        text = "甲" * 8 + "。" + "乙" * 8 + "。" + "丙" * 8 + "。"
        entities = [{"start": 7, "end": 11, "label": "X"}]              # crosses the first sentence end
        windows = chunks(text, entities, 12)
        self.assertTrue(all(e - s <= 12 for s, e in windows))
        self.assertFalse(any(7 < cut < 11 for _, cut in windows[:-1]))
        rows, lost = window_rows("r", "s", text, [], entities, {}, 12)
        self.assertEqual(lost, 0)
        row = next(r for r in rows if r["entities"])
        e = row["entities"][0]
        self.assertEqual(row["text"][e["start"]:e["end"]], "甲。乙乙")

    def test_short_texts_are_one_window_with_their_own_id(self):
        rows, _ = window_rows("r", "s", "短句", [], [], {}, 300)
        self.assertEqual([r["id"] for r in rows], ["r"])

    def test_carve_is_stable(self):
        self.assertEqual({carve("同一句话", dev=0.5) for _ in range(3)}, {carve("同一句话", dev=0.5)})


class B2NerdLabelTests(unittest.TestCase):
    def test_readable_names_move_notes_into_descriptions(self):
        self.assertEqual(readable("组织机构名->公司"), ("公司", []))
        self.assertEqual(readable("地名->其它"), ("其它地名", []))
        self.assertEqual(readable("地名或地理政治实体(非标准)"), ("地名或地理政治实体", []))
        types = b2nerd_types(["现金或转账金额(带币种)"], {"现金或转账金额(带币种)": "涉案金额"})
        self.assertEqual(types[0]["name"], "现金或转账金额")
        self.assertEqual(types[0]["description"], "涉案金额（带币种）")

    def test_colliding_names_fall_back_to_labels_and_numbered_variants_merge(self):
        types = b2nerd_types(["人名->其它", "其它人名"], {})
        self.assertEqual([t["name"] for t in types], ["人名->其它", "其它人名"])
        labels = ["品牌名", "品牌名2", "颜色3"]
        self.assertEqual(merged_labels(labels, {"品牌名": "品牌"}), {"品牌名": "品牌名", "品牌名2": "品牌名", "颜色3": "颜色3"})

    def test_boson_markup_is_parsed_with_nesting_and_trimmed(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "Boson-NER").mkdir()
            (Path(root) / "Boson-NER/origindata.txt").write_text(
                "{{org_name:{{location:杭州}}江干区公安分局}}的{{person_name: 施宇翔}}说\n", encoding="utf-8")
            raw, = boson_records(root)
        self.assertEqual(raw.text, "杭州江干区公安分局的 施宇翔说")
        self.assertEqual(sorted((raw.text[e["start"]:e["end"]], e["label"]) for e in raw.payload),
                         [("施宇翔", "person_name"), ("杭州", "location"), ("杭州江干区公安分局", "org_name")])


class CaseTests(unittest.TestCase):
    def test_multi_label_gold_shares_the_mass(self):
        case = choice_case("c", {"文本": "x"}, "q", {"作词": "a", "作曲": "b", "无关系": "c"}, ["作曲", "作词"], {})
        example = valid(case)
        self.assertEqual(dict(zip([k for k, _ in example.options], example.target)), {"作词": 0.5, "作曲": 0.5, "无关系": 0.0})
        self.assertEqual(mark("我爱北京", 2, 4), "我爱【北京】")


class TripleTests(unittest.TestCase):
    def setUp(self):
        schema = {("图书作品", "作者"): {"人物"}, ("人物", "妻子"): {"人物"}, ("人物", "丈夫"): {"人物"}}
        self.source = TripleSource("toy", "toy", schema, {}, {"人物": "人物"})
        text = "《平凡的世界》是路遥写的，路遥的妻子是林达"
        entities = [(1, 6, {"图书作品"}), (8, 10, {"人物"}), (13, 15, {"人物"}), (19, 21, {"人物"})]
        self.raw = Raw("1", "train", text, (entities, [(0, "作者", 1), (2, "妻子", 3)]))

    def test_span_rows_ask_for_objects_of_each_subject(self):
        views = self.source.views(self.raw, random.Random(0), OPTIONS)
        rows = {row["types"][0]["name"]: row for _, row in views.spans}
        book = rows["平凡的世界的作者"]
        self.assertEqual([(self.raw.text[e["start"]:e["end"]], e["label"]) for e in book["entities"]], [("路遥", "作者")])
        person = next(row for row in rows.values() if row["types"][0]["name"].startswith("路遥"))
        self.assertEqual({t["key"] for t in person["types"]}, {"妻子", "丈夫"})
        self.assertEqual(len(views.structured["relations"]), 2)

    def test_pair_decisions_include_unrelated_pairs_and_are_valid(self):
        views = self.source.views(self.raw, random.Random(0), OPTIONS)
        golds = [valid(case).target for case in views.decisions]
        self.assertGreater(len(views.decisions), 2)
        self.assertTrue(any(case["_meta"]["negative"] for case in views.decisions))
        self.assertTrue(all(abs(sum(t) - 1) < 1e-6 for t in golds))


class FinReTests(unittest.TestCase):
    def test_candidates_hold_gold_converse_and_none(self):
        source = FinRE.__new__(FinRE)
        FinRE.__init__(source, "unused")
        criteria, gold = source.criteria("工行", "美国运通", ["被持股"], random.Random(0))
        self.assertEqual(gold, ["被持股"])
        self.assertIn("持股", criteria)
        self.assertIn("无关系", criteria)
        self.assertLessEqual(len(criteria), 16)
        self.assertEqual(criteria["被持股"], "“工行”的股份被“美国运通”持有")


class EventTests(unittest.TestCase):
    def test_rows_per_event_type_plus_an_absent_type_and_valid_decisions(self):
        source = FewFC("unused")
        text = "兴发集团将2000万股质押给浦发银行"
        raw = Raw("1", "train", text, [("质押", [(11, 13, "trigger"), (0, 4, "sub-org"), (14, 18, "obj-org")])])
        views = source.views(raw, random.Random(0), OPTIONS)
        rows = [row for _, row in views.spans]
        self.assertEqual(len(rows), 2)
        self.assertEqual([text[e["start"]:e["end"]] for e in rows[0]["entities"]], ["兴发集团", "质押", "浦发银行"])
        self.assertEqual(rows[1]["entities"], [])
        self.assertTrue(rows[1]["_meta"]["negative"])
        answers = {case["questions"]["q"]["type"]: case["gold"]["q"] for case in views.decisions}
        self.assertEqual(answers["choice"], "质押")
        for case in views.decisions:
            valid(case)


class LeakageTests(unittest.TestCase):
    def test_overlap_rules(self):
        def raw(uid, split, text):
            return Raw(uid, split, text, [])
        test = [raw("t1", "test", "这是一个测试集里的句子，长度足够长。"), raw("t2", "test", "另一条测试文本也很长很长很长。第二句同样有足够的长度了啊。")]
        train = [raw("a", "train", "这是一个测试集里的句子，长度足够长"),              # same text after normalization
                 raw("b", "train", "前面有别的内容并且足够长的一句话。这是一个测试集里的句子，长度足够长。"),  # contains t1
                 raw("c", "train", "第二句同样有足够的长度了啊。"),                        # one of its sentences is in t2
                 raw("d", "train", "完全不同的训练文本，与测试没有关系。第二句同样有足够的长度了啊。再来一句不同的话也足够长。"),
                 raw("e", "train", "毫不相干的句子，足够长足够长足够长。")]
        loaded = {"toy": ({"train": train, "test": test}, {})}
        stats = Counter()
        leakage_filter(loaded, stats)
        self.assertEqual([r.uid for r in loaded["toy"][0]["train"]], ["d", "e"])
        self.assertEqual(stats["toy:train:leak_dropped"], 3)


if __name__ == "__main__":
    unittest.main()
