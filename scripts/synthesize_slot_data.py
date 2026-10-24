"""Synthesize intent + slot training utterances with an OpenAI-compatible chat endpoint.

Each call asks for a batch of utterances of one intent, with a random phrasing style and random
hint words for diversity. Every returned item is validated: slot keys must belong to the intent,
values must be verbatim, non-overlapping substrings (decision.data.slots.align), and queries are
deduplicated and kept out of the `--exclude` test files.

    python scripts/synthesize_slot_data.py --endpoint http://localhost:8002/v1 --model Qwen3-32B \\
        --out datasets/slots/pool.jsonl --exclude examples/slot_filling/test_handwritten.jsonl
"""
import argparse
import json
import random
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from decision.data.slots import align, load_schema

STYLES = [
    "口语化、随意，像对智能音箱说话", "礼貌、完整的句子", "非常简短的命令", "带语气词或感叹，如“哎”“热死了”“好冷啊”",
    "以唤醒词开头，如“嘿管家”“小助手”“你好小智”", "先说原因或场景，再提出请求", "把对象放在动词前面的倒装语序",
    "带“请”“帮我”“麻烦”“能不能”等请求词", "疑问句形式的请求", "带有错别字以外的自然口头禅，如“那个”“就是”",
]
HINTS = {
    "device_control": ["空调", "电视", "灯", "窗帘", "热水器", "加湿器", "除湿机", "空气净化器", "扫地机器人", "洗衣机",
                       "烘干机", "冰箱", "电饭煲", "微波炉", "烤箱", "洗碗机", "电风扇", "电暖器", "地暖", "新风系统",
                       "门锁", "摄像头", "音箱", "投影仪", "台灯", "吸顶灯", "灯带", "浴霸", "排气扇", "抽油烟机",
                       "净水器", "饮水机", "晾衣架", "车库门", "窗户", "智能插座", "电动牙刷", "热水壶", "香薰机", "鱼缸灯"],
    "weather_query": ["北京", "上海", "广州", "深圳", "成都", "重庆", "西安", "南京", "苏州", "哈尔滨", "三亚", "拉萨",
                      "今天", "明天", "后天", "周末", "下周", "今晚", "明早", "这几天"],
    "music_play": ["周杰伦", "林俊杰", "邓紫棋", "陈奕迅", "王菲", "五月天", "薛之谦", "李健", "毛不易", "孙燕姿", "张学友",
                   "朴树", "许嵩", "蔡依林", "刘德华", "华晨宇"],
    "navigation": ["机场", "火车站", "医院", "学校", "公司", "商场", "公园", "博物馆", "酒店", "加油站", "地铁站", "景区"],
    "alarm_set": ["早上", "中午", "下午", "晚上", "半小时后", "每天", "周一", "明天", "开会", "吃药", "接孩子", "喝水",
                  "健身", "交水电费"],
    "chitchat": ["天气以外的闲聊", "情绪", "问助手自身", "讲故事", "感谢", "问候", "无意义的话"],
}


def prompt(schema, intent, count, rng):
    spec = schema["intents"][intent]
    style = rng.choice(STYLES)
    hints = "、".join(rng.sample(HINTS[intent], min(4, len(HINTS[intent]))))
    if not spec["slots"]:
        return (f"请生成{count}条互不重复的中文用户话语，意图是“{intent}”（{spec['description']}），"
                f"它们不应是控制设备、查天气、放音乐、导航或设闹钟的请求。风格：{style}。可参考的话题：{hints}。\n"
                '只输出 JSON 数组，每个元素形如 {"query": "...", "slots": {}}。')
    fields = "\n".join(f"- {key}（{slot['name']}）：{slot['description']}" for key, slot in spec["slots"].items())
    rules = "\n".join(f"{i}. {rule}" for i, rule in enumerate(schema["guidelines"], 1))
    variety = ""
    if intent == "device_control":
        variety = "约一半的句子带位置；约三分之一的句子是调节属性或设定数值、档位、模式；其余是开关类操作。"
    return (f"你是中文智能助手的数据标注员。请生成{count}条互不重复的用户话语，意图是“{intent}”（{spec['description']}），"
            f"并标注字段。\n可用字段：\n{fields}\n标注规范：\n{rules}\n"
            f"风格：{style}。可参考的词（不必局限于这些，鼓励换成其他说法）：{hints}。{variety}\n"
            '只输出 JSON 数组，每个元素形如 {"query": "...", "slots": {"字段名": "原文片段"}}。')


def call(endpoint, model, content, temperature, timeout=180):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 3000,
                       "temperature": temperature, "top_p": 0.95,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", body,
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]


def parse(content):
    match = re.search(r"\[.*\]", content, re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return items if isinstance(items, list) else []


def validate(item, intent, schema):
    """A clean record, or the reason it is dropped."""
    if not isinstance(item, dict) or not isinstance(item.get("query"), str) or not isinstance(item.get("slots", {}), dict):
        return None, "malformed"
    query = item["query"].strip()
    slots = {k: v.strip() if isinstance(v, str) else v for k, v in (item.get("slots") or {}).items() if v not in ("", None)}
    if not 2 <= len(query) <= 40:
        return None, "length"
    if not set(slots) <= set(schema["intents"][intent]["slots"]):
        return None, "unknown_slot"
    if align(query, slots) is None:
        return None, "not_aligned"
    return {"query": query, "intent": intent, "slots": slots}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default="http://localhost:8002/v1")
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--schema", default="examples/slot_filling/schema.json")
    parser.add_argument("--out", default="datasets/slots/pool.jsonl")
    parser.add_argument("--counts", default="device_control=1800,weather_query=360,music_play=360,navigation=360,"
                                            "alarm_set=360,chitchat=360")
    parser.add_argument("--per-call", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    schema, rng = load_schema(args.schema), random.Random(args.seed)
    targets = {k: int(v) for k, v in (x.split("=") for x in args.counts.split(","))}
    seen = {json.loads(line)["query"] for path in args.exclude for line in open(path, encoding="utf-8") if line.strip()}
    kept, stats = {intent: [] for intent in targets}, Counter()

    def job(intent):
        content = prompt(schema, intent, args.per_call, random.Random(rng.random()))
        try:
            return intent, parse(call(args.endpoint, args.model, content, args.temperature))
        except Exception as exc:  # one failed call should not end the run
            return intent, exc

    with ThreadPoolExecutor(args.workers) as pool:
        for round_ in range(args.max_rounds):
            missing = {intent: n - len(kept[intent]) for intent, n in targets.items() if len(kept[intent]) < n}
            if not missing:
                break
            # Over-ask by half: about a fifth of items are dropped or duplicated.
            jobs = [intent for intent, n in missing.items() for _ in range(-(-int(n * 1.5) // args.per_call))]
            for intent, items in pool.map(job, jobs):
                stats["calls"] += 1
                if isinstance(items, Exception):
                    stats["call_errors"] += 1
                    continue
                for item in items:
                    stats["raw"] += 1
                    record, reason = validate(item, intent, schema)
                    if record is None:
                        stats[f"drop:{reason}"] += 1
                    elif record["query"] in seen:
                        stats["drop:duplicate"] += 1
                    elif len(kept[intent]) < targets[intent]:
                        seen.add(record["query"])
                        kept[intent].append(record)
            print(json.dumps({"round": round_, "kept": {k: len(v) for k, v in kept.items()}, **stats},
                             ensure_ascii=False), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = [record for intent in targets for record in kept[intent]]
    rng.shuffle(records)
    with open(out, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"model": args.model, "counts": {k: len(v) for k, v in kept.items()}, "stats": dict(stats),
               "slot_counts": dict(Counter(k for r in records for k in r["slots"]))}
    out.with_suffix(".stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
