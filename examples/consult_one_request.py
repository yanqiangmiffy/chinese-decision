"""物业咨询来电：一个请求里同时抽取“客户称呼”和“咨询业务”（两个抽取类型放在一起），并判断客户态度和来电意图。

两种用法：
    python consult_one_request.py                    # 用找到的最佳定义，每段对话只发一个 /v1/predict 请求
    python consult_one_request.py --search           # 在标注好的对话上试遍候选定义，找出两个类型能一起抽对的组合

自定义抽取类型放在一起时会互相影响（加一个类型会改变模型读整段文本的方式），所以定义要一起调：
--search 会把 NAME_CANDIDATES × BUSINESS_CANDIDATES 两种顺序都试一遍，按“两个都抽对”的对话数排名。
找到更好的组合后，改 BEST_TYPES 即可。

在 4B 上搜索的结论（9 段标注对话，120 种组合）：
    - 只靠改定义，最好的组合（即 BEST_TYPES）两个都抽对 4/9。“咨询业务”一直很稳，单字的姓（熊、王、李）
      一加上业务类型，概率就从 0.75 掉到 0.2–0.45，落在“不是实体”那一边；张女士这种多字称呼不受影响。
    - 同一组定义，把解码时“不是实体”的偏置从 ×5（模型默认）降到 ×2，两个都抽对 8/9，没有新增误报
      （熊熊例子：熊 0.62、车位管理费 0.72），只漏了“小吴”这种昵称。这个偏置目前只能在模型配置里改，请求里还不能传。

其他参数：
    --url http://xxx.xxx.xxx.xxx:8012                # 服务地址（默认取环境变量 DECISION_URL）
    --state "A: 欢迎咨询，请问您贵姓。B：姓熊？……"   # 测自己的对话，可重复
    --file dialogs.txt                               # 每行一段对话，"\\n" 表示换行
    --raw                                            # 打印完整的 JSON 响应
    --top 15                                         # --search 时显示前几名

只用标准库，Python 3.8+ 即可运行。服务需要鉴权时，用 --token 或环境变量 DECISION_TOKEN 传入，不要把 token 写进代码。
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

QUESTIONS = {
    "patch": {
        "type": "choice",
        "instructions": "客户的态度是？",
        "criteria": {"肯定": "客户态度平和或配合", "否定": "客户不配合或拒绝", "投诉": "客户在投诉或表达不满"},
    },
    "intent": {
        "type": "choice",
        "instructions": "客户这次来电是为了什么？",
        "criteria": {
            "咨询": "询问价格、收费标准、规定或办理方式等信息",
            "办理": "要办理某项手续或业务，如补办、备案、缴费",
            "投诉": "投诉或表达不满，如多扣费、服务差",
            "报修": "报告故障，要求维修",
        },
    },
}

# 候选定义：(类型名, 说明)。类型名也会影响模型，所以名字不同的候选也算不同定义；输出时统一显示为“客户称呼”“咨询业务”
NAME_CANDIDATES = [
    ("客户称呼", "客户称呼、姓氏等信息。"),
    ("客户称呼", "客户的姓氏或称呼，如“姓熊”中的“熊”、王先生、张女士"),
    ("客户称呼", "客户的姓或名字（常常只有一个字），如熊、王、李、陈、张女士"),
    ("客户称呼", "客户的姓氏或称呼（人名），如“姓熊”中的“熊”、王先生、张女士；不是业务或费用"),
    ("客户称呼", "来电客户报的姓或姓名，通常在“贵姓”“怎么称呼”之后的回答里，如熊、王、张女士"),
    ("客户称呼", "对话中客户自称的姓，只有一个字时也要抽，如熊、王；或带“先生”“女士”的称呼，如张女士"),
    ("客户称呼", "人名或姓氏，如“姓熊”的“熊”、“我姓王”的“王”、张女士"),
    ("客户姓氏", "客户自报的姓，如“我姓王”中的“王”、“免贵姓李”中的“李”、“姓熊”中的“熊”"),
    ("客户姓氏", "客户的姓，如熊、王、李、张、刘、陈"),
    ("人名", "人物的姓名，包括只说了姓的情况，如“姓熊”中的“熊”、张女士"),
    ("姓名", "人的姓名或姓氏，如熊、王、张女士"),
    ("称呼", "对客户的称呼，可以只有姓，如熊、王、张女士、李先生"),
    ("联系人", "联系人的姓或姓名，如熊、王、张女士"),
]
BUSINESS_CANDIDATES = [
    ("咨询业务", "客户问到的物业收费项目或服务的名称（名词），如车位管理费、电梯费、停车费、水电费、装修备案、门禁卡；只抽名称，不抽“多少钱”“怎么交”等问法"),
    ("咨询业务", "客户咨询的物业收费项目或服务（名词），如车位管理费、电梯费、装修备案、门禁卡；不包括“多少钱”“怎么交”等问法"),
    ("咨询事项", "客户问到的具体业务或费用名称，如车位管理费、电梯费、装修备案；只抽名称，不抽整句问题"),
    ("咨询业务", "客户问到的物业费用或服务名称，如车位管理费、电梯费、停车费、装修备案；不是人名或姓氏"),
    ("咨询业务", "费用或业务的名称，如车位管理费、电梯费、停车费、水电费、装修备案、门禁卡"),
]

# 最佳组合（--search 的第 1 名，按这个顺序放进请求）
BEST_TYPES = [
    ("客户称呼", "人名的一种：客户的姓氏或称呼，如“姓熊”中的“熊”、王先生、张女士，单独一个字的姓也算"),
    ("咨询事项", "客户问到的具体业务或费用名称，如车位管理费、电梯费、装修备案；只抽名称，不抽整句问题"),
]

# 标注好的对话：(对话, 客户称呼, 咨询业务)，--search 用；抽到的片段包含答案、且最多多 2 个字就算对
LABELED = [
    ("A: 欢迎咨询，请问您贵姓。B：姓熊？A: 您要咨询什么业务。B: 车位管理费一个月多钱？ A: 50元一个月一车。\n ", "熊", "车位管理费"),
    ("A: 欢迎咨询，请问您贵姓。B：姓熊？A: 您要咨询什么业务。B: 车位管理费？ A: 50元一个月一车。\n ", "熊", "车位管理费"),
    ("A: 您好，请问怎么称呼？B：我姓王。A: 请问您要办理什么业务？B: 我想问一下电梯费怎么交。", "王", "电梯费"),
    ("A: 欢迎来电，您贵姓？B：免贵姓李。A: 有什么可以帮您？B: 我家的水电费好像多扣了。", "李", "水电费"),
    ("A: 请问您是？B：我是3栋的张女士。A: 您要咨询什么？B: 停车费能不能按季度交？", "张女士", "停车费"),
    ("A: 您好。B：你好，我想办理装修备案。A: 好的，请问您贵姓？B：姓刘。", "刘", "装修备案"),
    ("A: 您好，请问贵姓？B：姓陈。A: 请问有什么事？B: 门禁卡丢了想补办一张。", "陈", "门禁卡"),
    ("A: 您好，这里是物业客服，请问您贵姓？B：我姓周。A: 周先生您好，请问咨询什么？B: 垃圾清运费是怎么收的？", "周", "垃圾清运费"),
    ("A: 请问怎么称呼您？B：叫我小吴就行。A: 好的，有什么可以帮您？B: 电动车充电费多少钱一次？", "小吴", "充电费"),
]
EXAMPLES = [state for state, _, _ in LABELED]


def post(url, path, body, token=None, timeout=30):
    """POST a JSON body to the service; returns the parsed JSON response."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode("utf-8"), headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:      # 400 / 422: the server says what is wrong with the request
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')}") from None


def predict(url, state, types=BEST_TYPES, threshold=0.3, token=None, timeout=30):
    """一个 /v1/predict 请求：两个 choice 题 + 两个抽取类型放在一起。"""
    return post(url, "/v1/predict", {"state": state, "questions": QUESTIONS, "types": dict(types),
                                     "threshold": threshold}, token, timeout)


def matches(text, expected):
    return expected in text and len(text) <= len(expected) + 2


def search(url, threshold, token, timeout, top):
    """Try every name x business candidate in both orders on LABELED; print the best combinations."""
    combos = [(n, b, order) for n in NAME_CANDIDATES for b in BUSINESS_CANDIDATES for order in ("称呼在前", "业务在前")]
    print(f"{len(combos)} 种组合 × {len(LABELED)} 段对话，共 {len(combos) * len(LABELED)} 个请求")
    results, started = [], time.time()
    for done, (name, business, order) in enumerate(combos, 1):
        types = [name, business] if order == "称呼在前" else [business, name]
        both = name_ok = business_ok = wrong = 0
        lowest = 1.0
        first = ""
        for k, (state, expected_name, expected_business) in enumerate(LABELED):
            entities = post(url, "/v1/extract", {"text": state, "types": dict(types), "threshold": threshold},
                            token, timeout)["entities"]
            names = [e for e in entities if e["label"] == name[0]]
            businesses = [e for e in entities if e["label"] == business[0]]
            hit_name = [e for e in names if matches(e["text"], expected_name)]
            hit_business = [e for e in businesses if matches(e["text"], expected_business)]
            name_ok += bool(hit_name)
            business_ok += bool(hit_business)
            both += bool(hit_name and hit_business)
            wrong += len(names) - len(hit_name) + len(businesses) - len(hit_business)
            if hit_name and hit_business:
                lowest = min(lowest, hit_name[0]["score"], hit_business[0]["score"])
            if k == 0:
                first = ", ".join(f"{e['text']}({e['score']:.2f})" for e in entities)
        results.append((both, name_ok + business_ok, -wrong, lowest, name, business, order, name_ok, business_ok, wrong, first))
        if done % 20 == 0:
            print(f"  已完成 {done}/{len(combos)}（{time.time() - started:.0f} s）", file=sys.stderr)
    results.sort(key=lambda r: r[:4], reverse=True)
    n = len(LABELED)
    print(f"\n前 {top} 名（按“两个都抽对”的对话数排序；最低分 = 都抽对的对话里较低的那个分数）")
    for rank, (both, _, _, lowest, name, business, order, name_ok, business_ok, wrong, first) in enumerate(results[:top], 1):
        print(f"{rank:2d}. 都对 {both}/{n}  称呼 {name_ok}/{n}  业务 {business_ok}/{n}  误报 {wrong}  最低分 {lowest:.2f}  {order}")
        print(f"    {name[0]}：{name[1]}")
        print(f"    {business[0]}：{business[1]}")
        print(f"    熊熊例子: {first or '什么都没抽到'}")


def summary(response, types=BEST_TYPES):
    """客户态度、来电意图、客户称呼、咨询业务 as display strings (labels mapped back to 客户称呼 / 咨询业务)."""
    answers, entities = response["answers"], response["entities"]
    name_label, business_label = types[0][0], types[1][0]
    if business_label in (n for n, _ in NAME_CANDIDATES):      # BEST_TYPES may list the business type first
        name_label, business_label = business_label, name_label

    def chosen(answer):
        return f"{answer['choice']} ({round(answer['probabilities'][answer['choice']], 4)})"

    def spans(label):
        return "、".join(f"{e['text']} ({e['score']})" for e in entities if e["label"] == label) or "未抽到"

    return {
        "客户态度": chosen(answers["patch"]),
        "来电意图": chosen(answers["intent"]),
        "客户称呼": spans(name_label),
        "咨询业务": spans(business_label),
        "耗时": f"{response.get('latency_ms')} ms",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("DECISION_URL", "http://127.0.0.1:8012"),
                        help="服务地址（默认取环境变量 DECISION_URL，否则 http://127.0.0.1:8012）")
    parser.add_argument("--token", default=os.environ.get("DECISION_TOKEN"), help="Bearer token（默认取 DECISION_TOKEN）")
    parser.add_argument("--state", action="append", help="一段对话；可以重复多次")
    parser.add_argument("--file", help="对话文件，每行一段，行内的 \\n 会换成换行")
    parser.add_argument("--threshold", type=float, default=0.3, help="只保留概率不低于它的实体（默认 0.3）")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--search", action="store_true", help="在标注好的对话上搜索定义组合")
    parser.add_argument("--top", type=int, default=10, help="--search 时显示前几名（默认 10）")
    parser.add_argument("--raw", action="store_true", help="打印完整的 JSON 响应")
    args = parser.parse_args()

    if args.search:
        search(args.url, args.threshold, args.token, args.timeout, args.top)
        return

    states = list(args.state or [])
    if args.file:
        with open(args.file, encoding="utf-8") as stream:
            states += [line.rstrip("\n").replace("\\n", "\n") for line in stream if line.strip()]
    states = states or EXAMPLES

    print("抽取类型（一个请求）:", json.dumps(dict(BEST_TYPES), ensure_ascii=False))
    failed = 0
    for state in states:
        print("对话:", state.strip().replace("\n", " / "))
        try:
            response = predict(args.url, state, BEST_TYPES, args.threshold, args.token, args.timeout)
        except (RuntimeError, OSError) as error:
            failed += 1
            print("  请求失败:", error)
            continue
        if args.raw:
            print(json.dumps(response, ensure_ascii=False, indent=2))
        else:
            print("  " + " | ".join(f"{key} {value}" for key, value in summary(response).items()))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
