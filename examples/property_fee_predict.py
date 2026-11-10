"""物业催费对话：一次请求同时判断缴费意愿、欠费项目，并抽取欠费金额。

调用统一模型服务（scripts/serve_unified.py）的 /v1/predict：
    - 缴费意愿：choice 题（肯定 / 否定 / 未知）
    - 欠费项目：默认用 choice 题（--item-mode choice）；也可以用重写过定义的抽取类型（--item-mode extract）
    - 欠费金额：抽取类型，返回原文片段和位置

为什么原来抽不出“车位管理费”：原定义“客户具体欠的是哪部分的物业费”把总称和细分项混在一起，模型给“车位管理费”
的概率只有 0.16，低于“不是实体”，所以不输出（不是只返回了第一个类型）。改法有两种：
    - choice 题：费用项目是有限集合，模型直接在候选里选，概率 0.94，最稳。
    - 抽取类型：定义里写清“物业费是总称、只抽细分项”，并把阈值提到 0.6，车位管理费 0.74、360元 0.70；
      候选列不全（项目名事先不知道）时用这种。

只用标准库，Python 3.8+ 即可运行：

    python property_fee_predict.py                                   # 跑内置的示例对话
    python property_fee_predict.py --compare                         # 对比改前、抽取类型（新定义）和选择题
    python property_fee_predict.py --item-mode extract               # 欠费项目用新定义的抽取类型
    python property_fee_predict.py --url http://xxx.xxx.xxx.xxx:8012   # 指定服务地址
    python property_fee_predict.py --state "A: 你的物业费还没交。B：多少钱？A: 360元的车位管理费。"
    python property_fee_predict.py --file dialogs.txt                # 每行一段对话，"\\n" 表示换行
    python property_fee_predict.py --raw                             # 打印完整的 JSON 响应

服务需要鉴权时，用 --token 或环境变量 DECISION_TOKEN 传入（会作为 Authorization: Bearer 发送），不要把 token 写进代码。
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# 改后：欠费项目作为选择题，欠费金额作为抽取类型
QUESTIONS = {
    "patch": {
        "type": "choice",
        "instructions": "客户是否愿意缴费。",
        "criteria": {"肯定": "客户愿意缴费", "否定": "客户不愿意缴费", "未知": "客户没有明确表示"},
    },
    "item": {
        "type": "choice",
        "instructions": "客户欠的是哪一项费用？",
        "criteria": {
            "车位管理费": "车位或停车位的管理费",
            "物业费": "物业服务费，没有说明具体项目",
            "水电费": "水费、电费",
            "电梯费": "电梯运行维护费",
            "其他": "其他费用或没有提到",
        },
    },
}
TYPES = {"欠费金额": "客户欠费多少",}

# 改后（抽取版）：欠费项目也作为抽取类型，但定义写清总称和细分项的关系；阈值要提到 0.6，
# 否则截图例子里会顺带抽出 0.50 的“物业费”
EXTRACT_QUESTIONS = {"patch": QUESTIONS["patch"]}
EXTRACT_TYPES = {
    "欠费金额": "欠款金额，带单位",
    "具体欠费项目": "物业费是总称，下面分车位管理费、电梯费、垃圾清运费等细分项；只抽客户实际欠缴的细分项名称，不抽总称",
}
EXTRACT_THRESHOLD = 0.6

# 改前：截图里的写法，欠费项目也作为抽取类型（只用于 --compare）
BEFORE_QUESTIONS = {"patch": QUESTIONS["patch"]}
BEFORE_TYPES = {"欠费金额": "客户欠费多少", "欠费项目": "客户具体欠的是哪部分的物业费"}

ITEM_LABELS = ("欠费项目", "具体欠费项目")

EXAMPLES = [
    "A: 你的物业费还没交。B：多少钱？A: 360元的车位管理费。B: 好的我待会处理。\n ",    # 截图里的例子
    "A: 你的物业费还没交。B：多少钱？A: 360元的车位管理费。\n ",                      # 截图里的例子（没有 B 的回答）
    "A: 你的物业费还没交。B：多少钱？\n ",
    "A: 您好，您家这个季度的物业费还没交，一共1280元。B：知道了，我明天去交。",
    "A: 提醒您，电梯费一共欠了215元。B：我上个月不是交过了吗？",
    "A: 您家物业费里的垃圾清运费还欠150元。B：这个月一起交吧。",
]


def predict(url, state, questions=QUESTIONS, types=TYPES, threshold=0.3, token=None, timeout=30):
    """POST one dialogue to /v1/predict; returns the parsed JSON response."""
    body = {"state": state, "questions": questions, "types": types, "threshold": threshold}
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url.rstrip("/") + "/v1/predict", json.dumps(body).encode("utf-8"), headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:      # 400 / 422: the server says what is wrong with the request
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')}") from None


def summary(response):
    """缴费意愿、欠费项目、欠费金额 as display strings; 欠费项目 comes from the choice question or the extraction."""
    answers, entities = response["answers"], response["entities"]

    def chosen(answer):
        return f"{answer['choice']} ({round(answer['probabilities'][answer['choice']], 4)})"

    def spans(labels, missing):
        return "、".join(f"{e['text']} ({e['score']})" for e in entities if e["label"] in labels) or missing

    return {
        "缴费意愿": chosen(answers["patch"]),
        "欠费项目": chosen(answers["item"]) if "item" in answers else spans(ITEM_LABELS, "未抽到"),
        "欠费金额": spans(("欠费金额",), "未提到"),
        "耗时": f"{response.get('latency_ms')} ms",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("DECISION_URL", "http://127.0.0.1:8012"),
                        help="服务地址（默认取环境变量 DECISION_URL，否则 http://127.0.0.1:8012）")
    parser.add_argument("--token", default=os.environ.get("DECISION_TOKEN"), help="Bearer token（默认取 DECISION_TOKEN）")
    parser.add_argument("--state", action="append", help="一段对话；可以重复多次")
    parser.add_argument("--file", help="对话文件，每行一段，行内的 \\n 会换成换行")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"只保留概率不低于它的实体（默认：选择题版 0.3，抽取版 {EXTRACT_THRESHOLD}）")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--item-mode", choices=("choice", "extract"), default="choice",
                        help="欠费项目用选择题（默认）还是新定义的抽取类型")
    parser.add_argument("--compare", action="store_true", help="同时跑改前、抽取版和选择题版做对比")
    parser.add_argument("--raw", action="store_true", help="打印完整的 JSON 响应")
    args = parser.parse_args()

    states = list(args.state or [])
    if args.file:
        with open(args.file, encoding="utf-8") as stream:
            states += [line.rstrip("\n").replace("\\n", "\n") for line in stream if line.strip()]
    states = states or EXAMPLES

    choice = ("选择题版（欠费项目用选择题）", QUESTIONS, TYPES, 0.3)
    extract = ("抽取版（欠费项目用新定义）", EXTRACT_QUESTIONS, EXTRACT_TYPES, EXTRACT_THRESHOLD)
    before = ("改前（欠费项目用原定义）", BEFORE_QUESTIONS, BEFORE_TYPES, 0.3)
    variants = [before, extract, choice] if args.compare else [choice if args.item_mode == "choice" else extract]

    failed = 0
    for state in states:
        print("对话:", state.strip().replace("\n", " / "))
        for label, questions, types, threshold in variants:
            threshold = threshold if args.threshold is None else args.threshold
            try:
                response = predict(args.url, state, questions, types, threshold, args.token, args.timeout)
            except (RuntimeError, OSError) as error:
                failed += 1
                print(f"  {label}: 请求失败: {error}")
                continue
            if args.raw:
                print(f"  {label}:")
                print(json.dumps(response, ensure_ascii=False, indent=2))
                continue
            result = summary(response)
            prefix = f"  {label}: " if args.compare else "  "
            print(prefix + " | ".join(f"{key} {value}" for key, value in result.items()))
        print()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
