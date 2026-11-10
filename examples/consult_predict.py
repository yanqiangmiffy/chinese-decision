"""物业咨询来电：判断客户态度和来电意图，抽取客户称呼和咨询业务。

调用统一模型服务（scripts/serve_unified.py），每段对话发两个请求：
    1. /v1/predict：两个 choice 题（客户态度、来电意图）+ 抽取“客户称呼”
    2. /v1/extract：抽取“咨询业务”

为什么分两个请求：两个抽取类型放在同一个请求里会互相挤压（称呼抽对了业务就丢，反之亦然），分开后各自单独抽取，
称呼 7/7、业务 6/7（原写法一个请求两个类型：1/7、0/7）。原定义“客户咨询的什么业务”太模糊，新定义写明只抽名词。

只用标准库，Python 3.8+ 即可运行：

    python consult_predict.py                                   # 跑内置的示例对话
    python consult_predict.py --compare                         # 对比改前（原定义、一个请求）和改后
    python consult_predict.py --url http://xxx.xxx.xxx.xxx:8012   # 指定服务地址
    python consult_predict.py --state "A: 欢迎咨询，请问您贵姓。B：姓熊？A: 您要咨询什么业务。B: 车位管理费一个月多钱？"
    python consult_predict.py --file dialogs.txt                # 每行一段对话，"\\n" 表示换行
    python consult_predict.py --raw                             # 打印完整的 JSON 响应

服务需要鉴权时，用 --token 或环境变量 DECISION_TOKEN 传入（会作为 Authorization: Bearer 发送），不要把 token 写进代码。
"""
import argparse
import json
import os
import sys
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
# 改后：两个抽取类型分开请求
NAME_TYPES = {"客户称呼": "客户的姓氏或称呼，如“姓熊”中的“熊”、王先生、张女士"}
BUSINESS_TYPES = {"咨询业务": "客户问到的物业收费项目或服务的名称（名词），如车位管理费、电梯费、停车费、水电费、装修备案、门禁卡；"
                          "只抽名称，不抽“多少钱”“怎么交”等问法"}
# 改前：截图里的写法，两个类型放在同一个请求里（只用于 --compare）
BEFORE_TYPES = {"客户称呼": "客户称呼、姓氏等信息。", "咨询业务": "客户咨询的什么业务"}

EXAMPLES = [
    "A: 欢迎咨询，请问您贵姓。B：姓熊？A: 您要咨询什么业务。B: 车位管理费一个月多钱？ A: 50元一个月一车。\n ",  # 截图里的例子
    "A: 欢迎咨询，请问您贵姓。B：姓熊？A: 您要咨询什么业务。B: 车位管理费？ A: 50元一个月一车。\n ",            # 截图里的例子
    "A: 您好，请问怎么称呼？B：我姓王。A: 请问您要办理什么业务？B: 我想问一下电梯费怎么交。",
    "A: 欢迎来电，您贵姓？B：免贵姓李。A: 有什么可以帮您？B: 我家的水电费好像多扣了。",
    "A: 请问您是？B：我是3栋的张女士。A: 您要咨询什么？B: 停车费能不能按季度交？",
    "A: 您好，请问贵姓？B：姓陈。A: 请问有什么事？B: 门禁卡丢了想补办一张。",
]


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


def predict(url, state, threshold=0.3, token=None, timeout=30):
    """改后：choice 题和客户称呼一个请求，咨询业务单独一个请求；合并成一个响应。"""
    first = post(url, "/v1/predict", {"state": state, "questions": QUESTIONS, "types": NAME_TYPES,
                                      "threshold": threshold}, token, timeout)
    second = post(url, "/v1/extract", {"text": state, "types": BUSINESS_TYPES, "threshold": threshold}, token, timeout)
    return dict(first, entities=sorted(first["entities"] + second["entities"], key=lambda e: e["start"]),
                latency_ms=round(first.get("latency_ms", 0) + second.get("latency_ms", 0), 1))


def predict_before(url, state, threshold=0.3, token=None, timeout=30):
    """改前：原定义，两个抽取类型放在同一个请求里。"""
    return post(url, "/v1/predict", {"state": state, "questions": QUESTIONS, "types": BEFORE_TYPES,
                                     "threshold": threshold}, token, timeout)


def summary(response):
    """客户态度、来电意图、客户称呼、咨询业务 as display strings."""
    answers, entities = response["answers"], response["entities"]

    def chosen(answer):
        return f"{answer['choice']} ({round(answer['probabilities'][answer['choice']], 4)})"

    def spans(label):
        return "、".join(f"{e['text']} ({e['score']})" for e in entities if e["label"] == label) or "未抽到"

    return {
        "客户态度": chosen(answers["patch"]),
        "来电意图": chosen(answers["intent"]),
        "客户称呼": spans("客户称呼"),
        "咨询业务": spans("咨询业务"),
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
    parser.add_argument("--compare", action="store_true", help="同时跑改前的请求（原定义、一个请求两个类型）做对比")
    parser.add_argument("--raw", action="store_true", help="打印完整的 JSON 响应")
    args = parser.parse_args()

    states = list(args.state or [])
    if args.file:
        with open(args.file, encoding="utf-8") as stream:
            states += [line.rstrip("\n").replace("\\n", "\n") for line in stream if line.strip()]
    states = states or EXAMPLES

    variants = [("改后（新定义，分两个请求）", predict)]
    if args.compare:
        variants.insert(0, ("改前（原定义，一个请求）", predict_before))

    failed = 0
    for state in states:
        print("对话:", state.strip().replace("\n", " / "))
        for label, call in variants:
            try:
                response = call(args.url, state, args.threshold, args.token, args.timeout)
            except (RuntimeError, OSError) as error:
                failed += 1
                print(f"  {label}: 请求失败: {error}")
                continue
            if args.raw:
                print(f"  {label}:")
                print(json.dumps(response, ensure_ascii=False, indent=2))
                continue
            prefix = f"  {label}: " if args.compare else "  "
            print(prefix + " | ".join(f"{key} {value}" for key, value in summary(response).items()))
        print()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
