"""hmm / System One decision response format, shared by the server and its checks."""
import math


def build_answer(kind, options, probabilities):
    """Format one question's candidate probabilities (in `options` order) as an API answer."""
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    k = len(probabilities)
    # Distribution sharpness in [0, 1] (hmm's formula); not an empirical accuracy.
    answer = {"type": kind, "probabilities": {key: p for (key, _), p in zip(options, probabilities)},
              "confidence": (k * probabilities[best] - 1) / (k - 1)}
    if kind == "noul":
        answer["noul"] = probabilities[1]
    elif kind == "choice":
        answer["choice"] = options[best][0]
    else:
        answer.update(score=sum(i * p for i, p in enumerate(probabilities)), legend=dict(options))
    return answer


def check_response(response, payload):
    """Raise ValueError unless `response` is a valid decision response to request `payload`."""
    if response["usage"]["output_tokens"] != 0 or response["usage"]["input_tokens"] <= 0:
        raise ValueError("decision API must read inputs without generating output tokens")
    if set(response["answers"]) != set(payload["questions"]):
        raise ValueError("response question keys do not match the request")
    for name, answer in response["answers"].items():
        question = payload["questions"][name]
        kind = question["type"]
        if answer["type"] != kind:
            raise ValueError("response type mismatch")
        expected = ({"false", "true"} if kind == "noul" else
                    set(question["criteria"]) if kind == "choice" else
                    {str(i) for i in range(len(question["criteria"]))})
        probabilities = answer["probabilities"]
        if set(probabilities) != expected:
            raise ValueError("response candidates do not match request candidates")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("invalid probability")
        if abs(sum(probabilities.values()) - 1) > 1e-5:
            raise ValueError("probabilities are not normalized")
        if not -1e-6 <= answer["confidence"] <= 1 + 1e-6:
            raise ValueError("invalid confidence")
        if kind == "noul" and abs(answer["noul"] - probabilities["true"]) > 1e-6:
            raise ValueError("noul probability mismatch")
        if kind == "choice" and answer["choice"] != max(probabilities, key=probabilities.__getitem__):
            raise ValueError("choice does not maximize probability")
        if kind == "score":
            expected_score = sum(int(k) * p for k, p in probabilities.items())
            if abs(answer["score"] - expected_score) > 1e-5:
                raise ValueError("score is not the expected level")
            if answer["legend"] != {str(i): v for i, v in enumerate(question["criteria"])}:
                raise ValueError("score legend mismatch")
