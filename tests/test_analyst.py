"""The analyst may only state numbers that came back from a tool (offline path, no API key)."""
import re

from quantlab.llm import analyst, tools


def _numbers(text: str) -> set[float]:
    return {round(float(x), 1) for x in re.findall(r"-?\d+(?:\.\d+)?(?=%)", text)}


def test_tool_specs_are_well_formed():
    specs = tools.anthropic_tool_specs()
    assert len(specs) == len(tools.TOOLS) == 7
    for s in specs:
        props = s["input_schema"]["properties"]
        assert set(s["input_schema"]["required"]) <= set(props)


def test_stress_answer_quotes_only_tool_output():
    ans = analyst.offline_answer("What if the S&P drops 20% and VIX hits 50?")
    assert ans.mode == "offline"
    assert [t["tool"] for t in ans.trace] == ["stress_hedge"]
    out = ans.trace[0]["output"]
    allowed = {round(abs(v) * 100, 1) for v in (out["unhedged_pnl"], out["put_pnl"], out["hedged_pnl"], out["premium_paid_pct"])}
    allowed.add(round(out["inputs"]["strike"] / out["inputs"]["spx"] * 100, 1))
    assert _numbers(ans.text) <= allowed
    # a put can only help in a crash
    assert out["hedged_pnl"] > out["unhedged_pnl"]


def test_every_routed_question_leaves_a_trace():
    for q in ("What does a put cost today?", "How did the signals do in 2008?",
              "What is the risk premia allocation?", "What are the signals saying?"):
        assert analyst.offline_answer(q).trace, q
