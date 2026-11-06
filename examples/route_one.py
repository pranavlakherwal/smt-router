"""Route one task two ways and print why. Needs only pydantic. Calls no model and no network.

python examples/route_one.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.engine import RoutingEngine  # noqa: E402
from router.models import Task, TaskType  # noqa: E402

QUESTION = "A shop sells pens at 3 for 2 dollars. How much do 12 pens cost?"


def main() -> int:
    engine = RoutingEngine()
    print("Capability inputs:")
    for name, what in engine.stats()["data_sources"].items():
        print(f"  {name}: {what}")

    print("\n1. The three options for maths word problems")
    for o in engine.options("maths")["options"]:
        if o["entry"]:
            print(
                f"   {o['bar']}: {o['entry']}, {o['right']} of {o['n']} then {o['fresh_right']} of {o['fresh_n']}, "
                f"${o['usd_per_1000']:.2f} per 1,000 tasks"
            )
        else:
            print(f"   {o['bar']}: none. {o['why']}")

    print("\n2. The cheapest endpoint that clears 90 on this kind of job")
    decision = engine.route(Task(query=QUESTION, kind="maths", bar=90))
    print(engine.explain(decision, max_filtered=3))

    print("\n3. The same task by the paper's score, with no bar")
    decision = engine.route(Task(query=QUESTION, task_type=TaskType.MATH))
    print(engine.explain(decision, max_filtered=0))

    print("\n4. A kind of job that was never measured is refused, not guessed")
    decision = engine.route(Task(query="Write a function that reverses a list.", task_type=TaskType.CODING, bar=90))
    print(f"   selected: {decision.selected}; {decision.reason}")
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
