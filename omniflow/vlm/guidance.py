from __future__ import annotations

import re

DEFAULT_STEP_GUIDANCE = (
    "Prefer an installed native app over a browser or web search for phone-app "
    "tasks. Launch it with open_app using a package supplied by the runtime. If "
    "the current app is off target, use home, back, or open_app instead of "
    "continuing there. Prefer direct search and exact user-provided text over "
    "browsing long menus, history suggestions, or repeated swipes. Treat "
    "previous_action_error, recent_actions, and execution_history as authoritative: "
    "never repeat an action that already succeeded or made no observable progress. "
    "If a primary button is disabled, satisfy a visible required choice first. "
    "Never authorize payment, enter payment credentials, accept payment-app terms, "
    "enter a password or verification code, or trigger biometric authentication. "
    "When an order reaches payment confirmation, finish with a pending unpaid order "
    "without clicking a payment control."
)

ORDERING_STEP_GUIDANCE = (
    "For ordering tasks, advance through the visible forward path without reopening "
    "or resubmitting a correct search. Choose a semantically compatible product "
    "variant unless the user specified an exact flavor, ingredient, dietary, size, "
    "temperature, sugar, or other required constraint. Select required options before "
    "a disabled primary button, add the requested item once, and default quantity to "
    "one unless specified. Do not add paid extras, memberships, coupons, or unrelated "
    "recommendations. Stop before payment."
)

_ORDERING_TERMS = re.compile(
    r"点外卖|叫外卖|订外卖|点餐|订餐|下单|点一|点杯|点份|咖啡|拿铁|奶茶|"
    r"order(?: me)?|food delivery|takeaway|takeout|coffee|latte|milk tea|burger|pizza",
    re.IGNORECASE,
)


def resolve_step_guidance(goal: str, explicit: str = "") -> str:
    custom = str(explicit or "").strip()
    if custom:
        return custom
    guidance = DEFAULT_STEP_GUIDANCE
    if _ORDERING_TERMS.search(str(goal or "")):
        guidance = f"{guidance}\n\n{ORDERING_STEP_GUIDANCE}"
    return guidance


__all__ = [
    "DEFAULT_STEP_GUIDANCE",
    "ORDERING_STEP_GUIDANCE",
    "resolve_step_guidance",
]
