import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

FREE_MAX = Decimal("0")
BUDGET_MAX = Decimal("2.50")
STANDARD_MAX = Decimal("5.00")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate tiered OpenRouter model JSON files from one allowlist."
    )
    parser.add_argument(
        "allowlist",
        help="Path to the allowlist JSON file containing all allowed model ids"
    )
    parser.add_argument(
        "output_dir",
        help="Directory where free.json, budget.json, standard.json and premium.json will be written"
    )
    return parser.parse_args()


def load_allowlist(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models", [])

    if not isinstance(models, list) or not all(isinstance(x, str) and x.strip() for x in models):
        raise ValueError(f"{path} must contain a string array at 'models'")

    return [m.strip() for m in models]


def fetch_openrouter_models(api_key: str):
    req = urllib.request.Request(
        OPENROUTER_MODELS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")

    data = json.loads(raw)
    models = data.get("data", [])

    if not isinstance(models, list):
        raise ValueError("OpenRouter response missing data array")

    return models


def as_decimal(value) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def price_per_million(pricing_value) -> Decimal:
    value = as_decimal(pricing_value)
    if value is None:
        return Decimal("0")
    return value * Decimal("1000000")


def detect_tier(completion_per_mtok_usd: Decimal) -> str:
    if completion_per_mtok_usd == FREE_MAX:
        return "free"
    if completion_per_mtok_usd <= BUDGET_MAX:
        return "budget"
    if completion_per_mtok_usd <= STANDARD_MAX:
        return "standard"
    return "premium"


def build_model_entry(model: dict) -> tuple[dict, str]:
    pricing = model.get("pricing") or {}
    supported = set(model.get("supported_parameters") or [])

    prompt_per_mtok = price_per_million(pricing.get("prompt"))
    completion_per_mtok = price_per_million(pricing.get("completion"))

    tier = detect_tier(completion_per_mtok)

    entry = {
        "id": model.get("id"),
        "name": model.get("name") or model.get("id"),
        "prompt_per_mtok_usd": float(prompt_per_mtok.quantize(Decimal("0.000001"))),
        "completion_per_mtok_usd": float(completion_per_mtok.quantize(Decimal("0.000001"))),
        "context_length": model.get("context_length"),
        "supports_tools": "tools" in supported,
        "supports_response_format": "response_format" in supported,
        "supports_reasoning": "reasoning" in supported
    }

    return entry, tier


def build_outputs(all_models, wanted_ids: list[str], allowlist_name: str) -> dict[str, dict]:
    by_id = {}
    for model in all_models:
        mid = model.get("id")
        if isinstance(mid, str) and mid.strip():
            by_id[mid] = model

    tier_models = defaultdict(list)
    missing_models = []

    for model_id in wanted_ids:
        model = by_id.get(model_id)
        if not model:
            missing_models.append(model_id)
            continue

        entry, tier = build_model_entry(model)
        tier_models[tier].append(entry)

    for tier in ("free", "budget", "standard", "premium"):
        tier_models[tier].sort(key=lambda item: (item["completion_per_mtok_usd"], item["id"]))

    outputs = {}
    for tier in ("free", "budget", "standard", "premium"):
        outputs[tier] = {
            "source": "openrouter",
            "generated_by": "github-actions",
            "allowlist": allowlist_name,
            "tier": tier,
            "tier_rules": {
                "free": "completion_per_mtok_usd == 0",
                "budget": "0 < completion_per_mtok_usd <= 2.50",
                "standard": "2.50 < completion_per_mtok_usd <= 5.00",
                "premium": "completion_per_mtok_usd > 5.00"
            },
            "count": len(tier_models[tier]),
            "missing_count": len(missing_models),
            "missing_models": missing_models,
            "models": tier_models[tier]
        }

    return outputs


def write_outputs(output_dir: Path, outputs: dict[str, dict]):
    output_dir.mkdir(parents=True, exist_ok=True)

    for tier, payload in outputs.items():
        target = output_dir / f"{tier}.json"
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
        print(f"Wrote {target} with {payload['count']} models")


def main():
    args = parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    allowlist_path = Path(args.allowlist)
    output_dir = Path(args.output_dir)

    wanted_ids = load_allowlist(allowlist_path)
    all_models = fetch_openrouter_models(api_key)
    outputs = build_outputs(all_models, wanted_ids, allowlist_path.as_posix())
    write_outputs(output_dir, outputs)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
