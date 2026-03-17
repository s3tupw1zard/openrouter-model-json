import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a hosted OpenRouter model JSON from an allowlist file."
    )
    parser.add_argument(
        "allowlist",
        help="Path to the allowlist JSON file"
    )
    parser.add_argument(
        "output",
        help="Path to the output JSON file to write"
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


def as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def build_output(all_models, wanted_ids, source_name: str):
    by_id = {}
    for model in all_models:
        mid = model.get("id")
        if isinstance(mid, str) and mid.strip():
            by_id[mid] = model

    selected = []
    missing = []

    for model_id in wanted_ids:
        model = by_id.get(model_id)
        if not model:
            missing.append(model_id)
            continue

        pricing = model.get("pricing") or {}
        supported = set(model.get("supported_parameters") or [])

        selected.append({
            "id": model_id,
            "name": model.get("name") or model_id,
            "prompt_per_mtok_usd": round((as_float(pricing.get("prompt")) or 0) * 1_000_000, 6),
            "completion_per_mtok_usd": round((as_float(pricing.get("completion")) or 0) * 1_000_000, 6),
            "context_length": model.get("context_length"),
            "supports_tools": "tools" in supported,
            "supports_response_format": "response_format" in supported,
            "supports_reasoning": "reasoning" in supported
        })

    return {
        "source": "openrouter",
        "generated_by": "github-actions",
        "allowlist": source_name,
        "count": len(selected),
        "missing_count": len(missing),
        "missing_models": missing,
        "models": selected
    }


def main():
    args = parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    allowlist_path = Path(args.allowlist)
    output_path = Path(args.output)

    wanted_ids = load_allowlist(allowlist_path)
    all_models = fetch_openrouter_models(api_key)
    output = build_output(all_models, wanted_ids, allowlist_path.as_posix())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    print(f"Wrote {output_path} with {output['count']} models from {allowlist_path}")

    if output["missing_models"]:
        print("Missing models:")
        for model_id in output["missing_models"]:
            print(f"- {model_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)