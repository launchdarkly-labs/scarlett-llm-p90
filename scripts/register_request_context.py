"""Register the `request` context kind without calling a model provider.

LaunchDarkly creates a context kind the first time an SDK evaluates a context
of that kind. Metrics and experiments cannot use `request` until it exists.
This script evaluates bfcl-model-comparison once for a request context with
inspect_config, which reads the variation and never invokes a provider or
emits generation events. Run it once before creating metrics.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from launchdarkly_ai_server import inspect_config, shutdown

CONFIG_KEY = "bfcl-model-comparison"


async def main() -> None:
    context = {"kind": "request", "key": "bfcl-context-kind-setup"}
    try:
        inspected = await inspect_config(CONFIG_KEY, context)
    finally:
        await shutdown()
    meta = inspected.get("meta")
    if meta is None:
        raise SystemExit(
            "LaunchDarkly was unreachable or LD_SDK_KEY was rejected. "
            "No context was registered."
        )
    variation = meta.get("variationKey") or meta.get("variation_key") or "unknown"
    print(
        f"Evaluated {CONFIG_KEY} for a request context: "
        f"enabled={inspected['enabled']} variation={variation}"
    )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
