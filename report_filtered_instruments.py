#!/usr/bin/env python3

import sys

from extract import (
    build_filtered_payload,
    expiry_to_datetime,
    load_instruments,
    store_payload_in_memory,
)
from state import get_payload


def load_payload():
    payload = get_payload()
    if payload is not None:
        return payload

    # Fallback: build payload directly from disk to keep this script useful
    payload = build_filtered_payload(load_instruments())
    store_payload_in_memory(payload)
    return payload


def _format_contract_line(
    *,
    prefix,
    trading_symbol,
    instrument_key,
    extra=None,
):
    base = f"{prefix:<14} {trading_symbol} ({instrument_key})"
    return f"{base} {extra}" if extra else base


def summarize_index(label, entry):
    lines = [f"=== Index: {label} ==="]

    spot = entry.get("spot")
    if spot:
        lines.append(
            _format_contract_line(
                prefix="Spot",
                trading_symbol=spot.get("trading_symbol", "—"),
                instrument_key=spot.get("instrument_key", "—"),
            )
        )

    for idx, future in enumerate(entry.get("futures", []), start=1):
        expiry_ms = future.get("expiry")
        expiry_text = (
            expiry_to_datetime(expiry_ms).isoformat() if expiry_ms else "N/A"
        )
        lines.append(
            _format_contract_line(
                prefix=f"Future #{idx}",
                trading_symbol=future.get("trading_symbol", "—"),
                instrument_key=future.get("instrument_key", "—"),
                extra=f"[expiry: {expiry_text}]",
            )
        )

    options = entry.get("options", {})
    for bucket_name in ("current_expiry", "secondary_expiry"):
        bucket = options.get(bucket_name)
        if not bucket:
            continue

        weekly_flag = "weekly" if bucket.get("weekly") else "monthly"
        expiry_iso = bucket.get("expiry_iso")
        if not expiry_iso and bucket.get("expiry"):
            expiry_iso = expiry_to_datetime(bucket["expiry"]).isoformat()

        lines.append(
            f"-- Options ({bucket_name.replace('_', ' ').title()} | {weekly_flag} | expiry {expiry_iso})"
        )

        strikes = bucket.get("slice", {}).get("strikes", [])
        for strike_entry in strikes:
            strike = strike_entry.get("strike")
            strike_label = f"Strike {strike}" if strike is not None else "Strike ?"
            contracts = strike_entry.get("contracts", {})
            for opt_type in ("CE", "PE"):
                contract = contracts.get(opt_type)
                if not contract:
                    continue
                lines.append(
                    _format_contract_line(
                        prefix=f"  {strike_label} {opt_type}",
                        trading_symbol=contract.get("trading_symbol", "—"),
                        instrument_key=contract.get("instrument_key", "—"),
                    )
                )

    return lines


def build_report(payload):
    indices = payload.get("indices", {})
    snippets = []
    for label, entry in indices.items():
        snippets.extend(summarize_index(label, entry))
        snippets.append("")  # spacer
    return "\n".join(snippets).strip()


def main():
    payload = load_payload()
    report = build_report(payload)
    print(report)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI helper
        print(f"[report_filtered_instruments] Error: {exc}", file=sys.stderr)
        sys.exit(1)
