#!/usr/bin/env python3

"""
Report Filtered Instruments Module - Generates Human-Readable Summary
Creates formatted console output showing filtered instruments (indices, futures, options)
"""

import sys  # For system functions (exit codes)

# Import extraction functions
from extract import (
    build_filtered_payload,  # Build filtered payload from instruments
    expiry_to_datetime,  # Convert expiry timestamp to datetime
    load_instruments,  # Load complete instruments list from disk
    store_payload_in_memory,  # Store payload in memory state
)
from state import get_payload  # Get payload from memory


def load_payload():
    """
    Load filtered payload from memory or build if not available
    
    Returns:
        dict: Filtered payload containing indices and equities
    
    Process:
        1. Try to get payload from memory (fast path)
        2. If not in memory, build from disk instruments (fallback)
        3. Store built payload in memory for future use
        4. Return payload
    
    Why fallback needed: Script can run standalone without server
    """
    payload = get_payload()  # Try to get from memory
    if payload is not None:  # If found in memory
        return payload  # Return directly

    # Fallback: Build fresh payload from disk
    payload = build_filtered_payload(load_instruments())
    store_payload_in_memory(payload)  # Store for future use
    return payload


def _format_contract_line(
    *,
    prefix,
    trading_symbol,
    instrument_key,
    extra=None,
):
    """
    Format a single contract line for display
    
    Args:
        prefix: Label (e.g., "Spot", "Future #1", "Strike 24000 CE")
        trading_symbol: Human-readable symbol (e.g., "NIFTY")
        instrument_key: Upstox API key (e.g., "NSE_INDEX|Nifty 50")
        extra: Optional extra info (e.g., "[expiry: 2025-01-30...]")
    
    Returns:
        str: Formatted line
    
    Format: <prefix (14 chars left-aligned)> <symbol> (<key>) [extra]
    """
    base = f"{prefix:<14} {trading_symbol} ({instrument_key})"  # Format base
    return f"{base} {extra}" if extra else base  # Add extra if provided


def summarize_index(label, entry):
    """
    Generate summary lines for a single index
    
    Args:
        label: Index name (e.g., "NIFTY", "BANKNIFTY")
        entry: Index data dict containing spot, futures, options
    
    Returns:
        list: Lines of formatted text
    
    Process:
        1. Add header line
        2. Add spot contract
        3. Add all futures contracts with expiry
        4. Add options by expiry (current and secondary)
        5. For each strike, add CE and PE contracts
    """
    lines = [f"=== Index: {label} ==="]  # Header

    # Add spot contract
    spot = entry.get("spot")
    if spot:
        lines.append(
            _format_contract_line(
                prefix="Spot",
                trading_symbol=spot.get("trading_symbol", "—"),
                instrument_key=spot.get("instrument_key", "—"),
            )
        )

    # Add futures contracts
    for idx, future in enumerate(entry.get("futures", []), start=1):
        expiry_ms = future.get("expiry")  # Get expiry timestamp
        # Convert to ISO format string
        expiry_text = (
            expiry_to_datetime(expiry_ms).isoformat() if expiry_ms else "N/A"
        )
        lines.append(
            _format_contract_line(
                prefix=f"Future #{idx}",  # Numbered label
                trading_symbol=future.get("trading_symbol", "—"),
                instrument_key=future.get("instrument_key", "—"),
                extra=f"[expiry: {expiry_text}]",  # Show expiry
            )
        )

    # Add options (current and secondary expiry)
    options = entry.get("options", {})
    for bucket_name in ("current_expiry", "secondary_expiry"):
        bucket = options.get(bucket_name)
        if not bucket:  # If this expiry bucket doesn't exist
            continue  # Skip to next

        # Get expiry metadata
        weekly_flag = "weekly" if bucket.get("weekly") else "monthly"
        expiry_iso = bucket.get("expiry_iso")
        if not expiry_iso and bucket.get("expiry"):
            expiry_iso = expiry_to_datetime(bucket["expiry"]).isoformat()

        # Add options section header
        lines.append(
            f"-- Options ({bucket_name.replace('_', ' ').title()} | {weekly_flag} | expiry {expiry_iso})"
        )

        # Add strikes and their CE/PE contracts
        strikes = bucket.get("slice", {}).get("strikes", [])
        for strike_entry in strikes:
            strike = strike_entry.get("strike")  # Strike price
            strike_label = f"Strike {strike}" if strike is not None else "Strike ?"
            contracts = strike_entry.get("contracts", {})  # CE and PE dict
            # Add CE and PE for this strike
            for opt_type in ("CE", "PE"):
                contract = contracts.get(opt_type)
                if not contract:  # If contract missing
                    continue  # Skip
                lines.append(
                    _format_contract_line(
                        prefix=f"  {strike_label} {opt_type}",  # Indented
                        trading_symbol=contract.get("trading_symbol", "—"),
                        instrument_key=contract.get("instrument_key", "—"),
                    )
                )

    return lines  # Return all formatted lines


def build_report(payload):
    """
    Build complete report from payload
    
    Args:
        payload: Complete filtered payload dict
    
    Returns:
        str: Formatted report text (newline separated)
    
    Process:
        1. Extract all indices from payload
        2. Generate summary for each index
        3. Join all sections with newlines
        4. Return complete report string
    """
    indices = payload.get("indices", {})  # Get indices dict
    snippets = []  # Collect all text snippets
    for label, entry in indices.items():  # Iterate over indices
        snippets.extend(summarize_index(label, entry))  # Add summary lines
        snippets.append("")  # Add blank line separator
    return "\n".join(snippets).strip()  # Join with newlines and strip trailing


def main():
    """
    CLI entry point - runs when executed directly
    
    Process:
        1. Load payload
        2. Build report
        3. Print to console
    
    Usage:
        python report_filtered_instruments.py
    """
    payload = load_payload()  # Load payload
    report = build_report(payload)  # Build formatted report
    print(report)  # Print to console


# Run main if executed directly (not imported)
if __name__ == "__main__":
    try:
        main()  # Execute main function
    except Exception as exc:  # pragma: no cover
        # If error occurs, print to stderr and exit with error code
        print(f"[report_filtered_instruments] Error: {exc}", file=sys.stderr)
        sys.exit(1)  # Exit with failure code
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
