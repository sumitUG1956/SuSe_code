#!/usr/bin/env python3

"""Filter the Upstox instrument dump for selected contracts."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from state import set_payload

DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "complete.json"
REPORT_SCRIPT_PATH = Path(__file__).with_name("report_filtered_instruments.py")


def load_instruments(path=DATA_PATH):
    """Load the full instrument list from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def find_spot(instruments, *, segment, trading_symbol):
    """Locate the spot contract matching the provided identifiers."""
    for instrument in instruments:
        if (
            instrument.get("segment") == segment
            and instrument.get("trading_symbol") == trading_symbol
        ):
            return instrument
    raise LookupError(f"Spot instrument not found for {segment=} {trading_symbol=}")


def find_futures(
    instruments,
    *,
    asset_symbol,
    segment,
    count=2,
):
    """Collect the earliest expiring future contracts for an asset."""
    futures = [
        instrument
        for instrument in instruments
        if instrument.get("segment") == segment
        and instrument.get("asset_symbol") == asset_symbol
        and instrument.get("instrument_type") == "FUT"
    ]
    futures.sort(key=lambda inst: inst.get("expiry", 0))
    return futures[:count]


def collect_options(
    instruments,
    *,
    asset_symbol,
    segment,
):
    """Gather all option contracts for a given asset/segment combination."""
    return [
        instrument
        for instrument in instruments
        if instrument.get("segment") == segment
        and instrument.get("asset_symbol") == asset_symbol
        and instrument.get("instrument_type") in {"CE", "PE"}
    ]


def group_options_by_expiry(options):
    """Group option contracts by expiry timestamp."""
    grouped = {}
    for option in options:
        expiry = int(option["expiry"])
        bucket = grouped.setdefault(
            expiry,
            {"expiry": expiry, "weekly": option.get("weekly", False), "contracts": []},
        )
        bucket["contracts"].append(option)
    buckets = list(grouped.values())
    buckets.sort(key=lambda bucket: bucket["expiry"])
    return buckets


def expiry_to_datetime(expiry_ms):
    """Convert an epoch millisecond expiry to a normalized 15:30 IST datetime."""
    raw_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=timezone.utc)
    ist_dt = raw_dt.astimezone(ZoneInfo("Asia/Kolkata"))
    return ist_dt.replace(hour=15, minute=30, second=0, microsecond=0)


def select_expiries(grouped_expiries):
    """Select current and secondary expiries according to business rules."""
    if not grouped_expiries:
        return None, None

    now = datetime.now(tz=timezone.utc)
    current = None
    monthly = None

    for bucket in grouped_expiries:
        expiry_dt = expiry_to_datetime(bucket["expiry"])
        if expiry_dt >= now and current is None:
            current = bucket
        if not bucket.get("weekly", False) and expiry_dt >= now and monthly is None:
            monthly = bucket
        if current and monthly:
            break

    secondary = None
    if monthly is None:
        secondary = grouped_expiries[1] if len(grouped_expiries) > 1 else None
    elif current is None:
        secondary = monthly
    elif monthly["expiry"] != current["expiry"]:
        secondary = monthly
    else:
        monthly_dt = expiry_to_datetime(monthly["expiry"])
        for bucket in grouped_expiries:
            if bucket["expiry"] <= monthly["expiry"]:
                continue
            expiry_dt = expiry_to_datetime(bucket["expiry"])
            if (
                bucket.get("weekly", False)
                and (
                    expiry_dt.year > monthly_dt.year
                    or (expiry_dt.year == monthly_dt.year and expiry_dt.month > monthly_dt.month)
                )
            ):
                secondary = bucket
                break
        if secondary is None:
            secondary = grouped_expiries[1] if len(grouped_expiries) > 1 else None

    return current, secondary


def map_contracts_by_strike(contracts):
    """Group option contracts by strike price."""
    strikes = {}
    for contract in contracts:
        strike = float(contract.get("strike_price", 0.0))
        holders = strikes.setdefault(strike, {})
        holders[contract["instrument_type"]] = contract
    return strikes


def _select_strikes_near_spot(
    strikes,
    *,
    spot_price,
    direction,
    count,
):
    """Select strikes on one side of the spot price."""
    if direction == "itm":
        candidates = [strike for strike in strikes if strike < spot_price]
        candidates.sort(key=lambda strike: (spot_price - strike, -strike))
        if not candidates:
            candidates = list(reversed(strikes))
    else:
        candidates = [strike for strike in strikes if strike > spot_price]
        candidates.sort(key=lambda strike: (strike - spot_price, strike))
        if not candidates:
            candidates = list(strikes)
    return candidates[:count]


def summarize_option_slice(
    contracts,
    *,
    spot_price,
    strikes_each_side=10,
):
    """Summarize option strikes around the given spot price."""
    strike_map = map_contracts_by_strike(contracts)
    if not strike_map:
        return {}

    strikes = sorted(strike_map.keys())
    atm_strike = min(
        strikes,
        key=lambda strike: (abs(strike - spot_price), strike),
    )

    itm_strikes = _select_strikes_near_spot(
        strikes, spot_price=spot_price, direction="itm", count=strikes_each_side
    )
    otm_strikes = _select_strikes_near_spot(
        strikes, spot_price=spot_price, direction="otm", count=strikes_each_side
    )

    window_strikes = sorted(
        {atm_strike, *itm_strikes, *otm_strikes}
    )

    entries = []
    for strike in window_strikes:
        contracts_at_strike = strike_map.get(strike, {})
        if strike == atm_strike:
            position = "ATM"
        elif strike < spot_price:
            position = "ITM"
        else:
            position = "OTM"

        entries.append(
            {
                "strike": strike,
                "position": position,
                "contracts": contracts_at_strike,
            }
        )

    return {
        "spot_price": spot_price,
        "atm_strike": atm_strike,
        "strikes": entries,
    }


FULL_INDEX_TARGETS = [
    {
        "label": "NIFTY",
        "spot": {"segment": "NSE_INDEX", "trading_symbol": "NIFTY"},
        "futures": {"segment": "NSE_FO", "asset_symbol": "NIFTY"},
        "options": {
            "segment": "NSE_FO",
            "asset_symbol": "NIFTY",
        },
    },
    {
        "label": "BANKNIFTY",
        "spot": {"segment": "NSE_INDEX", "trading_symbol": "BANKNIFTY"},
        "futures": {"segment": "NSE_FO", "asset_symbol": "BANKNIFTY"},
        "options": {
            "segment": "NSE_FO",
            "asset_symbol": "BANKNIFTY",
        },
    },
    {
        "label": "SENSEX",
        "spot": {"segment": "BSE_INDEX", "trading_symbol": "SENSEX"},
        "futures": {"segment": "BSE_FO", "asset_symbol": "SENSEX"},
        "options": {
            "segment": "BSE_FO",
            "asset_symbol": "SENSEX",
        },
    },
]

FULL_EQUITY_SYMBOLS = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "ICICIBANK",
    "AXISBANK",
    "INFY",
    "BHARTIARTL",
    "ITC",
    "M&M",
    "SBIN",
    "LT",
]

# Toggle to keep the heavy config in code but use a slim set while prototyping.
LIMITED_MODE = False

if LIMITED_MODE:
    INDEX_TARGETS = [
        {
            "label": "NIFTY",
            "spot": {"segment": "NSE_INDEX", "trading_symbol": "NIFTY"},
            "futures": {"segment": "NSE_FO", "asset_symbol": "NIFTY", "count": 1},
            "options": None,
        },
        {
            "label": "BANKNIFTY",
            "spot": {"segment": "NSE_INDEX", "trading_symbol": "BANKNIFTY"},
            "futures": {"segment": "NSE_FO", "asset_symbol": "BANKNIFTY", "count": 1},
            "options": None,
        },
    ]
    EQUITY_SYMBOLS = []
else:
    INDEX_TARGETS = FULL_INDEX_TARGETS
    EQUITY_SYMBOLS = FULL_EQUITY_SYMBOLS

DEFAULT_SPOT_PRICES = {
    "NIFTY": 26100.0,
    "BANKNIFTY": 59100.0,
    "SENSEX": 85300.0,
}


def build_filtered_payload(
    instruments,
    *,
    spot_overrides=None,
):
    """Create the filtered payload aggregating spot, futures, and option snapshots."""
    spot_prices = {**DEFAULT_SPOT_PRICES}
    if spot_overrides:
        spot_prices.update(spot_overrides)

    results = {"indices": {}, "equities": {}}

    for target in INDEX_TARGETS:
        label = target["label"]
        spot = find_spot(instruments, **target["spot"])
        futures = find_futures(instruments, **target["futures"])

        options_summary = {}
        options_cfg = target.get("options")
        if options_cfg:
            option_contracts = collect_options(
                instruments,
                asset_symbol=options_cfg["asset_symbol"],
                segment=options_cfg["segment"],
            )
            grouped_expiries = group_options_by_expiry(option_contracts)
            current_expiry, secondary_expiry = select_expiries(grouped_expiries)

            spot_price = spot_prices.get(label)
            if current_expiry and spot_price is not None:
                options_summary["current_expiry"] = {
                    "expiry": current_expiry["expiry"],
                    "expiry_iso": expiry_to_datetime(current_expiry["expiry"]).isoformat(),
                    "weekly": current_expiry.get("weekly", False),
                    "slice": summarize_option_slice(
                        current_expiry["contracts"], spot_price=spot_price
                    ),
                }
            if secondary_expiry and spot_price is not None:
                options_summary["secondary_expiry"] = {
                    "expiry": secondary_expiry["expiry"],
                    "expiry_iso": expiry_to_datetime(secondary_expiry["expiry"]).isoformat(),
                    "weekly": secondary_expiry.get("weekly", False),
                    "slice": summarize_option_slice(
                        secondary_expiry["contracts"], spot_price=spot_price
                    ),
                }

        results["indices"][label] = {
            "spot": spot,
            "futures": futures,
            "options": options_summary,
        }

    for symbol in EQUITY_SYMBOLS:
        spot = find_spot(instruments, segment="NSE_EQ", trading_symbol=symbol)
        futures = find_futures(instruments, segment="NSE_FO", asset_symbol=symbol)
        results["equities"][symbol] = {
            "spot": spot,
            "futures": futures,
        }

    return results


def collect_trading_symbol_entries(payload):
    """Flatten all contracts into a trading_symbol catalog."""
    catalog = {}

    def add_entry(
        category,
        trading_symbol,
        instrument_key,
        **extra,
    ):
        if not trading_symbol or not instrument_key:
            return
        catalog[instrument_key] = {
            "category": category,
            "trading_symbol": trading_symbol,
            "instrument_key": instrument_key,
            **{k: v for k, v in extra.items() if v is not None},
        }

    for label, entry in payload.get("indices", {}).items():
        spot = entry.get("spot")
        if spot:
            add_entry("index_spot", spot.get("trading_symbol"), spot.get("instrument_key"), label=label)

        for future in entry.get("futures", []):
            add_entry(
                "index_future",
                future.get("trading_symbol"),
                future.get("instrument_key"),
                label=label,
                expiry=future.get("expiry"),
            )

        options = entry.get("options", {})
        for bucket_name, bucket in options.items():
            expiry = bucket.get("expiry")
            strikes = bucket.get("slice", {}).get("strikes", [])
            for strike_entry in strikes:
                strike_value = strike_entry.get("strike")
                contracts = strike_entry.get("contracts", {})
                for opt_type, contract in contracts.items():
                    add_entry(
                        "index_option",
                        contract.get("trading_symbol"),
                        contract.get("instrument_key"),
                        label=label,
                        option_type=contract.get("instrument_type", opt_type),
                        strike=strike_value,
                        expiry=expiry,
                        bucket=bucket_name,
                    )

    for symbol, entry in payload.get("equities", {}).items():
        spot = entry.get("spot")
        if spot:
            add_entry("equity_spot", spot.get("trading_symbol"), spot.get("instrument_key"), label=symbol)

        for future in entry.get("futures", []):
            add_entry(
                "equity_future",
                future.get("trading_symbol"),
                future.get("instrument_key"),
                label=symbol,
                expiry=future.get("expiry"),
            )

    return list(catalog.values())


def store_payload_in_memory(payload):
    """Persist the payload and derived catalog in the in-memory state."""
    catalog = collect_trading_symbol_entries(payload)
    set_payload(payload, catalog=catalog)
    return catalog


def refresh_payload_in_memory(
    *,
    instruments=None,
    spot_overrides=None,
):
    """Build the filtered payload and store it in the process RAM."""
    data = instruments if instruments is not None else load_instruments()
    payload = build_filtered_payload(data, spot_overrides=spot_overrides)
    store_payload_in_memory(payload)
    return payload


def run_report_script(script_path=REPORT_SCRIPT_PATH):
    """Invoke the post-processing script that prints contract references."""
    if not script_path.exists():
        print(f"[extract] Report script '{script_path}' missing; skipping.")
        return

    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[extract] Report script exited with {exc.returncode}", file=sys.stderr)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[extract] Unable to run report script: {exc}", file=sys.stderr)


def main():
    refresh_payload_in_memory()
    print("[extract] Stored filtered payload in process memory")
    run_report_script()


if __name__ == "__main__":
    main()
