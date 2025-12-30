#!/usr/bin/env python3
# Shebang line - Python 3 से script run करने के लिए

"""
REPORT FILTERED INSTRUMENTS MODULE (फ़िल्टर्ड इंस्ट्रूमेंट्स रिपोर्ट मॉड्यूल)
============================================================================

Purpose: Filtered instruments का human-readable summary report generate करना
यह script memory में stored payload को read करके instruments की formatted list print करता है

Report Format:
- Indices (NIFTY, BANKNIFTY, SENSEX)
  - Spot contracts
  - Future contracts (with expiry)
  - Option contracts (current और secondary expiry, strikes by position - ITM/ATM/OTM)

Usage:
    $ python report_filtered_instruments.py
    
    Output:
    === Index: NIFTY ===
    Spot           NIFTY (NSE_INDEX|Nifty 50)
    Future #1      NIFTY25JANFUT (NSE_FO|NIFTY25JANFUT) [expiry: 2025-01-30T15:30:00+05:30]
    -- Options (Current Expiry | weekly | expiry 2025-01-02T15:30:00+05:30)
      Strike 24000 CE  NIFTY 24000 CE 02 JAN 25 (NSE_FO|...)
      Strike 24000 PE  NIFTY 24000 PE 02 JAN 25 (NSE_FO|...)
      ...
"""

import sys  # System functions (exit codes) के लिए

# extract module से required functions import करो
from extract import (
    build_filtered_payload,  # Instruments से payload build करने के लिए
    expiry_to_datetime,  # Expiry milliseconds को readable datetime में convert करने के लिए
    load_instruments,  # Disk से complete instruments list load करने के लिए
    store_payload_in_memory,  # Payload को memory में store करने के लिए
)
from state import get_payload  # Memory से current payload get करने के लिए


def load_payload():
    """
    Memory से filtered payload load करो (या fallback में disk से build करो)
    
    Purpose: Report generation के लिए instruments data provide करना
    
    Returns:
        dict: Filtered payload containing indices और equities
              Structure: {"indices": {...}, "equities": {...}}
    
    Process:
        1. पहले memory (in-process state) से payload get करने की कोशिश करो
        2. अगर memory में नहीं है, तो disk से instruments load करके payload build करो
        3. Built payload को memory में store करो (future use के लिए)
        4. Payload return करो
    
    Why fallback needed: यह script standalone भी run हो सकती है (without server)
    
    Example:
        payload = load_payload()
        # payload = {"indices": {"NIFTY": {...}, "BANKNIFTY": {...}}, "equities": {...}}
    """
    payload = get_payload()  # Memory (global _STATE) से payload get करने की कोशिश करो
    if payload is not None:  # अगर payload memory में मिल गया
        return payload  # Directly return करो (fast path)

    # Fallback: Memory में नहीं मिला, तो fresh build करो
    # यह useful है जब script standalone run हो रही है (server के बिना)
    payload = build_filtered_payload(load_instruments())  # Disk से instruments load करके filtered payload build करो
    store_payload_in_memory(payload)  # Built payload को memory में store करो (ताकि दोबारा build न करना पड़े)
    return payload  # Payload return करो


def _format_contract_line(
    *,
    prefix,
    trading_symbol,
    instrument_key,
    extra=None,
):
    """
    Single contract के लिए formatted line string बनाओ (pretty printing के लिए)
    
    Purpose: Consistent formatting के साथ instrument details को display करना
    
    Args:
        prefix: Left-aligned label (e.g., "Spot", "Future #1", "Strike 24000 CE")
        trading_symbol: Human-readable trading symbol (e.g., "NIFTY", "NIFTY25JANFUT")
        instrument_key: Unique key for Upstox API (e.g., "NSE_INDEX|Nifty 50")
        extra: Optional extra information (e.g., "[expiry: 2025-01-30T15:30:00+05:30]")
    
    Returns:
        str: Formatted line
             Example: "Spot           NIFTY (NSE_INDEX|Nifty 50)"
                      "Future #1      NIFTY25JANFUT (NSE_FO|NIFTY25JANFUT) [expiry: ...]"
    
    Format:
        <prefix (left-aligned, 14 chars)> <trading_symbol> (<instrument_key>) [extra]
    """
    # Base string बनाओ: prefix को 14 characters में left-align करके trading_symbol और instrument_key add करो
    base = f"{prefix:<14} {trading_symbol} ({instrument_key})"  # <14 means left-align in 14 char width
    return f"{base} {extra}" if extra else base  # अगर extra info है तो add करो, नहीं तो base ही return करो


def summarize_index(label, entry):
    """
    Single index (जैसे NIFTY, BANKNIFTY) के लिए detailed summary lines बनाओ
    
    Purpose: Index की सभी contracts (spot, futures, options) को formatted report में convert करना
    
    Args:
        label: Index name (e.g., "NIFTY", "BANKNIFTY", "SENSEX")
        entry: Index data dict containing "spot", "futures", "options" keys
               Structure: {
                   "spot": {...},
                   "futures": [{...}, {...}],
                   "options": {
                       "current_expiry": {...},
                       "secondary_expiry": {...}
                   }
               }
    
    Returns:
        list[str]: Lines of formatted text (strings), console पर print करने के लिए
    
    Process:
        1. Index header line add करो (=== Index: NIFTY ===)
        2. Spot contract की details add करो
        3. सभी future contracts की details add करो (with expiry)
        4. Options section process करो:
           - Current और secondary expiry दोनों के लिए
           - हर strike पर CE और PE contracts list करो
           - Position indicator add करो (ITM/ATM/OTM)
    
    Example output:
        === Index: NIFTY ===
        Spot           NIFTY (NSE_INDEX|Nifty 50)
        Future #1      NIFTY25JANFUT (NSE_FO|...) [expiry: 2025-01-30T15:30:00+05:30]
        -- Options (Current Expiry | weekly | expiry 2025-01-02T15:30:00+05:30)
          Strike 24000 CE  NIFTY 24000 CE 02 JAN 25 (NSE_FO|...)
          Strike 24000 PE  NIFTY 24000 PE 02 JAN 25 (NSE_FO|...)
    """
    lines = [f"=== Index: {label} ==="]  # Header line बनाओ (index name के साथ)

    # SPOT CONTRACT SECTION
    spot = entry.get("spot")  # Spot data get करो (e.g., NIFTY spot, BANKNIFTY spot)
    if spot:  # अगर spot data exist करता है
        lines.append(  # Formatted spot line add करो
            _format_contract_line(
                prefix="Spot",  # Label: "Spot"
                trading_symbol=spot.get("trading_symbol", "—"),  # Trading symbol (fallback: "—")
                instrument_key=spot.get("instrument_key", "—"),  # Instrument key (fallback: "—")
            )
        )

    # FUTURES SECTION
    # सभी futures contracts को enumerate करके process करो (numbering के साथ)
    for idx, future in enumerate(entry.get("futures", []), start=1):  # idx = 1, 2, 3, ...
        expiry_ms = future.get("expiry")  # Expiry timestamp (milliseconds) get करो
        # Expiry को human-readable ISO format में convert करो
        expiry_text = (
            expiry_to_datetime(expiry_ms).isoformat() if expiry_ms else "N/A"  # अगर expiry है तो ISO format, नहीं तो "N/A"
        )
        lines.append(  # Formatted future line add करो
            _format_contract_line(
                prefix=f"Future #{idx}",  # Label: "Future #1", "Future #2", etc.
                trading_symbol=future.get("trading_symbol", "—"),  # Trading symbol
                instrument_key=future.get("instrument_key", "—"),  # Instrument key
                extra=f"[expiry: {expiry_text}]",  # Extra info: expiry timestamp
            )
        )

    # OPTIONS SECTION
    options = entry.get("options", {})  # Options data get करो (contains current_expiry, secondary_expiry)
    # Current और secondary दोनों expiry buckets को process करो
    for bucket_name in ("current_expiry", "secondary_expiry"):  # दोनों types के लिए iterate
        bucket = options.get(bucket_name)  # Specific expiry bucket get करो
        if not bucket:  # अगर यह bucket empty है
            continue  # Skip करो, next bucket process करो

        # Expiry metadata extract करो
        weekly_flag = "weekly" if bucket.get("weekly") else "monthly"  # Weekly या monthly expiry determine करो
        expiry_iso = bucket.get("expiry_iso")  # Pre-formatted expiry ISO string get करो
        if not expiry_iso and bucket.get("expiry"):  # अगर ISO string नहीं है लेकिन raw expiry है
            expiry_iso = expiry_to_datetime(bucket["expiry"]).isoformat()  # Convert करो

        # Options section header add करो
        lines.append(
            f"-- Options ({bucket_name.replace('_', ' ').title()} | {weekly_flag} | expiry {expiry_iso})"
            # Example: "-- Options (Current Expiry | weekly | expiry 2025-01-02T15:30:00+05:30)"
        )

        # सभी strikes को process करो
        strikes = bucket.get("slice", {}).get("strikes", [])  # Strike entries की list get करो
        for strike_entry in strikes:  # हर strike के लिए
            strike = strike_entry.get("strike")  # Strike price (e.g., 24000, 24100)
            strike_label = f"Strike {strike}" if strike is not None else "Strike ?"  # Label बनाओ
            contracts = strike_entry.get("contracts", {})  # CE और PE contracts dict get करो
            # CE और PE दोनों के लिए process करो
            for opt_type in ("CE", "PE"):  # CE = Call, PE = Put
                contract = contracts.get(opt_type)  # Specific option type का contract get करो
                if not contract:  # अगर contract नहीं है (missing CE या PE)
                    continue  # Skip करो
                lines.append(  # Option contract line add करो
                    _format_contract_line(
                        prefix=f"  {strike_label} {opt_type}",  # Indented label: "  Strike 24000 CE"
                        trading_symbol=contract.get("trading_symbol", "—"),  # Trading symbol
                        instrument_key=contract.get("instrument_key", "—"),  # Instrument key
                    )
                )

    return lines  # सभी formatted lines return करो


def build_report(payload):
    """
    Complete report build करो (सभी indices के लिए)
    
    Purpose: Filtered payload से formatted text report generate करना
    
    Args:
        payload: Complete filtered payload dict
                 Structure: {"indices": {...}, "equities": {...}}
    
    Returns:
        str: Complete formatted report (newlines के साथ joined)
    
    Process:
        1. Payload से सभी indices extract करो
        2. हर index के लिए summarize_index() call करो
        3. सभी sections को newlines के साथ join करो
        4. Final report string return करो
    
    Example:
        payload = load_payload()
        report = build_report(payload)
        print(report)  # Console पर complete report print हो जाएगी
    """
    indices = payload.get("indices", {})  # Indices dict get करो (keys: NIFTY, BANKNIFTY, SENSEX)
    snippets = []  # सभी text snippets collect करने के लिए empty list
    for label, entry in indices.items():  # हर index के लिए iterate
        snippets.extend(summarize_index(label, entry))  # Index की summary lines add करो
        snippets.append("")  # Spacer line add करो (sections के बीच में blank line)
    return "\n".join(snippets).strip()  # सभी lines को newlines से join करके strip करो (trailing whitespace remove)


def main():
    """
    CLI entry point - जब script directly run हो (python report_filtered_instruments.py)
    
    Purpose: Command line से report generate और print करना
    
    Process:
        1. Payload load करो (memory से या fresh build करके)
        2. Report build करो (formatted text)
        3. Console पर report print करो
    
    Example usage:
        $ python report_filtered_instruments.py
        
        === Index: NIFTY ===
        Spot           NIFTY (NSE_INDEX|Nifty 50)
        ...
        
        === Index: BANKNIFTY ===
        ...
    
    Error handling:
        - अगर कोई exception आती है तो stderr पर error print करके exit code 1 के साथ exit करो
    """
    payload = load_payload()  # Payload load करो
    report = build_report(payload)  # Report build करो
    print(report)  # Console पर report print करो


# यह check करता है कि script directly run हो रही है (not imported as module)
if __name__ == "__main__":
    try:
        main()  # main() function call करो
    except Exception as exc:  # pragma: no cover - CLI helper
        # अगर कोई error आती है तो stderr पर print करो और exit code 1 के साथ exit
        print(f"[report_filtered_instruments] Error: {exc}", file=sys.stderr)
        sys.exit(1)  # Exit with error code 1 (indicates failure)
