import logging
import difflib
from google.adk.tools import FunctionTool
from utils.session_context import get_session_id, store_charge_type

logger = logging.getLogger(__name__)

# Common filler words to ignore when comparing descriptions
STOPWORDS = {
    "and",
    "&",
    "with",
    "for",
    "to",
    "a",
    "an",
    "the",
    "of",
}


CHARGE_TYPE_MASTER = [
    {"charge_type": "ZADJ", "description": "Fuel Surcharge"},
    {"charge_type": "ZDIESEL_VOUCHER", "description": "Freight Charge"},
    {"charge_type": "ZDUTY", "description": "Fuel Surcharge"},
    {"charge_type": "ZFUEL_SURCHARGE", "description": "Fuel Surcharge"},
    {"charge_type": "ZOF_BSF", "description": "Freight Charge"},
    {"charge_type": "ZOF_FUEL_CHRG", "description": "Fuel Surcharge"},
    {"charge_type": "ZOF_INSUR_CHG", "description": "Insurance Charge"},
    {"charge_type": "ZPRE12", "description": "Service Charge"},
    {"charge_type": "ZPRE9", "description": "Service Charge"},
    {"charge_type": "ZSRV_BASIC", "description": "Service Charge"},
    {"charge_type": "ZTRIP_ALLOWANCE", "description": "Freight Charge"},
    {"charge_type": "Z_BASE_FRGHT_RF", "description": "Freight Charge"},
    {"charge_type": "Z_EXPRESS", "description": "Freight Charge"},
    {"charge_type": "Z_FRKLFT_RFSC", "description": "Forklift at destination"},
    {"charge_type": "Z_FUEL_RFSC", "description": "Fuel Surcharge"},
    {"charge_type": "Z_LOAD_RFSC", "description": "Loading charge"},
    {"charge_type": "Z_PICKUP_RFSC", "description": "Pickup charge"},
    {"charge_type": "Z_UNLOAD_RFSC", "description": "Unloading charge"},
]


def _normalize(text: str) -> str:
    return (
        text.lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def _tokenize(text: str) -> set:
    return {
        word
        for word in _normalize(text).split()
        if word and word not in STOPWORDS
    }


def _fuzzy_score(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()



def resolve_charge_type(charge_description: str):
    """
    Resolve a free-text charge description to charge type(s) from CHARGE_TYPE_MASTER.

    Strategy:
    1. Exact description match (case-insensitive, ignoring underscores/dashes)
    2. Token overlap match
    3. Fuzzy string similarity as a fallback
    """
    if not charge_description or not charge_description.strip():
        return {
            "input_description": charge_description,
            "matches": [],
            "error": "Charge description is empty",
        }

    matches = []
    user_norm = _normalize(charge_description)

    # 1. Exact normalized description match
    exact_matches = [
        item
        for item in CHARGE_TYPE_MASTER
        if _normalize(item["description"]) == user_norm
    ]
    if exact_matches:
        matches = exact_matches
    else:
        # 2. Keyword overlap matching
        user_tokens = _tokenize(charge_description)
        token_matches = []
        for item in CHARGE_TYPE_MASTER:
            master_tokens = _tokenize(item["description"])
            if user_tokens & master_tokens:
                token_matches.append(item)

        if token_matches:
            matches = token_matches
        else:
            # 3. Fuzzy fallback (typo handling)
            for item in CHARGE_TYPE_MASTER:
                master_norm = _normalize(item["description"])
                score = _fuzzy_score(user_norm, master_norm)

                if score >= 0.70:  # confidence threshold
                    matches.append(item)

    logger.info(
        "Charge type resolution: '%s' -> %d match(es)",
        charge_description,
        len(matches),
    )

    # Do NOT auto-store the first match. The user must confirm which code to use;
    # only confirm_charge_type stores the chosen code. This avoids the agreement
    # being created with the wrong (first-listed) charge type.

    return {
        "input_description": charge_description,
        "matches": matches,
    }


resolve_charge_type_tool = FunctionTool(resolve_charge_type)