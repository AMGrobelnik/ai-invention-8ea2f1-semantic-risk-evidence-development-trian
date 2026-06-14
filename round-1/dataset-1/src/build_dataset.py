#!/usr/bin/env python3
"""Build SREDT dataset: LLM-generated risk scenarios + GDELT article retrieval."""

import json
import math
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import requests
from loguru import logger
from tqdm import tqdm
import openai

WORKSPACE = Path(__file__).parent
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Constants ─────────────────────────────────────────────────────────────────

ENERGY_COMPANIES = [
    {
        "company": "Shell", "country": "UK/NL", "aliases": ["Shell", "Royal Dutch Shell"],
        "gics_sector": "Energy", "gics_sector_code": "10",
        "gics_industry_group": "Energy", "gics_industry_group_code": "1010",
        "gics_industry": "Oil Gas and Consumable Fuels", "gics_industry_code": "101020",
        "gics_sub_industry": "Integrated Oil and Gas", "gics_sub_industry_code": "10102010",
    },
    {
        "company": "TotalEnergies", "country": "FR", "aliases": ["TotalEnergies", "Total"],
        "gics_sector": "Energy", "gics_sector_code": "10",
        "gics_industry_group": "Energy", "gics_industry_group_code": "1010",
        "gics_industry": "Oil Gas and Consumable Fuels", "gics_industry_code": "101020",
        "gics_sub_industry": "Integrated Oil and Gas", "gics_sub_industry_code": "10102010",
    },
    {
        "company": "BP", "country": "UK", "aliases": ["BP"],
        "gics_sector": "Energy", "gics_sector_code": "10",
        "gics_industry_group": "Energy", "gics_industry_group_code": "1010",
        "gics_industry": "Oil Gas and Consumable Fuels", "gics_industry_code": "101020",
        "gics_sub_industry": "Integrated Oil and Gas", "gics_sub_industry_code": "10102010",
    },
    {
        "company": "Equinor", "country": "NO", "aliases": ["Equinor"],
        "gics_sector": "Energy", "gics_sector_code": "10",
        "gics_industry_group": "Energy", "gics_industry_group_code": "1010",
        "gics_industry": "Oil Gas and Consumable Fuels", "gics_industry_code": "101020",
        "gics_sub_industry": "Oil and Gas Exploration and Production", "gics_sub_industry_code": "10102020",
    },
    {
        "company": "E.ON", "country": "DE", "aliases": ["E.ON", "EON"],
        "gics_sector": "Utilities", "gics_sector_code": "55",
        "gics_industry_group": "Utilities", "gics_industry_group_code": "5510",
        "gics_industry": "Electric Utilities", "gics_industry_code": "551010",
        "gics_sub_industry": "Electric Utilities", "gics_sub_industry_code": "55101010",
    },
    {
        "company": "RWE", "country": "DE", "aliases": ["RWE"],
        "gics_sector": "Utilities", "gics_sector_code": "55",
        "gics_industry_group": "Utilities", "gics_industry_group_code": "5510",
        "gics_industry": "Independent Power Producers and Energy Traders", "gics_industry_code": "551050",
        "gics_sub_industry": "Independent Power Producers and Energy Traders", "gics_sub_industry_code": "55105010",
    },
    {
        "company": "EDF", "country": "FR", "aliases": ["EDF"],
        "gics_sector": "Utilities", "gics_sector_code": "55",
        "gics_industry_group": "Utilities", "gics_industry_group_code": "5510",
        "gics_industry": "Electric Utilities", "gics_industry_code": "551010",
        "gics_sub_industry": "Electric Utilities", "gics_sub_industry_code": "55101010",
    },
    {
        "company": "Neste", "country": "FI", "aliases": ["Neste"],
        "gics_sector": "Energy", "gics_sector_code": "10",
        "gics_industry_group": "Energy", "gics_industry_group_code": "1010",
        "gics_industry": "Oil Gas and Consumable Fuels", "gics_industry_code": "101020",
        "gics_sub_industry": "Oil and Gas Refining and Marketing", "gics_sub_industry_code": "10102030",
    },
    {
        "company": "Eni", "country": "IT", "aliases": ["Eni"],
        "gics_sector": "Energy", "gics_sector_code": "10",
        "gics_industry_group": "Energy", "gics_industry_group_code": "1010",
        "gics_industry": "Oil Gas and Consumable Fuels", "gics_industry_code": "101020",
        "gics_sub_industry": "Integrated Oil and Gas", "gics_sub_industry_code": "10102010",
    },
    {
        "company": "Repsol", "country": "ES", "aliases": ["Repsol"],
        "gics_sector": "Energy", "gics_sector_code": "10",
        "gics_industry_group": "Energy", "gics_industry_group_code": "1010",
        "gics_industry": "Oil Gas and Consumable Fuels", "gics_industry_code": "101020",
        "gics_sub_industry": "Integrated Oil and Gas", "gics_sub_industry_code": "10102010",
    },
]

FINANCIALS_COMPANIES = [
    {
        "company": "BNP Paribas", "country": "FR", "aliases": ["BNP Paribas", "BNP"],
        "gics_sector": "Financials", "gics_sector_code": "40",
        "gics_industry_group": "Banks", "gics_industry_group_code": "4010",
        "gics_industry": "Banks", "gics_industry_code": "401010",
        "gics_sub_industry": "Diversified Banks", "gics_sub_industry_code": "40101010",
    },
    {
        "company": "Deutsche Bank", "country": "DE", "aliases": ["Deutsche Bank"],
        "gics_sector": "Financials", "gics_sector_code": "40",
        "gics_industry_group": "Banks", "gics_industry_group_code": "4010",
        "gics_industry": "Banks", "gics_industry_code": "401010",
        "gics_sub_industry": "Diversified Banks", "gics_sub_industry_code": "40101010",
    },
    {
        "company": "ING", "country": "NL", "aliases": ["ING"],
        "gics_sector": "Financials", "gics_sector_code": "40",
        "gics_industry_group": "Banks", "gics_industry_group_code": "4010",
        "gics_industry": "Banks", "gics_industry_code": "401010",
        "gics_sub_industry": "Diversified Banks", "gics_sub_industry_code": "40101010",
    },
    {
        "company": "Allianz", "country": "DE", "aliases": ["Allianz"],
        "gics_sector": "Financials", "gics_sector_code": "40",
        "gics_industry_group": "Insurance", "gics_industry_group_code": "4030",
        "gics_industry": "Insurance", "gics_industry_code": "403020",
        "gics_sub_industry": "Multi-line Insurance", "gics_sub_industry_code": "40302010",
    },
    {
        "company": "AXA", "country": "FR", "aliases": ["AXA"],
        "gics_sector": "Financials", "gics_sector_code": "40",
        "gics_industry_group": "Insurance", "gics_industry_group_code": "4030",
        "gics_industry": "Insurance", "gics_industry_code": "403020",
        "gics_sub_industry": "Multi-line Insurance", "gics_sub_industry_code": "40302010",
    },
    {
        "company": "Intesa Sanpaolo", "country": "IT", "aliases": ["Intesa Sanpaolo", "Intesa"],
        "gics_sector": "Financials", "gics_sector_code": "40",
        "gics_industry_group": "Banks", "gics_industry_group_code": "4010",
        "gics_industry": "Banks", "gics_industry_code": "401010",
        "gics_sub_industry": "Diversified Banks", "gics_sub_industry_code": "40101010",
    },
    {
        "company": "HSBC", "country": "UK", "aliases": ["HSBC"],
        "gics_sector": "Financials", "gics_sector_code": "40",
        "gics_industry_group": "Banks", "gics_industry_group_code": "4010",
        "gics_industry": "Banks", "gics_industry_code": "401010",
        "gics_sub_industry": "Diversified Banks", "gics_sub_industry_code": "40101010",
    },
    {
        "company": "Barclays", "country": "UK", "aliases": ["Barclays"],
        "gics_sector": "Financials", "gics_sector_code": "40",
        "gics_industry_group": "Banks", "gics_industry_group_code": "4010",
        "gics_industry": "Banks", "gics_industry_code": "401010",
        "gics_sub_industry": "Diversified Banks", "gics_sub_industry_code": "40101010",
    },
    {
        "company": "Societe Generale", "country": "FR", "aliases": ["Societe Generale", "SocGen"],
        "gics_sector": "Financials", "gics_sector_code": "40",
        "gics_industry_group": "Banks", "gics_industry_group_code": "4010",
        "gics_industry": "Banks", "gics_industry_code": "401010",
        "gics_sub_industry": "Diversified Banks", "gics_sub_industry_code": "40101010",
    },
    {
        "company": "ABN AMRO", "country": "NL", "aliases": ["ABN AMRO"],
        "gics_sector": "Financials", "gics_sector_code": "40",
        "gics_industry_group": "Banks", "gics_industry_group_code": "4010",
        "gics_industry": "Banks", "gics_industry_code": "401010",
        "gics_sub_industry": "Diversified Banks", "gics_sub_industry_code": "40101010",
    },
]

TRAIN_WINDOWS = [
    ("2025-07-01", "2025-09-29"),
    ("2025-08-01", "2025-10-30"),
    ("2025-09-01", "2025-11-30"),
    ("2025-10-01", "2025-12-30"),
]

TEST_WINDOWS = [
    ("2026-02-15", "2026-05-16"),
    ("2026-03-01", "2026-05-30"),
]

RISK_CATEGORIES_TRAIN = ["strategic", "credit", "market", "operational"]
RISK_CATEGORIES_TEST = ["liquidity", "compliance"]

GARP_RISK_MAP = {
    "strategic": ("Strategic", "Competitive positioning and market share"),
    "credit": ("Credit", "Counterparty default and credit exposure"),
    "market": ("Market", "Commodity price and energy price volatility"),
    "operational": ("Operational", "Process failures and internal controls"),
    "liquidity": ("Liquidity", "Funding liquidity and short-term refinancing"),
    "compliance": ("Compliance", "Regulatory change and policy uncertainty"),
    "reputational": ("Reputational", "ESG controversy and public perception"),
}

RISK_KEYWORDS = {
    "strategic": "strategy OR merger OR acquisition OR competition OR market",
    "credit": "credit OR debt OR default OR bond OR rating",
    "market": "price OR rate OR volatility OR market OR trading",
    "operational": "operations OR system OR outage OR failure OR compliance",
    "liquidity": "liquidity OR funding OR cash OR refinancing",
    "compliance": "regulation OR regulator OR fine OR compliance OR investigation",
    "reputational": "ESG OR reputation OR controversy OR litigation OR lawsuit",
}

TAXONOMY_LABELS = {
    "L1": {
        "energy": [
            "Integrated Oil and Gas",
            "Oil and Gas Exploration and Production",
            "Oil and Gas Refining and Marketing",
            "Oil and Gas Storage and Transportation",
            "Oil and Gas Drilling",
            "Oil and Gas Equipment and Services",
            "Coal and Consumable Fuels",
            "Electric Utilities",
            "Independent Power Producers and Energy Traders",
            "Renewable Electricity",
        ],
        "financials": [
            "Diversified Banks",
            "Regional Banks",
            "Consumer Finance",
            "Investment Banking and Brokerage",
            "Asset Management and Custody Banks",
            "Diversified Financial Services",
            "Life and Health Insurance",
            "Multi-line Insurance",
            "Property and Casualty Insurance",
            "Financial Exchanges and Data",
        ],
    },
    "L2": {
        "energy": ["Energy", "Utilities"],
        "financials": ["Banks", "Insurance", "Diversified Financials", "Capital Markets"],
    },
    "L3": [
        "Strategic risk: competitive positioning and market share",
        "Strategic risk: mergers acquisitions and corporate governance",
        "Strategic risk: technology disruption and digital transformation",
        "Credit risk: counterparty default and credit exposure",
        "Credit risk: sovereign and country risk",
        "Market risk: interest rate and yield curve exposure",
        "Market risk: foreign exchange and currency risk",
        "Market risk: commodity price and energy price volatility",
        "Market risk: equity market and asset price risk",
        "Operational risk: process failures and internal controls",
        "Operational risk: cybersecurity and information security",
        "Operational risk: supply chain and third-party vendor risk",
        "Operational risk: human capital and key personnel risk",
        "Liquidity risk: funding liquidity and short-term refinancing",
        "Liquidity risk: market liquidity and asset liquidation",
        "Compliance risk: regulatory change and policy uncertainty",
        "Compliance risk: environmental regulation and ESG compliance",
        "Compliance risk: anti-money laundering and financial crime",
        "Reputational risk: ESG controversy and public perception",
        "Reputational risk: litigation and legal proceedings",
    ],
}

BUDGET_LIMIT_USD = 8.0
total_cost_usd = 0.0
cost_lock = threading.Lock()

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


# ── Helpers ───────────────────────────────────────────────────────────────────

def company_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_seendate(s: str) -> str:
    """Convert GDELT seendate (YYYYMMDDTHHMMSSZ) to YYYY-MM-DD."""
    try:
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    except Exception:
        return s[:10] if s else ""


def build_scenario_id(sector: str, company: str, risk_category: str, start_date: str) -> str:
    slug = company_slug(company)
    date_slug = start_date.replace("-", "")
    return f"{sector}_{slug}_{risk_category}_{date_slug}"


def build_scenarios_list() -> list[dict]:
    """Build the full list of scenarios to generate (without text yet)."""
    scenarios = []
    for sector_name, companies in [("energy", ENERGY_COMPANIES), ("financials", FINANCIALS_COMPANIES)]:
        for company_info in companies:
            # Training: 4 windows with 4 risk categories
            for i, (start, end) in enumerate(TRAIN_WINDOWS):
                risk_cat = RISK_CATEGORIES_TRAIN[i]
                garp_l1, garp_l2 = GARP_RISK_MAP[risk_cat]
                eval_date = _midpoint(start, end, 45)
                sc_id = build_scenario_id(sector_name, company_info["company"], risk_cat, start)
                scenarios.append({
                    "scenario_id": sc_id,
                    "sector": sector_name,
                    "risk_category": risk_cat,
                    "garp_l1": garp_l1,
                    "garp_l2": garp_l2,
                    "start_date": start,
                    "end_date": end,
                    "evaluation_date": eval_date,
                    "split": "train",
                    **{k: v for k, v in company_info.items() if k != "aliases"},
                    "_aliases": company_info["aliases"],
                })
            # Test: 2 windows with 2 risk categories
            for i, (start, end) in enumerate(TEST_WINDOWS):
                risk_cat = RISK_CATEGORIES_TEST[i]
                garp_l1, garp_l2 = GARP_RISK_MAP[risk_cat]
                eval_date = _midpoint(start, end, 45)
                sc_id = build_scenario_id(sector_name, company_info["company"], risk_cat, start)
                scenarios.append({
                    "scenario_id": sc_id,
                    "sector": sector_name,
                    "risk_category": risk_cat,
                    "garp_l1": garp_l1,
                    "garp_l2": garp_l2,
                    "start_date": start,
                    "end_date": end,
                    "evaluation_date": eval_date,
                    "split": "test",
                    **{k: v for k, v in company_info.items() if k != "aliases"},
                    "_aliases": company_info["aliases"],
                })
    logger.info(f"Built {len(scenarios)} scenario specs")
    return scenarios


def _midpoint(start: str, end: str, day: int) -> str:
    """Return the date `day` days after start (bounded by end)."""
    from datetime import datetime, timedelta
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    result = s + timedelta(days=day)
    result = min(result, e)
    return result.strftime("%Y-%m-%d")


# ── Phase 1: LLM Scenario Generation ─────────────────────────────────────────

def _call_openrouter(client: openai.OpenAI, scenario: dict, attempt: int = 0) -> Optional[dict]:
    global total_cost_usd

    system_prompt = (
        "You are a European corporate risk analyst. Generate a realistic 90-day forward-looking "
        "risk scenario for the specified company. Return ONLY valid JSON with no extra text."
    )
    user_prompt = (
        f"Generate a risk scenario for:\n"
        f"- Company: {scenario['company']}\n"
        f"- Sector: {scenario['sector']}\n"
        f"- GICS Sub-industry: {scenario['gics_sub_industry']}\n"
        f"- Risk category: {scenario['risk_category']}\n"
        f"- Forecast window: {scenario['start_date']} to {scenario['end_date']}\n\n"
        f"Return exactly this JSON:\n"
        f'{{\n'
        f'  "scenario_text": "<2-3 sentence specific risk forecast starting with the company name, '
        f'describing a concrete plausible risk event or trend in this 90-day window, written in '
        f'present-tense future-oriented style as if published at the window start date>",\n'
        f'  "risk_catalyst": "<10-15 word phrase identifying the specific trigger>",\n'
        f'  "expected_impact": "<one of: high|medium|low>"\n'
        f'}}\n\n'
        f'Be specific: name regulatory bodies, price levels, geographic regions, or policy names '
        f'where plausible. Vary across different risk_category calls for the same company.'
    )

    try:
        resp = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=400,
            timeout=60,
        )
        text = resp.choices[0].message.content or ""
        logger.debug(f"LLM response for {scenario['scenario_id']}: {text[:200]}")

        # Track cost
        if resp.usage:
            cost = (resp.usage.prompt_tokens * 0.27 / 1_000_000 +
                    resp.usage.completion_tokens * 0.39 / 1_000_000)
            with cost_lock:
                total_cost_usd += cost
                if total_cost_usd > BUDGET_LIMIT_USD:
                    raise RuntimeError(f"Budget limit exceeded: ${total_cost_usd:.4f}")

        # Parse JSON
        parsed = _parse_llm_json(text)
        if parsed:
            return parsed

        # Retry once with more lenient extraction
        if attempt < 2:
            logger.warning(f"JSON parse failed for {scenario['scenario_id']}, retrying (attempt {attempt+1})")
            time.sleep(1)
            return _call_openrouter(client, scenario, attempt + 1)
        return None

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"LLM call failed for {scenario['scenario_id']}: {e}")
        if attempt < 2:
            time.sleep(3)
            return _call_openrouter(client, scenario, attempt + 1)
        return None


def _parse_llm_json(text: str) -> Optional[dict]:
    text = text.strip()
    # Try direct parse
    try:
        data = json.loads(text)
        if "scenario_text" in data:
            return data
    except json.JSONDecodeError:
        pass
    # Extract between first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end+1])
            if "scenario_text" in data:
                return data
        except json.JSONDecodeError:
            pass
    # Regex fallback for scenario_text
    m = re.search(r'"scenario_text"\s*:\s*"([^"]+)"', text)
    if m:
        catalyst = re.search(r'"risk_catalyst"\s*:\s*"([^"]+)"', text)
        impact = re.search(r'"expected_impact"\s*:\s*"([^"]+)"', text)
        return {
            "scenario_text": m.group(1),
            "risk_catalyst": catalyst.group(1) if catalyst else "",
            "expected_impact": impact.group(1) if impact else "medium",
        }
    return None


def generate_scenarios(scenarios: list[dict]) -> list[dict]:
    """Generate scenario text for all scenarios via OpenRouter LLM."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    intermediate_path = WORKSPACE / "scenarios_generated.json"
    generated: dict[str, dict] = {}

    # Load existing if resuming
    if intermediate_path.exists():
        existing = json.loads(intermediate_path.read_text())
        for s in existing:
            generated[s["scenario_id"]] = s
        logger.info(f"Resuming: {len(generated)} scenarios already generated")

    remaining = [s for s in scenarios if s["scenario_id"] not in generated]
    logger.info(f"Generating text for {len(remaining)} scenarios via OpenRouter")

    failed = 0
    with ThreadPoolExecutor(max_workers=5) as pool:
        future_to_sc = {pool.submit(_call_openrouter, client, sc): sc for sc in remaining}
        for future in tqdm(as_completed(future_to_sc), total=len(remaining), desc="Generating scenarios"):
            sc = future_to_sc[future]
            try:
                result = future.result()
                if result:
                    sc_copy = {k: v for k, v in sc.items() if not k.startswith("_")}
                    sc_copy["scenario_text"] = result["scenario_text"]
                    sc_copy["risk_catalyst"] = result.get("risk_catalyst", "")
                    sc_copy["expected_impact"] = result.get("expected_impact", "medium")
                    generated[sc["scenario_id"]] = sc_copy
                else:
                    failed += 1
                    logger.warning(f"No result for {sc['scenario_id']}")
            except RuntimeError as e:
                logger.error(f"Budget limit: {e}")
                pool.shutdown(wait=False, cancel_futures=True)
                break
            except Exception as e:
                failed += 1
                logger.error(f"Error for {sc['scenario_id']}: {e}")

            # Save checkpoint every 10
            if len(generated) % 10 == 0:
                intermediate_path.write_text(json.dumps(list(generated.values()), indent=2))
                logger.info(f"Checkpoint: {len(generated)} generated, cost=${total_cost_usd:.4f}")

    # Final save
    intermediate_path.write_text(json.dumps(list(generated.values()), indent=2))
    logger.info(f"Generated {len(generated)} scenarios, {failed} failed, cost=${total_cost_usd:.4f}")

    if failed > len(scenarios) * 0.2:
        logger.warning(f"High failure rate: {failed}/{len(scenarios)} scenarios failed")

    return list(generated.values())


# ── Phase 2: GDELT Article Retrieval ─────────────────────────────────────────

GDELT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (research; academic use) gdelt-api-client/1.0",
    "Accept": "application/json",
}


_gdelt_rate_limited = threading.Event()  # set if we detect persistent 429
_gdelt_consecutive_429 = 0
_gdelt_lock = threading.Lock()


def _gdelt_request(query: str, start_date: str, end_date: str, max_records: int = 100) -> list[dict]:
    """Make a GDELT DOC API request. Returns [] on 429 without retrying (caller handles skip)."""
    global _gdelt_consecutive_429
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "startdatetime": f"{start_date.replace('-', '')}000000",
        "enddatetime": f"{end_date.replace('-', '')}235959",
        "sourcelang": "english",
        "sort": "datedesc",
    }
    try:
        resp = requests.get(GDELT_BASE_URL, params=params, headers=GDELT_HEADERS, timeout=45)
        if resp.status_code in (429, 503):
            with _gdelt_lock:
                _gdelt_consecutive_429 += 1
            logger.warning(f"GDELT 429 (consecutive: {_gdelt_consecutive_429})")
            return []
        resp.raise_for_status()
        with _gdelt_lock:
            _gdelt_consecutive_429 = 0  # reset on success
        try:
            data = resp.json()
            return data.get("articles", []) or []
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"GDELT returned non-JSON response")
            return []
    except requests.RequestException as e:
        logger.error(f"GDELT request error: {e}")
        return []


def _filter_articles(articles: list[dict]) -> list[dict]:
    """Filter and clean GDELT articles."""
    seen_urls = set()
    filtered = []
    bad_domains = {"translate", "cache", "webcache", "translate.google"}
    for art in articles:
        url = art.get("url", "")
        title = art.get("title", "").strip()
        if not url or not title:
            continue
        domain = art.get("domain", "")
        if any(b in domain for b in bad_domains):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        filtered.append(art)
    return filtered


def retrieve_articles(scenario: dict) -> tuple[list[dict], str, int]:
    """Retrieve articles for a scenario with fallback strategy. Returns (articles, query_used, fallback_level)."""
    company = scenario["company"]
    aliases = scenario.get("_aliases", [company])
    risk_cat = scenario["risk_category"]
    start = scenario["start_date"]
    end = scenario["end_date"]

    risk_kw = RISK_KEYWORDS.get(risk_cat, "")
    primary_alias = aliases[0]
    secondary_alias = aliases[1] if len(aliases) > 1 else None

    # Build company query part
    if secondary_alias and secondary_alias != primary_alias:
        company_q = f'"{primary_alias}" OR "{secondary_alias}"'
    else:
        company_q = f'"{primary_alias}"'

    queries = [
        (f'{company_q} {risk_kw}', 0),
        (f'{company_q}', 1),
        (f'{primary_alias} Europe {scenario["sector"]}', 2),
    ]

    for query, fallback_level in queries:
        time.sleep(6)  # strict 6s between every request
        articles = _gdelt_request(query, start, end, max_records=100)
        if not articles and _gdelt_consecutive_429 > 0:
            # API is rate-limited; don't waste more requests on fallbacks
            return [], query, fallback_level
        articles = _filter_articles(articles)
        threshold = 10 if fallback_level == 0 else (5 if fallback_level == 1 else 3)
        logger.debug(f"{scenario['scenario_id']} fallback={fallback_level}: {len(articles)} articles")
        if len(articles) >= threshold:
            return articles[:50], query, fallback_level

    return articles[:50] if articles else [], queries[-1][0], 2


def retrieve_all_articles(scenarios: list[dict]) -> dict[str, dict]:
    """Retrieve articles for all scenarios sequentially with rate-limit aware retry."""
    global _gdelt_consecutive_429
    partial_path = WORKSPACE / "articles_partial.json"
    results: dict[str, dict] = {}

    if partial_path.exists():
        try:
            existing = json.loads(partial_path.read_text())
            # Only resume entries that actually have articles (0-article entries get retried)
            results = {k: v for k, v in existing.items() if len(v.get("articles", [])) > 0}
            logger.info(f"Resuming article retrieval: {len(results)} scenarios with articles from prior run")
        except Exception:
            pass

    remaining = [s for s in scenarios if s["scenario_id"] not in results]
    logger.info(f"Retrieving articles for {len(remaining)} scenarios")

    deferred: list[dict] = []  # scenarios deferred due to 429
    consecutive_429_threshold = 3  # if 3+ in a row, start deferring

    def _process_batch(batch: list[dict], desc: str, allow_defer: bool = True) -> list[dict]:
        """Process a batch. If allow_defer=True, 429 scenarios go to returned defer list.
        If allow_defer=False, 429 scenarios are saved as empty-article entries."""
        global _gdelt_consecutive_429
        newly_deferred: list[dict] = []
        # Snapshot the batch to avoid mutation issues
        batch_snapshot = list(batch)
        for i, sc in enumerate(tqdm(batch_snapshot, desc=desc)):
            _gdelt_consecutive_429 = 0  # reset at start of each scenario
            arts, query, fallback = retrieve_articles(sc)
            sc_id = sc["scenario_id"]

            if not arts and _gdelt_consecutive_429 > 0 and allow_defer:
                newly_deferred.append(sc)
                logger.warning(f"Deferred {sc_id} (rate limited)")
                continue

            results[sc_id] = {
                "articles": arts,
                "query_used": query,
                "retrieval_fallback_level": fallback,
            }
            logger.debug(f"{sc_id}: {len(arts)} articles (fallback={fallback})")

            if (i + 1) % 10 == 0:
                partial_path.write_text(json.dumps(results, indent=2))
                logger.info(f"Checkpoint: {len(results)} scenarios retrieved")

        partial_path.write_text(json.dumps(results, indent=2))
        return newly_deferred

    # Pass 1: process all remaining, deferring rate-limited scenarios
    deferred = _process_batch(remaining, "Retrieving articles (pass 1)", allow_defer=True)
    logger.info(f"Pass 1 complete: {len(results)} retrieved, {len(deferred)} deferred")

    # Pass 2: retry deferred scenarios after a long wait (let GDELT rate limit clear)
    if deferred:
        wait_secs = 900  # 15 minutes: GDELT ban typically clears in 15-20 min
        logger.info(f"Waiting {wait_secs}s ({wait_secs//60}m) before retrying {len(deferred)} deferred scenarios...")
        time.sleep(wait_secs)
        _gdelt_consecutive_429 = 0
        still_deferred = _process_batch(deferred, "Retrieving articles (pass 2 - retry)", allow_defer=False)
        logger.info(f"Pass 2 complete. Failed (marked empty): {len(still_deferred)}")

    # Mark anything still not retrieved as empty
    for sc in scenarios:
        if sc["scenario_id"] not in results:
            results[sc["scenario_id"]] = {
                "articles": [],
                "query_used": "",
                "retrieval_fallback_level": -1,
            }

    partial_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Article retrieval complete: {len(results)} scenarios")
    return results


# ── Phase 3: Assemble Output ──────────────────────────────────────────────────

def _format_article(sc_id: str, idx: int, art: dict) -> dict:
    seendate = art.get("seendate", "")
    pub_date = parse_seendate(seendate)
    return {
        "article_id": f"{sc_id}_{idx+1:03d}",
        "title": art.get("title", "").strip(),
        "url": art.get("url", ""),
        "publication_date": pub_date,
        "domain": art.get("domain", ""),
        "language": art.get("language", "English"),
        "source_country": art.get("sourcecountry", ""),
    }


def assemble_output(scenarios: list[dict], articles_map: dict[str, dict]) -> dict:
    """Build the final data_out.json structure."""
    now = datetime.utcnow().isoformat() + "Z"
    output_scenarios = []
    total_articles = 0
    low_coverage_count = 0

    for sc in scenarios:
        sc_id = sc["scenario_id"]
        art_data = articles_map.get(sc_id, {})
        raw_arts = art_data.get("articles", [])
        query_used = art_data.get("query_used", "")
        fallback_level = art_data.get("retrieval_fallback_level", -1)

        formatted_arts = [_format_article(sc_id, i, a) for i, a in enumerate(raw_arts)]
        is_low_cov = len(formatted_arts) < 10
        if is_low_cov:
            low_coverage_count += 1
        total_articles += len(formatted_arts)

        sc_out = {
            "scenario_id": sc_id,
            "company": sc["company"],
            "sector": sc["sector"],
            "gics_sector": sc.get("gics_sector", ""),
            "gics_sector_code": sc.get("gics_sector_code", ""),
            "gics_industry_group": sc.get("gics_industry_group", ""),
            "gics_industry_group_code": sc.get("gics_industry_group_code", ""),
            "gics_industry": sc.get("gics_industry", ""),
            "gics_industry_code": sc.get("gics_industry_code", ""),
            "gics_sub_industry": sc.get("gics_sub_industry", ""),
            "gics_sub_industry_code": sc.get("gics_sub_industry_code", ""),
            "risk_category": sc["risk_category"],
            "garp_l1": sc.get("garp_l1", ""),
            "garp_l2": sc.get("garp_l2", ""),
            "start_date": sc["start_date"],
            "end_date": sc["end_date"],
            "evaluation_date": sc.get("evaluation_date", ""),
            "split": sc["split"],
            "scenario_text": sc.get("scenario_text", ""),
            "risk_catalyst": sc.get("risk_catalyst", ""),
            "expected_impact": sc.get("expected_impact", "medium"),
            "low_coverage": is_low_cov,
            "articles": formatted_arts,
            "query_used": query_used,
            "retrieval_fallback_level": fallback_level,
        }
        output_scenarios.append(sc_out)

    train_scenarios = [s for s in output_scenarios if s["split"] == "train"]
    test_scenarios = [s for s in output_scenarios if s["split"] == "test"]

    data = {
        "metadata": {
            "created": now,
            "num_scenarios_total": len(output_scenarios),
            "num_train": len(train_scenarios),
            "num_test": len(test_scenarios),
            "sectors": ["energy", "financials"],
            "date_range_train": {
                "earliest_start": "2025-07-01",
                "latest_end": "2025-12-30",
            },
            "date_range_test": {
                "earliest_start": "2026-02-15",
                "latest_end": "2026-05-30",
            },
            "evaluation_day": 45,
            "gdelt_api_version": "2.0 artlist mode",
            "total_articles": total_articles,
            "low_coverage_scenarios": low_coverage_count,
            "llm_cost_usd": round(total_cost_usd, 6),
        },
        "scenarios": output_scenarios,
    }
    return data


# ── Phase 4: Validation ───────────────────────────────────────────────────────

def validate_output(data: dict) -> bool:
    """Run validation checks. Returns True if all critical checks pass."""
    scenarios = data.get("scenarios", [])
    errors = []
    warnings = []

    if len(scenarios) < 60:
        errors.append(f"Too few scenarios: {len(scenarios)} < 60")

    sectors = set(s["sector"] for s in scenarios)
    if len(sectors) < 2:
        errors.append(f"Missing sectors: found {sectors}")

    splits = set(s["split"] for s in scenarios)
    if "train" not in splits:
        errors.append("No train scenarios")
    if "test" not in splits:
        errors.append("No test scenarios")

    ids = [s["scenario_id"] for s in scenarios]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate scenario_ids found")

    today = date.today()
    for sc in scenarios:
        try:
            start = date.fromisoformat(sc["start_date"])
            end = date.fromisoformat(sc["end_date"])
            if start >= end:
                errors.append(f"{sc['scenario_id']}: start >= end")
            if sc["split"] == "train":
                days_elapsed = (today - end).days
                if days_elapsed < 45:
                    warnings.append(f"{sc['scenario_id']}: train window not fully elapsed ({days_elapsed} days)")
        except ValueError as e:
            errors.append(f"{sc['scenario_id']}: bad date: {e}")

        for art in sc.get("articles", []):
            if not art.get("url"):
                warnings.append(f"{sc['scenario_id']}: article with empty URL")

    art_counts = [len(sc.get("articles", [])) for sc in scenarios]
    if art_counts:
        median_arts = sorted(art_counts)[len(art_counts) // 2]
        if median_arts < 10:
            warnings.append(f"Median articles per scenario: {median_arts} < 10 (coverage concern)")

    for e in errors:
        logger.error(f"VALIDATION ERROR: {e}")
    for w in warnings:
        logger.warning(f"VALIDATION WARNING: {w}")

    if errors:
        logger.error(f"Validation FAILED with {len(errors)} errors")
        return False
    logger.info(f"Validation PASSED ({len(warnings)} warnings)")
    return True


# ── Phase 5: Coverage Report ──────────────────────────────────────────────────

def build_coverage_report(data: dict) -> dict:
    scenarios = data["scenarios"]
    art_counts = [len(s["articles"]) for s in scenarios]

    domain_counts: dict[str, int] = {}
    for sc in scenarios:
        for art in sc["articles"]:
            d = art.get("domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1

    top_domains = sorted(domain_counts.items(), key=lambda x: -x[1])[:10]

    by_sector: dict[str, list[int]] = {}
    by_split: dict[str, list[int]] = {}
    for sc in scenarios:
        sec = sc["sector"]
        spl = sc["split"]
        n = len(sc["articles"])
        by_sector.setdefault(sec, []).append(n)
        by_split.setdefault(spl, []).append(n)

    def stats(counts: list[int]) -> dict:
        if not counts:
            return {}
        s = sorted(counts)
        return {
            "mean": round(sum(s) / len(s), 2),
            "median": s[len(s) // 2],
            "min": s[0],
            "max": s[-1],
            "count": len(s),
        }

    return {
        "overall": stats(art_counts),
        "by_sector": {k: stats(v) for k, v in by_sector.items()},
        "by_split": {k: stats(v) for k, v in by_split.items()},
        "low_coverage_scenarios": sum(1 for n in art_counts if n < 10),
        "top_domains": [{"domain": d, "count": c} for d, c in top_domains],
    }


# ── Mini / Preview Generation ─────────────────────────────────────────────────

def build_mini(data: dict, n_scenarios: int = 10, n_articles: int = 5) -> dict:
    scenarios = data["scenarios"]
    # Take 5 per sector, mix of train/test
    selected = []
    for sector in ["energy", "financials"]:
        sec_sc = [s for s in scenarios if s["sector"] == sector]
        train_sc = [s for s in sec_sc if s["split"] == "train"][:3]
        test_sc = [s for s in sec_sc if s["split"] == "test"][:2]
        selected.extend(train_sc + test_sc)

    mini_scenarios = []
    for sc in selected[:n_scenarios]:
        sc_copy = dict(sc)
        sc_copy["articles"] = sc["articles"][:n_articles]
        mini_scenarios.append(sc_copy)

    mini = dict(data)
    mini["metadata"] = dict(data["metadata"])
    mini["metadata"]["num_scenarios_total"] = len(mini_scenarios)
    mini["scenarios"] = mini_scenarios
    return mini


def build_preview(data: dict, n_scenarios: int = 3, n_articles: int = 3, max_str: int = 300) -> dict:
    def truncate(obj):
        if isinstance(obj, str):
            return obj[:max_str] + ("..." if len(obj) > max_str else "")
        elif isinstance(obj, dict):
            return {k: truncate(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [truncate(x) for x in obj]
        return obj

    mini = build_mini(data, n_scenarios=n_scenarios, n_articles=n_articles)
    return truncate(mini)


# ── Main ──────────────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def main():
    logger.info("=== SREDT Dataset Builder ===")

    # Step 1: Build scenario specs
    scenarios = build_scenarios_list()
    logger.info(f"Total scenarios planned: {len(scenarios)}")

    # Step 2: Generate scenario texts via LLM
    logger.info("--- Phase 1: LLM Scenario Generation ---")
    generated_scenarios = generate_scenarios(scenarios)
    logger.info(f"Scenarios with generated text: {len(generated_scenarios)}, cost=${total_cost_usd:.4f}")

    # Merge: keep all scenarios, use generated text where available
    gen_map = {s["scenario_id"]: s for s in generated_scenarios}
    full_scenarios = []
    for sc in scenarios:
        sc_id = sc["scenario_id"]
        if sc_id in gen_map:
            merged = dict(sc)
            merged.update(gen_map[sc_id])
            full_scenarios.append(merged)
        else:
            sc_copy = {k: v for k, v in sc.items() if not k.startswith("_")}
            sc_copy.update({
                "scenario_text": f"{sc['company']} faces significant {sc['risk_category']} risk in the {sc['start_date']} to {sc['end_date']} period, driven by European market pressures and regulatory environment changes.",
                "risk_catalyst": f"{sc['risk_category'].capitalize()} pressure in European {sc['sector']} sector",
                "expected_impact": "medium",
            })
            full_scenarios.append(sc_copy)
            logger.warning(f"Using fallback text for {sc_id}")

    # Step 3: GDELT article retrieval
    logger.info("--- Phase 2: GDELT Article Retrieval ---")
    # Inject aliases back for retrieval
    alias_map = {sc["company"]: sc.get("_aliases", [sc["company"]]) for sc in scenarios}
    for sc in full_scenarios:
        sc["_aliases"] = alias_map.get(sc["company"], [sc["company"]])

    articles_map = retrieve_all_articles(full_scenarios)

    # Remove internal aliases from scenarios
    for sc in full_scenarios:
        sc.pop("_aliases", None)

    # Step 4: Assemble output
    logger.info("--- Phase 3: Assembling Output ---")
    data = assemble_output(full_scenarios, articles_map)

    # Step 5: Validate
    logger.info("--- Phase 4: Validation ---")
    valid = validate_output(data)
    if not valid:
        logger.warning("Validation had errors — proceeding anyway (see errors above)")

    # Step 6: Save outputs
    out_dir = WORKSPACE
    data_out_path = out_dir / "data_out.json"
    data_out_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved data_out.json ({data_out_path.stat().st_size / 1e6:.1f} MB)")

    mini_data = build_mini(data)
    mini_path = out_dir / "data_out_mini.json"
    mini_path.write_text(json.dumps(mini_data, indent=2))
    logger.info(f"Saved data_out_mini.json ({mini_path.stat().st_size / 1024:.0f} KB)")

    preview_data = build_preview(data)
    preview_path = out_dir / "data_out_preview.json"
    preview_path.write_text(json.dumps(preview_data, indent=2))
    logger.info(f"Saved data_out_preview.json")

    taxonomy_path = out_dir / "taxonomy_labels.json"
    taxonomy_path.write_text(json.dumps(TAXONOMY_LABELS, indent=2))
    logger.info(f"Saved taxonomy_labels.json")

    coverage = build_coverage_report(data)
    coverage_path = out_dir / "coverage_report.json"
    coverage_path.write_text(json.dumps(coverage, indent=2))
    logger.info(f"Saved coverage_report.json")

    # Step 7: File size check
    size_mb = data_out_path.stat().st_size / 1e6
    if size_mb > 50:
        logger.warning(f"data_out.json is {size_mb:.1f} MB > 50 MB, truncating articles to 30 per scenario")
        for sc in data["scenarios"]:
            sc["articles"] = sc["articles"][:30]
        data["metadata"]["total_articles"] = sum(len(sc["articles"]) for sc in data["scenarios"])
        data_out_path.write_text(json.dumps(data, indent=2))
        new_size = data_out_path.stat().st_size / 1e6
        logger.info(f"After truncation: {new_size:.1f} MB")

    logger.info("=== DONE ===")
    logger.info(f"Scenarios: {data['metadata']['num_scenarios_total']} "
                f"(train={data['metadata']['num_train']}, test={data['metadata']['num_test']})")
    logger.info(f"Total articles: {data['metadata']['total_articles']}")
    logger.info(f"Low coverage: {data['metadata']['low_coverage_scenarios']}")
    logger.info(f"LLM cost: ${data['metadata']['llm_cost_usd']}")
    logger.info(f"Coverage stats: {coverage['overall']}")


if __name__ == "__main__":
    main()
