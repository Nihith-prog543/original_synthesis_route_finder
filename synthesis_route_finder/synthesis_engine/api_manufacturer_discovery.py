import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
try:
    from phi.tools.crawl4ai_tools import Crawl4aiTools
except ImportError:  # pragma: no cover - best effort fallback
    Crawl4aiTools = None

# === Load environment variables ===
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Supabase (PostgreSQL) connection string
SUPABASE_DB_URL = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    print("❌ Missing DATABASE_URL (Supabase connection string) in environment.")
    sys.exit(1)

CSV_DATA_PATH = os.getenv("MANUFACTURERS_CSV_PATH", r"/Users/prabhas/Desktop/prabhas5.csv")
GROQ_MODEL_ID = "llama-3.3-70b-versatile"


def load_dataset() -> pd.DataFrame:
    try:
        df = pd.read_csv(CSV_DATA_PATH, encoding="latin1")
        df["apiname"] = df["apiname"].str.strip().str.lower()
        df["country"] = df["country"].str.strip().str.lower()
        return df
    except Exception as exc:
        print(f"⚠️ CSV Load Failed: {exc}")
        return pd.DataFrame(columns=["apiname", "manufacturers", "country", "usdmf", "cep"])


# === Helper: Extract Manufacturer Info from Markdown ===
def extract_manufacturers(markdown_output, api_name, country_input, existing_manufacturers):
    manufacturers = []
    if not markdown_output:
        return manufacturers

    lines = markdown_output.splitlines()
    for line in lines:
        if "|" in line and not line.lower().startswith("| manufacturers") and not line.startswith("|---"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 5:
                manu_name = parts[0]
                country = parts[1].lower()
                usdmf_status = "Yes" if parts[2].strip().lower() in ["yes", "t"] else "No"
                cep_status = "Yes" if parts[3].strip().lower() in ["yes", "t"] else "No"
                source = parts[4]

                if manu_name.lower() not in existing_manufacturers and (country_input in country):
                    manufacturers.append(
                        {
                            "apiname": api_name,
                            "manufacturers": manu_name,
                            "country": country,
                            "usdmf": usdmf_status,
                            "cep": cep_status,
                            "source": source,
                        }
                    )

    return manufacturers


def create_pharma_agent(api_name, country_input, skip_list):
    if Crawl4aiTools is None:
        raise ImportError(
            "phi.tools.crawl4ai_tools is not installed. Install the 'phi' extras that provide Crawl4aiTools."
        )
    skip_clause = ", ".join(skip_list) if skip_list else "None"
    return Agent(
        name="Pharma Agent",
        role="Crawl pharma directories & FDA Orange Book for new manufacturers of API.",
        model=Groq(id=GROQ_MODEL_ID),
        tools=[Crawl4aiTools()],
        instructions=f"""
            Crawl FDA Orange Book, EMA, PMDA, DCGI, MHRA, and other regulator-backed sources for {api_name} API in {country_input}.
            Skip known manufacturers: {skip_clause}.
            Output strictly in Markdown table:
            | manufacturers | country | usdmf | cep | source |
            Where the 'source' column is a regulator acronym (e.g., FDA, EMA, PMDA, DCGI, MHRA, Health Canada).
            Exclude any manufacturer you cannot tie to one of those regulators.
        """,
        show_tool_calls=True,
        markdown=True,
    )


def run_discovery(
    api_name: str,
    country_input: str,
    persist: bool = True,
    existing_manufacturers: Optional[Set[str]] = None,
) -> Dict[str, object]:
    api_name = api_name.strip().lower()
    country_input = country_input.strip().lower()
    if not api_name or not country_input:
        raise ValueError("API name and country are required.")

    print(f"🔍 Looking up: API = '{api_name}' | Country = '{country_input}'")

    df = load_dataset()
    existing_df = df[
        (df["apiname"].str.contains(api_name, na=False))
        & (df["country"].str.contains(country_input, na=False))
    ]
    csv_existing_manufacturers = set(
        existing_df["manufacturers"].dropna().str.strip().str.lower().unique()
    )

    if existing_manufacturers is None:
        existing_manufacturers = csv_existing_manufacturers
    else:
        existing_manufacturers = {name.strip().lower() for name in existing_manufacturers if name}

    batch_size = 30
    existing_list = sorted(existing_manufacturers)
    batches = [existing_list[i : i + batch_size] for i in range(0, len(existing_list), batch_size)] or [[]]

    pharma_rows: List[Dict[str, str]] = []

    for skip_batch in batches:
        try:
            pharma_agent = create_pharma_agent(api_name, country_input, skip_batch)
            pharma_result = pharma_agent.run(f"Crawl for {api_name} API manufacturers in {country_input}.")
            if pharma_result:
                extracted = extract_manufacturers(
                    pharma_result.content, api_name, country_input, existing_manufacturers
                )
                for row in extracted:
                    row["source"] = row.get("source") or "regulator"
                pharma_rows.extend(extracted)
            time.sleep(2)
        except Exception as exc:
            print(f"⚠️ Pharma Agent failed: {exc}")

    combined_scraped_rows = pharma_rows

    if not combined_scraped_rows:
        message = f"❌ No API manufacturers found for '{api_name}' in '{country_input}'."
        print(message)
        return {"success": False, "message": message, "new_records": []}

    fresh_df = pd.DataFrame(combined_scraped_rows)
    if persist and not fresh_df.empty:
        insert_into_supabase(fresh_df)

    return {
        "success": True,
        "new_records": combined_scraped_rows,
        "inserted_count": len(combined_scraped_rows),
        "pharma_records": pharma_rows,
    }


# === Insert into Supabase PostgreSQL ===
def insert_into_supabase(fresh_df):
    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)

        cursor = conn.cursor()
        combined_df = fresh_df.drop_duplicates()

        insert_query = """
        INSERT INTO manufacturers (apiname, manufacturers, country, usdmf, cep)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (apiname,manufacturers, country) DO NOTHING;
        """

        for _, row in combined_df.iterrows():
            cursor.execute(
                insert_query,
                (row["apiname"], row["manufacturers"], row["country"], row["usdmf"], row["cep"]),
            )

        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Data inserted into Supabase successfully!")
    except Exception as e:
        print(f"❌ Error inserting into Supabase: {e}")


class ApiManufacturerDiscoveryService:
    """
    Wraps the discovery workflow so it can be reused inside the Flask app.
    Normalizes newly scraped records and stores them through ApiManufacturerService.
    """

    def __init__(self, manufacturer_service=None, source_label: str = "web_discovery"):
        self.manufacturer_service = manufacturer_service
        self.source_label = source_label

    def _get_skip_set(self, api_name: str, country: str) -> Tuple[List[Dict[str, str]], Set[str]]:
        if not self.manufacturer_service:
            return [], set()
        existing_records = self.manufacturer_service.query(api_name, country)
        skip = {
            (record.get("manufacturer") or "").strip().lower()
            for record in existing_records
            if record.get("manufacturer")
        }
        return existing_records, skip

    def discover(self, api_name: str, country: str) -> Dict[str, object]:
        api_name = (api_name or "").strip()
        country = (country or "").strip()

        if not api_name or not country:
            return {"success": False, "error": "API name and country are required"}

        existing_records, skip_set = self._get_skip_set(api_name, country)

        discovery_result = run_discovery(
            api_name,
            country,
            persist=True,
            existing_manufacturers=skip_set if skip_set else None,
        )

        if not discovery_result.get("success"):
            return discovery_result

        pharma_rows = discovery_result.get("pharma_records", [])
        normalized_rows = []
        for row in pharma_rows:
            manufacturer_name = row.get("manufacturers") or row.get("manufacturer")
            if not manufacturer_name:
                continue
            regulator_source = row.get("source") or self.source_label
            normalized_rows.append(
                {
                    "api_name": row.get("apiname", api_name),
                    "manufacturer": manufacturer_name,
                    "country": row.get("country", country),
                    "usdmf": row.get("usdmf", ""),
                    "cep": row.get("cep", ""),
                    "source_name": regulator_source,
                    "source_url": row.get("source_url", ""),
                }
            )

        inserted_rows = []
        inserted_count = 0
        if self.manufacturer_service:
            insert_result = self.manufacturer_service.insert_records(
                normalized_rows, source_label=self.source_label
            )
            inserted_rows = insert_result.get("rows", [])
            inserted_count = insert_result.get("inserted", 0)
        else:
            inserted_rows = normalized_rows
            inserted_count = len(inserted_rows)

        all_records = existing_records + inserted_rows

        return {
            "success": True,
            "existing_records": existing_records,
            "new_records": inserted_rows,
            "all_records": all_records,
            "inserted_count": inserted_count,
        }

    def purge_discovery_results(
        self,
        source_name: str,
        api_name: Optional[str] = None,
        country: Optional[str] = None,
        use_like: bool = True,
    ) -> Dict[str, object]:
        if not self.manufacturer_service:
            return {"success": False, "error": "Manufacturer service not configured"}

        deleted = self.manufacturer_service.delete_by_source(
            source_name=source_name,
            api_name=api_name,
            country=country,
            use_like=use_like,
        )
        return {"success": True, "deleted": deleted}


def main():
    if len(sys.argv) < 3:
        print("Usage: python api_manufacturer_discovery.py <api_name> <country>")
        sys.exit(1)

    _, api_arg, country_arg = sys.argv[:3]
    result = run_discovery(api_arg, country_arg)
    if not result.get("success"):
        sys.exit(0)


if __name__ == "__main__":
    main()

