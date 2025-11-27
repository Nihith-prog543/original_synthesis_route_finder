import os
import time
from typing import List, Dict, Set

import pandas as pd
from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.openai.chat import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from phi.tools.crawl4ai_tools import Crawl4aiTools


class ApiManufacturerDiscoveryService:
    """
    Groq-powered discovery flow that mirrors the user-provided script:
    - Uses DuckDuckGo + Crawl4AI agents (no Google CSE)
    - Skips known manufacturers from the seed CSV + Supabase
    - Persists discoveries through ApiManufacturerService (Supabase)
    """

    def __init__(self, manufacturer_service):
        self.manufacturer_service = manufacturer_service
        self.batch_size = int(os.getenv("MANUFACTURER_DISCOVERY_BATCH", "30"))
        self.csv_path = os.getenv("MANUFACTURER_CSV_PATH", self._default_csv_path())

        groq_model_id = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")
        self.primary_model = None
        try:
            self.primary_model = Groq(id=groq_model_id)
        except Exception:
            self.primary_model = None

        openai_key = os.getenv("OPENAI_API_KEY")
        self.fallback_model = None
        if openai_key:
            try:
                openai_model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o")
                self.fallback_model = OpenAIChat(id=openai_model_id, api_key=openai_key)
            except Exception:
                self.fallback_model = None

    def discover(self, api_name: str, country: str):
        api_name = (api_name or "").strip()
        country = (country or "").strip()
        if not api_name or not country:
            return {"success": False, "error": "API name and country are required for discovery."}
        if not self.primary_model and not self.fallback_model:
            return {"success": False, "error": "No LLM configured. Set GROQ_API_KEY or OPENAI_API_KEY."}

        existing_records = self.manufacturer_service.query(api_name, country)
        skip_from_db = {
            (rec.get("manufacturer") or "").strip().lower()
            for rec in existing_records
            if rec.get("manufacturer")
        }
        csv_df = self._load_seed_dataframe()
        seed_matches = csv_df[
            (csv_df["apiname"].str.contains(api_name.lower(), na=False))
            & (csv_df["country"].str.contains(country.lower(), na=False))
        ]
        skip_from_csv = set(seed_matches["manufacturers"].dropna())
        combined_skip = sorted({name.lower() for name in skip_from_db | skip_from_csv})

        discoveries = self._run_agents(api_name.lower(), country.lower(), combined_skip)
        if discoveries:
            insert_result = self.manufacturer_service.insert_records(discoveries, "groq_agents")
            inserted_rows = insert_result["rows"]
            inserted_count = insert_result["inserted"]
        else:
            inserted_rows = []
            inserted_count = 0

        refreshed = self.manufacturer_service.query(api_name, country)
        return {
            "success": True,
            "existing_records": existing_records,
            "new_records": inserted_rows,
            "all_records": refreshed,
            "inserted_count": inserted_count,
        }

    # ------------------------------------------------------------------
    # Agent pipeline
    # ------------------------------------------------------------------

    def _run_agents(self, api_name: str, country: str, skip_list: List[str]) -> List[Dict]:
        skip_batches = [skip_list[i : i + self.batch_size] for i in range(0, len(skip_list), self.batch_size)] or [[]]
        collected: List[Dict] = []
        models = [self.primary_model, self.fallback_model]

        for model in models:
            if not model:
                continue

            for batch in skip_batches:
                web_agent = self._create_agent("Web Agent", api_name, country, batch, DuckDuckGoTools(), model)
                pharma_agent = self._create_agent("Pharma Agent", api_name, country, batch, Crawl4aiTools(), model)

                try:
                    web_result = web_agent.run(f"Find {api_name} API manufacturers in {country}.")
                    if web_result and getattr(web_result, "content", None):
                        collected.extend(self._extract_manufacturers(web_result.content, batch, api_name, country, "web_agent"))
                except Exception:
                    pass

                time.sleep(1)

                try:
                    pharma_result = pharma_agent.run(f"Crawl for {api_name} API manufacturers in {country}.")
                    if pharma_result and getattr(pharma_result, "content", None):
                        collected.extend(self._extract_manufacturers(pharma_result.content, batch, api_name, country, "pharma_agent"))
                except Exception:
                    pass

                time.sleep(1)
                if collected:
                    return collected

        # Deduplicate by manufacturer+country to avoid double inserts
        unique = {}
        for row in collected:
            key = (row["manufacturer"].lower(), row["country"].lower())
            if key not in unique:
                unique[key] = row
        return list(unique.values())

    def _create_agent(self, name: str, api_name: str, country: str, skip_list: List[str], tool, model):
        skip_clause = ", ".join(skip_list) if skip_list else "None"
        instructions = f"""
Search FDA Orange Book, Pharmaoffer, Pharmacompass and other trusted pharma directories for API manufacturers of {api_name} in {country}.
Avoid known manufacturers: {skip_clause}.
Return strictly as a Markdown table:
| manufacturers | country | usdmf | cep |
"""
        return Agent(
            name=name,
            role="Discover new API manufacturers",
            model=model,
            tools=[tool],
            instructions=instructions,
            show_tool_calls=True,
            markdown=True,
        )

    def _extract_manufacturers(
        self,
        markdown_output: str,
        skip_batch: List[str],
        api_name: str,
        country: str,
        source_label: str,
    ) -> List[Dict]:
        manufacturers = []
        if not markdown_output:
            return manufacturers

        skip_lower = {item.lower() for item in skip_batch}
        lines = markdown_output.splitlines()
        for line in lines:
            if "|" not in line or line.lower().startswith("| manufacturers") or line.startswith("|---"):
                continue
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) < 4:
                continue

            manu_name = parts[0]
            manu_lower = manu_name.lower()
            if manu_lower in skip_lower:
                continue

            country_val = parts[1].lower()
            if country not in country_val:
                continue

            usdmf_status = "Yes" if parts[2].lower() in ("yes", "y", "t") else "No"
            cep_status = "Yes" if parts[3].lower() in ("yes", "y", "t") else "No"

            manufacturers.append(
                {
                    "api_name": api_name,
                    "manufacturer": manu_name,
                    "country": country_val,
                    "usdmf": usdmf_status,
                    "cep": cep_status,
                    "source_name": source_label,
                    "source_url": "",
                }
            )
        return manufacturers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_seed_dataframe(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_path, encoding="latin1")
        except Exception:
            return pd.DataFrame(columns=["apiname", "manufacturers", "country", "usdmf", "cep"])

        df.columns = [col.strip().lower() for col in df.columns]
        for required in ("apiname", "manufacturers", "country"):
            if required not in df.columns:
                df[required] = ""
        df["apiname"] = df["apiname"].astype(str).str.strip().str.lower()
        df["country"] = df["country"].astype(str).str.strip().str.lower()
        df["manufacturers"] = df["manufacturers"].astype(str).str.strip().str.lower()
        return df

    def _default_csv_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_root = os.path.dirname(project_root)
        candidate = os.path.join(repo_root, "API_Manufacturers_List.csv")
        if os.path.exists(candidate):
            return candidate
        return candidate  # fallback path even if nonexistent

