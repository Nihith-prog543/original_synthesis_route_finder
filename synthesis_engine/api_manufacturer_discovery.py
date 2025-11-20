import time
import os
import requests
from agno.models.groq import Groq
from urllib.parse import urlparse


class GoogleSearchTool:
    """Custom Google Search tool for agno Agent"""
    
    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
    
    def google_search(self, query: str, max_results: int = 10) -> str:
        """Search Google Custom Search Engine and return results as formatted text"""
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": min(max_results, 10)  # Google allows max 10 per request
            }
            res = requests.get(url, params=params, timeout=15)
            data = res.json()
            items = data.get("items", [])
            
            # Format results as text for the agent
            results_text = ""
            for item in items:
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                results_text += f"Title: {title}\nURL: {link}\nSnippet: {snippet}\n\n"
            
            return results_text if results_text else "No results found."
        except Exception as e:
            return f"Search error: {str(e)}"


class ApiManufacturerDiscoveryService:
    """
    Runs Groq-powered agents to discover new manufacturers and store them via ApiManufacturerService.
    """

    def __init__(self, manufacturer_service):
        self.manufacturer_service = manufacturer_service
    
        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            from groq import Groq as GroqClient
            self.groq_client = GroqClient(api_key=groq_api_key)
        else:
            self.groq_client = None
    
        # Initialize Google Search tool
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        if google_api_key and google_cse_id:
            self.google_tool = GoogleSearchTool(google_api_key, google_cse_id)
        else:
            self.google_tool = None
    
        self.trusted_domains = {
            "pharmacompass.com",
            "pharmaoffer.com",
            "orangebook.fda.gov",
            "fda.gov",
            "ema.europa.eu",
            "cdsco.gov.in",
            "who.int",
            "dcat.org",
            "scrip.pharmaintelligence.informa.com",
        }

    def discover(self, api_name: str, country: str):
        api_name = (api_name or "").strip()
        country = (country or "").strip()

        if not api_name or not country:
            return {
                "success": False,
                "error": "API name and country are required for discovery.",
            }

        existing_records = self.manufacturer_service.query(api_name, country)
        skip_list = sorted(
            {
                (rec.get("manufacturer") or "").strip().lower()
                for rec in existing_records
                if rec.get("manufacturer")
            }
        )
        batches = [
            skip_list[i : i + 30] for i in range(0, len(skip_list), 30)
        ] or [[]]

        discovered_records = []
        for batch in batches:
            discovered_records.extend(
                self._discover_with_google(api_name, country, batch)
            )
            if discovered_records:
                break  # stop once we find fresh data

        if discovered_records:
            insert_result = self.manufacturer_service.insert_records(discovered_records, "groq_discovery")
            newly_inserted = insert_result["rows"]
            inserted_count = insert_result["inserted"]
        else:
            newly_inserted = []
            inserted_count = 0

        refreshed = self.manufacturer_service.query(api_name, country)

        return {
            "success": True,
            "existing_records": existing_records,
            "new_records": newly_inserted,
            "all_records": refreshed,
            "inserted_count": inserted_count,
        }

    def _discover_with_google(self, api_name, country, skip_batch):
        records = []
        if not self.google_tool:
            return records

        try:
            time.sleep(1)  # Small delay to respect rate limits
            query = self._build_google_query(api_name, country)
            search_results = self._google_search(query)
            if not search_results:
                return records

            groq_output = self._analyze_with_groq(
                api_name, search_results, country, skip_batch
            )
            if groq_output:
                records.extend(
                    self._extract_manufacturers(
                        groq_output, api_name, country, skip_batch
                    )
                )
            else:
                # Attempt a lightweight heuristic extraction if Groq is unavailable
                records.extend(
                    self._extract_from_search_snippets(
                        search_results, api_name, country, skip_batch
                    )
                )
        except Exception:
            # Continue even if Google search fails
            pass

        return records

    def _build_google_query(self, api_name, country):
        trusted_sites = [
            "pharmacompass.com",
            "pharmaoffer.com",
            "fda.gov",
            "ema.europa.eu",
            "cdsco.gov.in",
        ]
        site_clause = " OR ".join(f"site:{domain}" for domain in trusted_sites)
        return (
            f"\"{api_name}\" API manufacturers \"{country}\" {site_clause}"
        )

    def _google_search(self, query: str) -> str:
        """Perform Google Custom Search and return formatted results"""
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_tool.api_key,
                "cx": self.google_tool.cse_id,
                "q": query,
                "num": 10
            }
            res = requests.get(url, params=params, timeout=15)
            data = res.json()
            items = data.get("items", [])
            
            # Format results as text
            results_text = ""
            for item in items:
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                results_text += f"Title: {title}\nURL: {link}\nSnippet: {snippet}\n\n"
            
            return results_text if results_text else None
        except Exception:
            return None
    
    def _analyze_with_groq(self, api_name: str, search_results: str, country: str, skip_list: list) -> str:
        """Use Groq to analyze Google search results and extract manufacturers"""
        if not self.groq_client:
            return None
            
        skip_clause = ", ".join(skip_list[:10]) if skip_list else "None"  # Limit skip list size
        
        prompt = f"""
You are a pharmaceutical business intelligence expert. Extract API manufacturers from the following search results for {api_name} in {country}.

Skip these known manufacturers: {skip_clause}

Search Results:
{search_results}

Return ONLY a markdown table with these exact columns:
| manufacturers | country | usdmf | cep | source_name | source_url |

Requirements:
- source_url must be an HTTPS link from a trusted domain (PharmaCompass, Pharmaoffer, FDA, EMA, CDSCO)
- source_name should be the website name
- Only include manufacturers that appear in the search results
- Focus on manufacturers in {country}
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a pharmaceutical data extraction expert. Return only valid markdown tables."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2000
            )
            return response.choices[0].message.content if response.choices else None
        except Exception:
            return None

    def _extract_manufacturers(self, markdown_output, api_name, country, skip_batch):
        manufacturers = []
        if not markdown_output:
            return manufacturers

        lines = markdown_output.splitlines()
        existing_lower = {name.lower() for name in skip_batch}
        for line in lines:
            if "|" in line and not line.lower().startswith("| manufacturers") and not line.startswith("|---"):
                raw_parts = [p.strip() for p in line.split("|")]
                if len(raw_parts) < 8:
                    continue
                parts = raw_parts[1:-1]  # drop leading/trailing blanks caused by table edges
                if len(parts) < 6:
                    continue
                manu_name = parts[0]
                manu_lower = manu_name.lower()
                if manu_lower in existing_lower:
                    continue
                country_val = parts[1]
                usdmf = "Yes" if parts[2].strip().lower() in ["yes", "t"] else "No"
                cep = "Yes" if parts[3].strip().lower() in ["yes", "t"] else "No"
                source_name = parts[4]
                source_url = parts[5]

                if not self._is_valid_source(source_url):
                    continue

                if country.lower() in country_val.lower():
                    manufacturers.append(
                        {
                            "api_name": api_name,
                            "manufacturer": manu_name,
                            "country": country_val,
                            "usdmf": usdmf,
                            "cep": cep,
                            "source_name": source_name,
                            "source_url": source_url,
                        }
                    )
        return manufacturers

    def _extract_from_search_snippets(self, search_results, api_name, country, skip_batch):
        """
        Simple fallback parser that looks for manufacturer names in Google snippets.
        This is less reliable than Groq extraction but ensures we return something when LLM analysis fails.
        """
        manufacturers = []
        existing_lower = {name.lower() for name in skip_batch}
        entries = [
            block.strip()
            for block in search_results.split("\n\n")
            if block.strip().startswith("Title:")
        ]

        for entry in entries:
            lines = entry.splitlines()
            title = next((line[6:].strip() for line in lines if line.startswith("Title:")), "")
            url = next((line[5:].strip() for line in lines if line.startswith("URL:")), "")
            snippet = next((line[8:].strip() for line in lines if line.startswith("Snippet:")), "")

            if not (title and url and self._is_valid_source(url)):
                continue

            candidate = title.split("-")[0].split("|")[0].strip()
            if not candidate or candidate.lower() in existing_lower:
                continue

            inferred_country = country if country.lower() in snippet.lower() else country
            manufacturers.append(
                {
                    "api_name": api_name,
                    "manufacturer": candidate,
                    "country": inferred_country,
                    "usdmf": "Unknown",
                    "cep": "Unknown",
                    "source_name": urlparse(url).netloc,
                    "source_url": url,
                }
            )

        return manufacturers

    def _is_valid_source(self, url: str) -> bool:
        if not url:
            return False
        parsed = urlparse(url.strip())
        if parsed.scheme.lower() != "https":
            return False
        domain = parsed.netloc.lower()
        return any(domain == trusted or domain.endswith(f".{trusted}") for trusted in self.trusted_domains)

