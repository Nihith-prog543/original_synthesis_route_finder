# synthesis_engine/api_buyer_finder.py - API Buyer Search Logic
import pandas as pd
import logging
import warnings
import os
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import OpenAI
from groq import Groq
import re
import time
import requests
from bs4 import BeautifulSoup

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy.*")

class ApiBuyerFinder:
    def __init__(self):
        # ====== SQLITE CONFIG ======
        # Use absolute path to ensure we always use the same database file
        # First, check if environment variable is set (highest priority)
        db_filename = os.getenv("SQLITE_DB_FILENAME")
        
        if db_filename:
            # Environment variable is set, use it
            if not os.path.isabs(db_filename):
                # If relative path, make it absolute relative to current working directory
                db_filename = os.path.abspath(db_filename)
        else:
            # Try to find the database in common locations
            # Priority 1: manufactures_api folder (where Excel data was imported)
            manufactures_api_path = r"C:\Users\HP\Desktop\manufactures_api\viruj_local.db"
            if os.path.exists(manufactures_api_path):
                db_filename = manufactures_api_path
                logger.info(f"📁 Found database in manufactures_api folder")
            else:
                # Priority 2: Current project directory
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_file_dir)
                db_filename = os.path.join(project_root, "viruj_local.db")
                logger.info(f"📁 Using default database in project directory")
        
        self.SQLITE_DB_FILENAME = db_filename
        self.SQLALCHEMY_SQLITE_URI = f"sqlite:///{self.SQLITE_DB_FILENAME}"
        logger.info(f"📁 Database file path: {self.SQLITE_DB_FILENAME}")
        logger.info(f"📁 Database file exists: {os.path.exists(self.SQLITE_DB_FILENAME)}")
        # ============================

        # API keys loaded from environment variables (required - no defaults for security)
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "1475eceacf9eb4d06")

        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.groq_client = Groq(api_key=self.GROQ_API_KEY)

        self.TRUSTED_SOURCES = [
            "1mg.com", "netmeds.com", "apollo247.com", "drugs.com", "goodrx.com", "fda.gov",
            "cdsco.gov.in", "ema.europa.eu", "manufacturer", "company website", "product label",
            "regulatory filing", "retail listing", "apteka.ru", "zdravcity.ru", "piluli.ru",
            "pharmacompass.com", "pharmaoffer.com", "orangebook.fda.gov", "medindia.net",
            "tabletwise.net", "pharmeasy.in", "medplusmart.com"
        ]

        self.API_ONLY_KEYWORDS = [
            "api manufacturer", "api supplier", "bulk drug", "active pharmaceutical ingredient", 
            "raw material", "intermediate", "api only", "bulk supplier", "chemical manufacturer",
            "ingredient supplier", "raw api", "bulk api", "pharmaceutical ingredient"
        ]

    def _safe_val(self, v):
        """
        Convert pandas/numpy/datetime values to plain Python types for SQL binding.
        Returns None for NaN/pd.NA, ISO string for timestamps, python scalars for numpy types,
        and stringifies complex objects.
        """
        if v is None:
            return None
        # pandas NA / numpy nan
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        # pandas Timestamp or datetime
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.isoformat()
        # numpy scalar -> python scalar
        if isinstance(v, (np.generic,)):
            return v.item()
        # pandas Series / list / dict -> stringify
        if isinstance(v, (pd.Series, list, tuple, dict, set)):
            try:
                return str(v)
            except Exception:
                return None
        # bytes -> decode
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8")
            except Exception:
                return str(v)
        # primitive python types
        if isinstance(v, (str, int, float, bool)):
            return v
        # fallback
        try:
            return str(v)
        except Exception:
            return None

    def get_db_engine(self):
        """Return a SQLAlchemy engine for local SQLite DB and ensure the `viruj` table exists."""
        try:
            engine = create_engine(
                self.SQLALCHEMY_SQLITE_URI, 
                connect_args={"check_same_thread": False}, 
                echo=False
            )
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS viruj (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                form TEXT,
                strength TEXT,
                verification_source TEXT,
                confidence INTEGER,
                url TEXT,
                additional_info TEXT,
                created_at TEXT,
                updated_at TEXT,
                api TEXT,
                country TEXT,
                UNIQUE(api, country, company)
            );
            """
            with engine.begin() as conn:
                conn.execute(text(create_table_sql))
            logger.info(f"✅ SQLite DB ready at {self.SQLITE_DB_FILENAME}")
            return engine
        except SQLAlchemyError as e:
            logger.error(f"❌ DB connection/create failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected DB error: {e}")
            return None

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        stop=stop_after_attempt(3)
    )
    def agent_run_with_retry(self, prompt: str, context: str = "", instructions: str = "") -> str:
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": full_prompt})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content

    def fetch_existing_data(self, api: str, country: str) -> pd.DataFrame:
        engine = self.get_db_engine()
        if not engine:
            return pd.DataFrame()
        
        # Trim and normalize the input
        api_clean = api.strip() if api else ""
        country_clean = country.strip() if country else ""
        
        logger.info(f"🔍 Fetching existing data for API: '{api_clean}', Country: '{country_clean}'")
        
        # Use case-insensitive matching and handle NULL/empty values
        # Also trim whitespace from database values
        query = text("""
            SELECT company, form, strength, additional_info, api, country
            FROM viruj 
            WHERE TRIM(LOWER(COALESCE(api, ''))) = TRIM(LOWER(:api)) 
              AND TRIM(LOWER(COALESCE(country, ''))) = TRIM(LOWER(:country))
        """)
        
        with engine.begin() as conn:
            result = conn.execute(query, {"api": api_clean, "country": country_clean})
            rows = result.fetchall()
        
        # Also try a count query to see total matches
        count_query = text("""
            SELECT COUNT(*) as total
            FROM viruj 
            WHERE TRIM(LOWER(COALESCE(api, ''))) = TRIM(LOWER(:api)) 
              AND TRIM(LOWER(COALESCE(country, ''))) = TRIM(LOWER(:country))
        """)
        with engine.begin() as conn:
            count_result = conn.execute(count_query, {"api": api_clean, "country": country_clean})
            total_count = count_result.fetchone()[0]
            logger.info(f"📊 Query found {total_count} total matching records")
        
        if not rows:
            # Try a more lenient search with LIKE pattern matching
            logger.warning(f"⚠️ No exact matches found. Trying lenient search for API: '{api_clean}', Country: '{country_clean}'")
            lenient_query = text("""
                SELECT company, form, strength, additional_info, api, country
                FROM viruj 
                WHERE (api LIKE :api_pattern OR TRIM(LOWER(COALESCE(api, ''))) LIKE TRIM(LOWER(:api_like)))
                  AND (country LIKE :country_pattern OR TRIM(LOWER(COALESCE(country, ''))) LIKE TRIM(LOWER(:country_like)))
            """)
            with engine.begin() as conn:
                lenient_result = conn.execute(lenient_query, {
                    "api_pattern": f"%{api_clean}%",
                    "api_like": f"%{api_clean}%",
                    "country_pattern": f"%{country_clean}%",
                    "country_like": f"%{country_clean}%"
                })
                lenient_rows = lenient_result.fetchall()
                if lenient_rows:
                    logger.info(f"✅ Lenient search found {len(lenient_rows)} matching records")
                    df = pd.DataFrame(lenient_rows, columns=['company', 'form', 'strength', 'additional_info', 'api', 'country'])
                    return df
            
            # Debug: Check what's actually in the database
            logger.warning(f"⚠️ No existing companies found for API: '{api_clean}', Country: '{country_clean}'")
            debug_query = text("""
                SELECT DISTINCT api, country, COUNT(*) as cnt
                FROM viruj 
                WHERE api LIKE :api_pattern OR country LIKE :country_pattern
                GROUP BY api, country
                LIMIT 10
            """)
            with engine.begin() as conn:
                debug_result = conn.execute(debug_query, {
                    "api_pattern": f"%{api_clean}%",
                    "country_pattern": f"%{country_clean}%"
                })
                debug_rows = debug_result.fetchall()
                if debug_rows:
                    logger.info(f"🔍 Debug: Found similar records in DB:")
                    for row in debug_rows:
                        logger.info(f"   API: '{row[0]}', Country: '{row[1]}', Count: {row[2]}")
            
            return pd.DataFrame(columns=['company', 'form', 'strength', 'additional_info', 'api', 'country'])
        
        # Use pd.Index for columns for extra type safety
        df = pd.DataFrame(rows, columns=['company', 'form', 'strength', 'additional_info', 'api', 'country'])
        logger.info(f"✅ Successfully fetched {len(df)} existing companies from database")
        return df

    def fetch_existing_companies(self, api: str, country: str) -> list:
        engine = self.get_db_engine()
        if not engine:
            return []
        # Use case-insensitive matching
        query = text("SELECT DISTINCT company FROM viruj WHERE LOWER(api) = LOWER(:api) AND LOWER(country) = LOWER(:country)")
        with engine.begin() as conn:
            result = conn.execute(query, {"api": api, "country": country})
            return [row[0] for row in result.fetchall()]

    def build_enhanced_prompt(self, api: str, country: str, existing_companies: list) -> str:
        blacklist = ", ".join(existing_companies) if existing_companies else "None"
        prompt = f"""
**CRITICAL PHARMACEUTICAL RESEARCH TASK**

You are a pharmaceutical manufacturing data expert. Your ONLY task is to find companies that manufacture FINISHED DOSAGE FORMS (FDF) or FINISHED DOSAGE COMBINATIONS (FDC) that contain the specific API '{api}' as an ingredient in {country}.

**ABSOLUTE EXCLUSION CRITERIA - DO NOT INCLUDE:**
1. Companies already in database: {blacklist}
2. API-only manufacturers (companies that only produce raw APIs, bulk drugs, active pharmaceutical ingredients, intermediates)
3. Raw material suppliers or chemical manufacturers
4. Importers, distributors or trading companies
5. Marketing authorization holders who don't manufacture
6. Contract manufacturers unless they own the manufacturing facility
7. Companies that manufacture FDF/FDC but DO NOT use the API '{api}' in their products
8. Companies whose products contain different APIs but not '{api}'

**MANDATORY INCLUSION CRITERIA - ONLY INCLUDE IF ALL ARE TRUE:**
1. Company manufactures FINISHED DOSAGE FORMS (tablets, capsules, injections, etc.) in {country}
2. The finished product MUST contain '{api}' as an ingredient (either alone or in combination)
3. Manufacturing facility is located in {country}
4. You have DIRECT EVIDENCE that their product contains '{api}'
5. You can provide the exact product name/brand that contains '{api}'

**VERIFICATION REQUIREMENTS:**
For each company, you MUST provide:
- Exact product name/brand containing '{api}'
- Direct evidence (product label, regulatory approval, official product listing)
- Manufacturing location in {country}

**MANDATORY PRE-SUBMISSION CHECKLIST:**
Before including ANY company, verify:
□ Company is NOT in exclusion list: {blacklist}
□ Company is NOT an API-only manufacturer
□ Company is NOT just an importer/distributor
□ Company DOES manufacture FDF/FDC in {country}
□ You have DIRECT evidence of '{api}' in their product
□ Confidence level is 90% or higher

**CRITICAL WARNING:** 
- If you cannot find direct evidence that a company's product contains '{api}', DO NOT include them
- If you're unsure whether a company uses '{api}' in their products, DO NOT include them
- If a company only makes raw '{api}' but not finished products, DO NOT include them
- When in doubt, EXCLUDE the company

**OUTPUT FORMAT (MANDATORY):**
Return ONLY a markdown table with these exact columns:
| Company | Form | Strength | Additional Info |

**EXAMPLES OF WHAT TO INCLUDE:**
- "ABC Pharma manufactures Loxoprofen 60mg tablets (brand: LoxoTab) in their Mumbai facility"
- "XYZ Labs produces Loxoprofen + Paracetamol combination tablets in Bangladesh"

**EXAMPLES OF WHAT TO EXCLUDE:**
- "DEF Chemicals supplies raw Loxoprofen sodium to other manufacturers"
- "GHI Trading imports Loxoprofen tablets from other countries"
- "JKL Pharma manufactures various pain relief tablets" (without confirming they use '{api}')

**FINAL INSTRUCTION:**
If you cannot find companies that meet ALL the above criteria with 90%+ confidence, return only the table headers. Quality over quantity - it's better to return no results than incorrect results.

"""
        return prompt

    def build_simple_groq_prompt(self, api: str, country: str, existing_companies: list) -> str:
        blacklist = ", ".join(existing_companies) if existing_companies else "None"
        return f"""
**PHARMACEUTICAL MANUFACTURER SEARCH (GROQ)**

Find companies in {country} that might manufacture finished dosage forms (e.g., tablets, capsules, injections) that could contain '{api}' as an ingredient. 

Focus on identifying potential manufacturers. Exclude companies explicitly known to be only API suppliers, distributors, or already in the database: {blacklist}.

**OUTPUT FORMAT (MANDATORY):**
Return ONLY a markdown table with these exact columns:
| Company | Form | Strength | Additional Info |

If you cannot find any potential companies, return only the table headers.
"""

    def build_fallback_prompt(self, api: str, country: str, existing_companies: list) -> str:
        blacklist = ", ".join(existing_companies) if existing_companies else "None"
        return f"""
**SIMPLIFIED PHARMACEUTICAL SEARCH**

Find companies in {country} that manufacture tablets, capsules or other finished medicines containing '{api}' as an ingredient.

**STRICT EXCLUSIONS:**
- Already listed companies: {blacklist}
- Raw material/API suppliers
- Importers or distributors
- Companies that don't use '{api}' in their products

**REQUIREMENTS:**
- Must manufacture finished medicines in {country}
- Must use '{api}' in their products
- Must provide product name containing '{api}'
- Must have 90%+ confidence

**VERIFICATION NEEDED:**
- Product name with '{api}'
- Manufacturing location in {country}
- Not just raw material supplier

**FORMAT:**
| Company | Form | Strength | Additional Info |

**CRITICAL:** Only include if you can prove they use '{api}' in finished products. If unsure, exclude.
"""

    def enhanced_parse_markdown_table(self, markdown: str) -> pd.DataFrame:
        """Enhanced markdown table parser with better error handling"""
        if not markdown or markdown.strip() == "":
            logger.warning("⚠️ Empty response from agent")
            return pd.DataFrame()
        
        logger.info(f"📝 Raw agent response length: {len(markdown)} characters")
        
        # Find table section
        lines = markdown.strip().split("\n")
        table_lines = []
        in_table = False
        
        for line in lines:
            if "|" in line and not line.strip().startswith("```"):
                table_lines.append(line.strip())
                in_table = True
            elif in_table and line.strip() == "":
                continue
            elif in_table and "|" not in line:
                break
        
        if len(table_lines) < 2:
            logger.warning("⚠️ No valid table found in response")
            return pd.DataFrame()
        
        # Parse headers
        header_line = table_lines[0]
        headers = [h.strip() for h in header_line.split("|")[1:-1]]
        
        # Skip separator line and parse data
        data_lines = table_lines[2:] if len(table_lines) > 2 else []
        
        if not data_lines:
            logger.warning("⚠️ No data rows found in table")
            return pd.DataFrame()
        
        data = []
        for i, row in enumerate(data_lines):
            cols = [c.strip() for c in row.split("|")[1:-1]]
            if len(cols) == len(headers):
                data.append(cols)
            else:
                logger.warning(f"⚠️ Row {i+1} has {len(cols)} columns, expected {len(headers)}")
        
        if not data:
            logger.warning("⚠️ No valid data rows parsed")
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=pd.Index(headers))
        logger.info(f"✅ Parsed {len(df)} rows from table")
        return df

    def validate_and_filter_results(self, df: pd.DataFrame, api: str) -> pd.DataFrame:
        """Enhanced validation with API-specific filtering"""
        if df.empty:
            return df
        
        logger.info(f"🔍 Validating {len(df)} rows for API '{api}'...")
        
        # Check required columns. Simplified.
        required_columns = ['Company', 'Form', 'Strength', 'Additional Info']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"⚠️ Missing columns: {missing_columns}")
            return pd.DataFrame()
        
        # Filter valid rows
        valid_rows = []
        for idx, row in df.iterrows():
            company_name = str(row.get('Company', '')).strip()
            if not company_name:
                logger.warning(f"❌ Row {idx}: Missing company name")
                continue
            
            # Removed confidence check as it's no longer in the requested output
            # Removed API-only detection check - rely on agent to filter this
            # Removed importer/distributor check - rely on agent to filter this
            
            # Verify API is mentioned in product context (using Additional Info now)
            form = str(row.get('Form', '')).lower()
            strength = str(row.get('Strength', '')).lower()
            additional_info = str(row.get('Additional Info', '')).lower()
            
            api_lower = api.lower()
            api_mentioned = any(api_lower in field for field in [form, strength, additional_info])
            
            if not api_mentioned:
                logger.warning(f"❌ Row {idx}: API '{api}' not clearly mentioned in product details - {company_name}")
                continue
            
            # Removed URL validity check and manufacturing location check as they are not in the simplified output
            
            valid_rows.append(row)
            logger.info(f"✅ Row {idx}: Valid - {company_name}")
        
        if not valid_rows:
            logger.warning("❌ No valid rows found after filtering")
            return pd.DataFrame()
        
        validated_df = pd.DataFrame(valid_rows)
        logger.info(f"✅ {len(validated_df)} valid rows after filtering")
        return validated_df

    def is_verified_source(self, source: str, url: str) -> bool:
        source = source.lower()
        url = url.lower()
        return any(keyword in source or keyword in url for keyword in self.TRUSTED_SOURCES)

    def is_valid_row(self, row) -> bool:
        # Simplified validation for the new output format
        return all([
            pd.notna(row.get('Company')) and row['Company'].strip(),
            pd.notna(row.get('Form')) and row['Form'].strip(),
            pd.notna(row.get('Strength')) and row['Strength'].strip(),
            pd.notna(row.get('Additional Info')) and row['Additional Info'].strip()
        ])

    def clean_and_prepare_dataframe(self, df: pd.DataFrame, api: str, country: str) -> pd.DataFrame:
        df = df.copy()
        
        # Normalize column names
        column_mapping = {
            'Product Name': 'product_name',
            'Manufacturing Location': 'manufacturing_location',
            'Verification Source': 'verification_source',
            'Confidence (%)': 'confidence',
            'Additional Info': 'additional_info',
            'URL': 'url',
            'Company': 'company',
            'Form': 'form',
            'Strength': 'strength'
        }
        
        df.columns = [column_mapping.get(col, col.strip().lower().replace(" ", "_").replace("(%)", "")) for col in df.columns]
        
        now = pd.Timestamp.now()
        df['created_at'] = now
        df['updated_at'] = now
        df['api'] = api
        df['country'] = country

        if 'confidence' in df.columns:
            df['confidence'] = df['confidence'].apply(lambda x: int(str(x).strip('%')) if pd.notna(x) and str(x).strip('%').isdigit() else 0)
        else:
            df['confidence'] = 0

        # Ensure all required columns exist for insertion. Simplified to match DB schema.
        required_cols = ['company', 'form', 'strength', 'additional_info', 'api', 'country', 'verification_source', 'confidence', 'url']
        for col in required_cols:
            if col not in df.columns:
                if col == 'verification_source':
                    df[col] = 'Web Search'
                elif col == 'confidence':
                    df[col] = 80
                elif col == 'url':
                    df[col] = ''
                else:
                    df[col] = ''
            if col == 'confidence':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(80).astype(int)
            else:
                df[col] = df[col].fillna('').astype(str)

        return df

    def insert_into_viruj(self, df: pd.DataFrame, api: str, country: str):
        if df.empty:
            logger.warning("⚠️ No data to insert into viruj.")
            return pd.DataFrame()

        engine = self.get_db_engine()
        if not engine:
            return pd.DataFrame()

        newly_inserted = []

        try:
            df = self.clean_and_prepare_dataframe(df, api, country)

            insert_stmt = text("""
                INSERT OR IGNORE INTO viruj 
                (company, form, strength, verification_source, confidence, url, additional_info, created_at, updated_at, api, country)
                VALUES (:company, :form, :strength, :verification_source, :confidence, :url, :additional_info, :created_at, :updated_at, :api, :country)
            """)

            with engine.begin() as conn:
                for _, row in df.iterrows():
                    # Prepare safe params using _safe_val
                    company_name = self._safe_val(row.get('company', ''))
                    if not company_name or str(company_name).strip().lower() in ["unknown", ""]:
                        continue

                    params = {
                        'company': company_name,
                        'form': self._safe_val(row.get('form', '')),
                        'strength': self._safe_val(row.get('strength', 'Unknown')),
                        'verification_source': self._safe_val(row.get('verification_source', 'Web Search')),
                        'confidence': int(self._safe_val(row.get('confidence', 80))) if self._safe_val(row.get('confidence', 80)) is not None else 80,
                        'url': self._safe_val(row.get('url', '')),
                        'additional_info': self._safe_val(row.get('additional_info', '')),
                        'created_at': self._safe_val(row.get('created_at', pd.Timestamp.now().isoformat())),
                        'updated_at': self._safe_val(row.get('updated_at', pd.Timestamp.now().isoformat())),
                        'api': self._safe_val(row.get('api', api)),
                        'country': self._safe_val(row.get('country', country))
                    }

                    result = conn.execute(insert_stmt, params)
                    # Check if row was actually inserted (rowcount > 0 means it was inserted, 0 means duplicate/ignored)
                    if result.rowcount and result.rowcount > 0:
                        newly_inserted.append(row)

            if newly_inserted:
                logger.info(f"✅ Inserted {len(newly_inserted)} new rows into viruj.")
            else:
                logger.info("⚠️ No new companies inserted (all duplicates).")

            # Return DataFrame with proper column structure matching database schema
            if newly_inserted:
                # Create a list of dictionaries with the correct column names
                result_data = []
                for row in newly_inserted:
                    result_data.append({
                        'company': self._safe_val(row.get('company', '')),
                        'form': self._safe_val(row.get('form', '')),
                        'strength': self._safe_val(row.get('strength', 'Unknown')),
                        'additional_info': self._safe_val(row.get('additional_info', '')),
                        'api': self._safe_val(row.get('api', api)),
                        'country': self._safe_val(row.get('country', country))
                    })
                return pd.DataFrame(result_data)
            else:
                return pd.DataFrame(columns=['company', 'form', 'strength', 'additional_info', 'api', 'country'])
        except Exception as e:
            logger.error(f"❌ viruj insert error: {e}")
            return pd.DataFrame()

    def run_agent_openai(self, api: str, country: str, existing_companies: list) -> pd.DataFrame:
        logger.info(f"🤖 Running OpenAI agent for {api} in {country}...")
        
        system_prompt = f"""You are a pharmaceutical manufacturing expert specializing in identifying companies that manufacture FINISHED DOSAGE FORMS containing specific APIs. 

CRITICAL RULES:
1. Only include companies that manufacture finished medicines (tablets, capsules, injections, etc.)
2. The finished product MUST contain the specified API as an ingredient
3. Exclude API-only manufacturers, raw material suppliers, importers, distributors
4. Always verify the API is actually used in the company's finished products
5. Provide concrete evidence and sources
6. Minimum 90% confidence required
7. When in doubt, exclude the company

Focus on quality over quantity. Better to find 2-3 accurate companies than 10 questionable ones."""
        
        prompt = self.build_enhanced_prompt(api, country, existing_companies)
        
        try:
            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000,
                top_p=0.8
            )
            
            response = None
            if chat_completion and hasattr(chat_completion, 'choices') and chat_completion.choices and hasattr(chat_completion.choices[0], 'message'):
                response = chat_completion.choices[0].message.content
            logger.info(f"📄 OpenAI Agent Raw Response:\n{response}") # Added this line for debugging
            logger.info(f"📄 OpenAI Agent Response (truncated):\n{str(response)[:500]}...")
            
            if not response or response.strip() == "":
                logger.warning("⚠️ Empty response from OpenAI")
                return pd.DataFrame()
            
            df = self.enhanced_parse_markdown_table(response)
            if df.empty:
                logger.warning("❌ OpenAI agent returned no valid data")
                return pd.DataFrame()
            
            df_validated = self.validate_and_filter_results(df, api)
            logger.info(f"✅ OpenAI agent found {len(df_validated)} valid companies")
            
            return df_validated
            
        except Exception as e:
            logger.error(f"❌ OpenAI Agent failed: {e}")
            return pd.DataFrame()

    def google_search(self, query: str, num_results: int = 10) -> list:
        """Perform Google Custom Search"""
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.GOOGLE_API_KEY,
                "cx": self.GOOGLE_CSE_ID,
                "q": query,
                "num": num_results
            }
            res = requests.get(url, params=params, timeout=15)
            return res.json().get("items", [])
        except Exception as e:
            logger.error(f"❌ Google search failed: {e}")
            return []

    def analyze_with_groq(self, api_name: str, snippets: str, country_filter: str = None) -> str:
        """Analyze search results with Groq to extract manufacturers"""
        if not self.groq_client:
            raise RuntimeError("GROQ client is not configured")
        
        if country_filter:
            country_instruction = f"Focus ONLY on manufacturers located in {country_filter}. Exclude manufacturers from other countries."
        else:
            country_instruction = "Include manufacturers from any country."
        
        prompt = f"""
You are a pharmaceutical business intelligence expert. Extract FDF manufacturers for {api_name}.
{country_instruction}
Return ONLY a markdown table with columns:
| API Name | FDF Manufacturer | Country | Product Form | Source URL | Evidence |
Search Results:
{snippets}
"""
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1200
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Groq analysis failed: {e}")
            return ""

    def parse_groq_table(self, groq_output: str, api_name: str) -> pd.DataFrame:
        """Parse Groq markdown table output into DataFrame"""
        try:
            lines = groq_output.strip().split("\n")
            table_lines = []
            for line in lines:
                if "|" in line and not line.strip().startswith("```"):
                    table_lines.append(line.strip())
            
            if len(table_lines) < 2:
                return pd.DataFrame()
            
            header_line = table_lines[0]
            headers = [h.strip() for h in header_line.split("|")[1:-1]]
            data_rows = []
            
            for line in table_lines[2:]:
                if "|" in line and not line.strip().startswith("---"):
                    row_data = [cell.strip() for cell in line.split("|")[1:-1]]
                    if len(row_data) == len(headers) and any(cell.strip() for cell in row_data):
                        data_rows.append(row_data)
            
            if data_rows:
                df = pd.DataFrame(data_rows, columns=headers)
                if "API Name" in df.columns:
                    df["API Name"] = api_name
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Error parsing Groq table: {e}")
            return pd.DataFrame()

    def extract_companies_from_results(self, results: list, api_name: str, country_filter: str = None) -> pd.DataFrame:
        """Extract company names directly from search results as fallback"""
        try:
            companies_data = []
            for item in results:
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                text = f"{title} {snippet}".lower()
                
                company_patterns = [
                    r"([A-Z][a-zA-Z\s&]+(?:Pharma|Pharmaceuticals?|Labs|Laboratories?|Limited|Ltd|Inc|Corp|Corporation))",
                    r"([A-Z][a-zA-Z\s&]+(?:Manufacturing|Manufacturer|Producer))",
                    r"([A-Z][a-zA-Z\s&]+(?:Company|Co\.|Group))",
                    r"([A-Z][a-zA-Z\s&]+(?:Drugs|Medicines|Healthcare))",
                ]
                
                companies = []
                for pattern in company_patterns:
                    matches = re.findall(pattern, f"{title} {snippet}", re.IGNORECASE)
                    companies.extend(matches)
                
                companies = list(set([c.strip() for c in companies if len(c.strip()) > 3]))
                
                for company in companies[:2]:
                    country = "Unknown"
                    if country_filter:
                        country = country_filter
                    else:
                        country_indicators = ["india", "usa", "germany", "china", "uk", "canada", "france", "japan"]
                        for indicator in country_indicators:
                            if indicator in text:
                                country = indicator.title()
                                break
                    
                    companies_data.append({
                        "API Name": api_name,
                        "FDF Manufacturer": company,
                        "Country": country,
                        "Product Form": "Unknown",
                        "Source URL": link,
                        "Evidence": f"Found in: {title}"
                    })
            
            if companies_data:
                return pd.DataFrame(companies_data)
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Error extracting companies: {e}")
            return pd.DataFrame()

    def run_agent_groq(self, api: str, country: str, existing_companies: list) -> pd.DataFrame:
        logger.info(f"🤖 Running Groq agent for {api} in {country}...")
        
        system_prompt = f"""You are a pharmaceutical research assistant. Your task is to find companies that manufacture finished dosage forms.

Your response MUST be a markdown table with specific columns."""
        
        prompt = self.build_simple_groq_prompt(api, country, existing_companies)
        
        try:
            time.sleep(1) # Add delay to avoid rate limiting
            
            chat_completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2500,
                top_p=0.8
            )
            
            response = None
            if chat_completion and hasattr(chat_completion, 'choices') and chat_completion.choices and hasattr(chat_completion.choices[0], 'message'):
                response = chat_completion.choices[0].message.content
            logger.info(f"📄 Groq Agent Response (truncated):\n{str(response)[:500]}...")
            
            if not response or response.strip() == "":
                logger.warning("⚠️ Empty response from Groq")
                return pd.DataFrame()
            
            df = self.enhanced_parse_markdown_table(response)
            if df.empty:
                logger.warning("❌ Groq agent returned no valid data")
                return pd.DataFrame()
            
            df_validated = self.validate_and_filter_results(df, api)
            logger.info(f"✅ Groq agent found {len(df_validated)} valid companies")
            
            return df_validated
            
        except Exception as e:
            logger.error(f"❌ Groq Agent failed: {e}")
            return pd.DataFrame()

    def find_api_buyers(self, api: str, country: str):
        logger.info(f"\n🔍 Starting enhanced pharmaceutical research for {api} in {country}")
        logger.info("=" * 80)
        
        # Step 1: Fetch existing data
        existing_data_df = self.fetch_existing_data(api, country)
        
        # Ensure 'company' column exists before proceeding
        if 'company' not in existing_data_df.columns:
            logger.warning(f"'company' column not found in existing_data_df for API: {api}, Country: {country}. Initializing empty existing_companies list.")
            existing_companies = []
        else:
            existing_companies = [c for c in existing_data_df['company'].tolist()]
        
        if existing_data_df.empty:
            logger.info(f"\n📊 No existing companies found for {api} in {country}")
        else:
            logger.info(f"\n📊 Found {len(existing_data_df)} existing companies for {api} in {country}:")
            for i, company in enumerate(existing_data_df['company'].tolist(), 1):
                logger.info(f"  {i}. {company}")

        # Step 2: Perform Google Search (like Streamlit app)
        logger.info(f"\n{'='*25} GOOGLE SEARCH {'='*25}")
        if country:
            query = f'"{api}" API buyers "{country}" OR "{api}" FDF manufacturers "{country}" OR "{api}" pharmaceutical companies "{country}"'
        else:
            query = f"{api} API buyers OR {api} FDF manufacturers OR {api} pharmaceutical companies"
        
        try:
            results = self.google_search(query, num_results=10)
            logger.info(f"✅ Found {len(results)} Google search results")
        except Exception as e:
            logger.error(f"❌ Google search failed: {e}")
            results = []

        # Step 3: Build snippets for Groq analysis
        snippets = ""
        for item in results:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            snippets += f"\nTitle: {title}\nURL: {link}\nSnippet: {snippet}\n"

        # Step 4: Analyze with Groq (like Streamlit app)
        logger.info(f"\n{'='*25} GROQ ANALYSIS {'='*25}")
        df = pd.DataFrame()
        if snippets:
            try:
                groq_output = self.analyze_with_groq(api, snippets, country)
                if groq_output:
                    df = self.parse_groq_table(groq_output, api)
                    logger.info(f"✅ Groq analysis returned {len(df)} companies")
            except Exception as e:
                logger.warning(f"⚠️ Groq analysis failed: {e}")

        # Step 5: Fallback to direct extraction if Groq failed
        if df.empty:
            logger.info("⚠️ No structured data from Groq. Attempting direct extraction...")
            df = self.extract_companies_from_results(results, api, country)
            if not df.empty:
                logger.info(f"✅ Direct extraction found {len(df)} companies")

        # Step 6: Normalize column names to match expected format
        if not df.empty:
            # Rename columns to match what clean_and_prepare_dataframe expects
            df = df.rename(columns={
                "FDF Manufacturer": "Company",
                "Product Form": "Form",
                "Source URL": "URL",
                "Evidence": "Additional Info"
            })
            
            # Ensure required columns exist
            if "Company" not in df.columns:
                if "FDF Manufacturer" in df.columns:
                    df["Company"] = df["FDF Manufacturer"]
                else:
                    logger.warning("⚠️ No company column found in results")
                    df = pd.DataFrame()
            
            if "Form" not in df.columns:
                df["Form"] = "Unknown"
            if "Strength" not in df.columns:
                df["Strength"] = "Unknown"
            if "Additional Info" not in df.columns:
                df["Additional Info"] = ""

        # Step 7: Filter out existing companies
        if not df.empty and existing_companies:
            df = df[~df["Company"].str.lower().isin([c.lower() for c in existing_companies])]
            logger.info(f"📊 After filtering existing companies: {len(df)} companies remain")

        # Step 8: Apply validation
        if not df.empty:
            final_validated = self.validate_and_filter_results(df, api)
        else:
            final_validated = pd.DataFrame()

        if final_validated.empty:
            logger.warning("❌ No companies found or passed validation")
            return {
                "success": True,
                "existing_data": existing_data_df.to_dict(orient="records") if not existing_data_df.empty else [],
                "newly_found_companies": []
            }
        
        logger.info(f"✅ Found {len(final_validated)} validated companies")
        
        # Step 9: Insert validated results
        inserted_df = self.insert_into_viruj(final_validated, api, country)
        
        # Summary
        logger.info(f"\n📋 FINAL SUMMARY:")
        logger.info(f"  - API: {api}")
        logger.info(f"  - Country: {country}")
        logger.info(f"  - Existing companies: {len(existing_companies)}")
        logger.info(f"  - New companies found: {len(final_validated)}")
        logger.info(f"  - Companies inserted: {len(inserted_df)}")

        # Prepare return data - ensure both have same structure
        existing_data_list = []
        if not existing_data_df.empty:
            existing_data_list = existing_data_df.to_dict(orient="records")
            logger.info(f"📊 Returning {len(existing_data_list)} existing companies to frontend")
        
        newly_found_list = []
        if not inserted_df.empty:
            newly_found_list = inserted_df.to_dict(orient="records")
            logger.info(f"📊 Returning {len(newly_found_list)} newly found companies to frontend")
        
        # Return results for frontend consumption
        return {
            "success": True,
            "existing_data": existing_data_list,
            "newly_found_companies": newly_found_list
        }
