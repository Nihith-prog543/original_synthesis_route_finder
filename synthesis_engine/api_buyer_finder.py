# synthesis_engine/api_buyer_finder.py - API Buyer Search Logic
import pandas as pd
import logging
import warnings
import os
import re
import time
from sqlalchemy import create_engine, text
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import OpenAI
from groq import Groq

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy.*")

# ====== API KEYS LOADED FROM ENV ======
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional fallback
# ======================================

# Initialize clients
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except TypeError as exc:
        logger.warning("OpenAI client initialization failed (%s). Continuing without OpenAI.", exc)
        client = None
else:
    client = None

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

class ApiBuyerFinder:
    def __init__(self):
        # ====== SQLITE CONFIG ======
        # Use viruj.localdb database
        db_filename = os.getenv("SQLITE_DB_FILENAME")
        
        if db_filename:
            if not os.path.isabs(db_filename):
                db_filename = os.path.abspath(db_filename)
        else:
            # Try to find viruj.localdb in common locations
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_file_dir)
            
            # Priority 1: Check the manufactures_api location (where user is checking)
            manufactures_api_db = r"C:\Users\HP\Desktop\manufactures_api\viruj_local.db"
            if os.path.exists(manufactures_api_db):
                db_filename = manufactures_api_db
            else:
                # Priority 2: Look for viruj.localdb in project root
                viruj_localdb_path = os.path.join(project_root, "viruj.localdb")
                if os.path.exists(viruj_localdb_path):
                    db_filename = viruj_localdb_path
                    logger.info(f"📁 Found viruj.localdb in project directory")
                else:
                    # Priority 3: Look for viruj_local.db (alternative naming)
                    viruj_local_db_path = os.path.join(project_root, "viruj_local.db")
                    if os.path.exists(viruj_local_db_path):
                        db_filename = viruj_local_db_path
                    else:
                        # Default to manufactures_api location
                        db_filename = manufactures_api_db
        
        self.SQLITE_DB_FILENAME = db_filename
        self.SQLALCHEMY_SQLITE_URI = f"sqlite:///{self.SQLITE_DB_FILENAME}"
        # ============================

        # API keys
        self.GROQ_API_KEY = GROQ_API_KEY
        self.OPENAI_API_KEY = OPENAI_API_KEY
        
        self.groq_client = groq_client
        self.openai_client = client

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

    def get_db_engine(self):
        """
        Get database engine - PostgreSQL (Supabase) if DATABASE_URL is set, 
        otherwise fallback to SQLite for local development.
        """
        try:
            # Priority 1: Check for PostgreSQL connection (Supabase/cloud)
            database_url = os.getenv("DATABASE_URL")
            if database_url and database_url.startswith("postgresql://"):
                logger.info("🔗 Using PostgreSQL (Supabase) database")
                engine = create_engine(
                    database_url,
                    pool_size=5,
                    max_overflow=10,
                    pool_recycle=3600,
                    echo=False
                )
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("✅ PostgreSQL database connected.")
                return engine
            
            # Priority 2: Fallback to SQLite (local development)
            logger.info("🔗 Using SQLite database (local development)")
            uri = self.SQLALCHEMY_SQLITE_URI
            engine = create_engine(
                uri,
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600,
                echo=False,
                connect_args={"check_same_thread": False}
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ SQLite database connected.")
            return engine
        except Exception as e:
            logger.error(f"❌ DB connection failed: {e}")
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

        if not self.openai_client:
            return ""
        
        response = self.openai_client.chat.completions.create(
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

        query = text("""
            SELECT company, form, strength, additional_info 
            FROM viruj 
            WHERE api = :api AND country = :country
        """)

        with engine.begin() as conn:
            result = conn.execute(query, {"api": api, "country": country})
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame(columns=['Company', 'Form', 'Strength', 'Additional Info'])

        df = pd.DataFrame(rows, columns=pd.Index(['Company', 'Form', 'Strength', 'Additional Info']))
        # Add api and country columns for frontend compatibility
        df['API'] = api
        df['Country'] = country
        return df

    def fetch_existing_companies(self, api: str, country: str):
        engine = self.get_db_engine()
        if not engine:
            return []

        query = text("SELECT DISTINCT company FROM viruj WHERE api = :api AND country = :country")
        with engine.begin() as conn:
            result = conn.execute(query, {"api": api, "country": country})
            return [row[0] for row in result.fetchall()]

    def build_enhanced_prompt(self, api: str, country: str, existing_companies: list, agent: str) -> str:
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

□ You have DIRECT EVIDENCE of '{api}' in their product

□ Confidence level is 90% or higher

**CRITICAL WARNING:** 

- If you cannot find direct evidence that a company's product contains '{api}', DO NOT include them

- If you're unsure whether a company uses '{api}' in their products, DO NOT include them

- If a company only makes raw '{api}' but not finished products, DO NOT include them

- When in doubt, EXCLUDE the company

**OUTPUT FORMAT (MANDATORY):**

Return ONLY a markdown table with these exact columns:

| Company | Product Name | Form | Strength | Manufacturing Location | Verification Source | Confidence (%) | URL | Additional Info |

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

| Company | Product Name | Form | Strength | Manufacturing Location | Verification Source | Confidence (%) | URL | Additional Info |

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

| Company | Product Name | Form | Strength | Manufacturing Location | Verification Source | Confidence (%) | URL | Additional Info |

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
        
        # Check required columns
        required_columns = ['Company', 'Form', 'Confidence (%)', 'URL']
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
            
            # Check confidence level
            try:
                confidence_str = str(row.get('Confidence (%)', '0')).strip().rstrip('%')
                confidence = int(confidence_str) if confidence_str.isdigit() else 0
                if confidence < 50:  # Relaxed from 90% to 50%
                    logger.warning(f"❌ Row {idx}: Low confidence ({confidence}%) - minimum 50% required")
                    continue
            except:
                logger.warning(f"❌ Row {idx}: Invalid confidence format")
                continue
            
            # Enhanced API-only detection
            company_lower = company_name.lower()
            additional_info = str(row.get('Additional Info', '')).lower()
            verification_source = str(row.get('Verification Source', '')).lower()
            
            # Expanded API-only keywords
            api_only_keywords = [
                'api manufacturer', 'api supplier', 'bulk drug', 'raw material', 'intermediate', 
                'active ingredient', 'pharmaceutical ingredient', 'chemical supplier', 'bulk supplier',
                'api only', 'raw api', 'bulk api', 'chemical manufacturer', 'ingredient supplier'
            ]
            
            combined_text = f"{company_lower} {additional_info} {verification_source}"
            if any(keyword in combined_text for keyword in api_only_keywords):
                logger.warning(f"❌ Row {idx}: API-only manufacturer detected - {company_name}")
                continue
            
            # Check for importer/distributor keywords
            import_keywords = [
                'importer', 'distributor', 'trading', 'import', 'distribution', 'wholesale',
                'trader', 'importing', 'export', 'exporter'
            ]
            
            if any(keyword in combined_text for keyword in import_keywords):
                logger.warning(f"❌ Row {idx}: Importer/Distributor detected - {company_name}")
                continue
            
            # Verify API is mentioned in product context
            product_name = str(row.get('Product Name', '')).lower()
            form = str(row.get('Form', '')).lower()
            strength = str(row.get('Strength', '')).lower()
            
            api_lower = api.lower()
            api_mentioned = any(api_lower in field for field in [product_name, form, strength, verification_source])
            
            if not api_mentioned:
                logger.warning(f"❌ Row {idx}: API '{api}' not clearly mentioned in product details - {company_name}")
                continue
            
            # Check URL validity
            url = str(row.get('URL', '')).strip()
            if not url or not (url.startswith('http://') or url.startswith('https://')):
                logger.warning(f"❌ Row {idx}: Invalid or missing URL - {company_name}")
                continue
            
            # Check manufacturing location if available
            if 'Manufacturing Location' in row:
                location = str(row.get('Manufacturing Location', '')).lower()
                if location and 'import' in location:
                    logger.warning(f"❌ Row {idx}: Manufacturing location indicates import - {company_name}")
                    continue
            
            valid_rows.append(row)
            logger.info(f"✅ Row {idx}: Valid - {company_name}")
        
        if not valid_rows:
            logger.warning("❌ No valid rows found after filtering")
            return pd.DataFrame()
        
        validated_df = pd.DataFrame(valid_rows)
        logger.info(f"✅ {len(validated_df)} valid rows after filtering")
        return validated_df

    def extract_urls(self, url_field):
        # Handles both plain URLs and markdown links
        urls = []
        # Find markdown links: [text](url)
        for match in re.findall(r'\[.*?\]\((https?://[^\)]+)\)', url_field):
            urls.append(match.strip())
        # Find plain URLs
        for match in re.findall(r'(https?://[^\s,]+)', url_field):
            if match.strip() not in urls:
                urls.append(match.strip())
        return urls

    def has_two_distinct_trusted_urls(self, row) -> bool:
        url_field = str(row.get('URL', ''))
        urls = self.extract_urls(url_field)
        if len(urls) < 2:
            return False
        trusted = [any(ts in u for ts in self.TRUSTED_SOURCES) for u in urls]
        return sum(trusted) >= 2 and len(set(urls)) >= 2

    def is_verified_source(self, source: str, url: str) -> bool:
        source = source.lower()
        url = url.lower()
        return any(keyword in source or keyword in url for keyword in self.TRUSTED_SOURCES)

    def is_valid_row(self, row) -> bool:
        try:
            confidence = row.get('Confidence (%)', '').strip().rstrip('%')
            confidence = int(confidence) if confidence.isdigit() else 0
        except:
            confidence = 0
        return all([
            pd.notna(row.get('Company')) and row['Company'].strip(),
            pd.notna(row.get('Form')) and row['Form'].strip(),
            pd.notna(row.get('URL')) and row['URL'].strip(),
            confidence >= 90,  # Increased from 80 to 90
            self.is_verified_source(row.get('Verification Source', ''), row['URL'])
        ])

    def is_api_only(self, row) -> bool:
        # Check company name, additional info, or verification source for API-only keywords
        text_fields = [
            row.get('Company', ''),
            row.get('Additional Info', ''),
            row.get('Verification Source', '')
        ]
        combined = " ".join([str(f).lower() for f in text_fields])
        return any(keyword in combined for keyword in self.API_ONLY_KEYWORDS)

    def is_double_verified(self, row) -> bool:
        sources = (row.get('Verification Source', '') + " " + row.get('URL', '')).lower()
        count = sum(kw in sources for kw in self.TRUSTED_SOURCES)
        # Require at least two different trusted sources
        return count >= 2

    def clean_and_prepare_dataframe(self, df: pd.DataFrame, api: str, country: str) -> pd.DataFrame:
        df = df.copy()
        
        # Normalize column names
        column_mapping = {
            'Product Name': 'product_name',
            'Manufacturing Location': 'manufacturing_location'
        }
        
        df.columns = [column_mapping.get(col, col.strip().lower().replace(" ", "_").replace("(%)", "")) for col in df.columns]
        
        # Convert timestamp to string format for SQLite
        from datetime import datetime
        now_str = datetime.now().isoformat()
        df['created_at'] = now_str
        df['updated_at'] = now_str
        df['api'] = api
        df['country'] = country
        if 'confidence' in df.columns:
            df['confidence'] = df['confidence'].apply(lambda x: int(str(x).strip('%')) if pd.notna(x) and str(x).strip('%').isdigit() else 0)
        elif 'confidence_' in df.columns:
            df['confidence'] = df['confidence_'].apply(lambda x: int(str(x).strip('%')) if pd.notna(x) and str(x).strip('%').isdigit() else 0)
        else:
            df['confidence'] = 0
        # Ensure all required columns exist
        required_cols = ['company', 'form', 'strength', 'verification_source', 'url', 'additional_info', 'product_name']
        for col in required_cols:
            if col not in df.columns:
                df[col] = ''
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
            with engine.begin() as conn:
                for idx, row in df.iterrows():
                    # Extract scalar values from the row (not Series objects)
                    # Use .iloc or direct access to get scalar values
                    insert_data = {
                        'company': str(row['company']) if pd.notna(row.get('company')) else '',
                        'form': str(row['form']) if pd.notna(row.get('form')) else '',
                        'strength': str(row['strength']) if pd.notna(row.get('strength')) else '',
                        'verification_source': str(row['verification_source']) if pd.notna(row.get('verification_source')) else '',
                        'confidence': int(row['confidence']) if pd.notna(row.get('confidence')) else 0,
                        'url': str(row['url']) if pd.notna(row.get('url')) else '',
                        'additional_info': str(row['additional_info']) if pd.notna(row.get('additional_info')) else '',
                        'created_at': str(row['created_at']) if pd.notna(row.get('created_at')) else '',
                        'updated_at': str(row['updated_at']) if pd.notna(row.get('updated_at')) else '',
                        'api': str(row['api']) if pd.notna(row.get('api')) else api,
                        'country': str(row['country']) if pd.notna(row.get('country')) else country
                    }
                    
                    # Check if record already exists
                    check_stmt = text("""
                        SELECT company FROM viruj 
                        WHERE api = :api AND country = :country AND company = :company
                    """)
                    existing = conn.execute(check_stmt, {
                        'api': insert_data['api'],
                        'country': insert_data['country'],
                        'company': insert_data['company']
                    }).fetchone()
                    
                    if not existing:
                        insert_stmt = text("""
                            INSERT INTO viruj (company, form, strength, verification_source, confidence, url, additional_info, created_at, updated_at, api, country)
                            VALUES (:company, :form, :strength, :verification_source, :confidence, :url, :additional_info, :created_at, :updated_at, :api, :country)
                        """)
                        conn.execute(insert_stmt, insert_data)
                        newly_inserted.append(row)
                        logger.info(f"✅ Inserted: {insert_data['company']}")
                    else:
                        logger.info(f"⚠️ Skipped duplicate: {insert_data['company']}")
            
            if newly_inserted:
                logger.info(f"✅ Inserted {len(newly_inserted)} new rows into viruj.")
            else:
                logger.info("⚠️ No new companies inserted (all duplicates).")
            return pd.DataFrame(newly_inserted)
        except Exception as e:
            logger.error(f"❌ viruj insert error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def run_agent_openai(self, api: str, country: str, existing_companies: list) -> pd.DataFrame:
        """Enhanced OpenAI agent with rate limit handling"""
        if not self.openai_client:
            logger.info("ℹ️ OpenAI client not configured; skipping OpenAI agent.")
            return pd.DataFrame()
        
        logger.info(f"🤖 Running OpenAI agent for {api} in {country}...")
        
        # Enhanced system prompt
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
        
        prompt = self.build_enhanced_prompt(api, country, existing_companies, "openai")
        
        try:
            # Add delay before OpenAI request to avoid rate limiting
            time.sleep(3)
            
            chat_completion = self.openai_client.chat.completions.create(
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
            logger.info(f"\n📄 OpenAI Agent Response:\n{str(response)[:500]}...")
            
            if not response or response.strip() == "":
                logger.warning("⚠️ Empty response from OpenAI")
                return pd.DataFrame()
            
            # Parse and validate results
            df = self.enhanced_parse_markdown_table(response)
            if df.empty:
                logger.warning("❌ OpenAI agent returned no valid data")
                return pd.DataFrame()
            
            # Apply strict validation
            df_validated = self.validate_and_filter_results(df, api)
            logger.info(f"✅ OpenAI agent found {len(df_validated)} valid companies")
            
            return df_validated
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "too many requests" in error_msg or "quota" in error_msg:
                logger.warning(f"⚠️ OpenAI Rate Limited: {e}")
                logger.info("🔄 Falling back to Groq-only mode for this iteration...")
                return pd.DataFrame()  # Return empty to trigger Groq fallback
            else:
                logger.warning(f"❌ OpenAI Agent failed: {e}")
                return pd.DataFrame()

    def run_agent_groq(self, api: str, country: str, existing_companies: list) -> pd.DataFrame:
        """Enhanced Groq agent with stricter validation"""
        if not self.groq_client:
            logger.info("ℹ️ Groq client not configured; skipping Groq agent.")
            return pd.DataFrame()
        
        logger.info(f"🤖 Running Groq agent for {api} in {country}...")
        
        # Enhanced system prompt
        system_prompt = f"""You are a pharmaceutical research assistant. Your task is to find companies that manufacture finished dosage forms.

Your response MUST be a markdown table with specific columns."""
        
        prompt = self.build_simple_groq_prompt(api, country, existing_companies)
        
        try:
            # Add delay to avoid rate limiting
            time.sleep(1)
            
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
            logger.info(f"\n📄 Groq Agent Raw Response:\n{response}")
            logger.info(f"\n📄 Groq Agent Response (truncated):\n{str(response)[:500]}...")
            
            if not response or response.strip() == "":
                logger.warning("⚠️ Empty response from Groq")
                return pd.DataFrame()
            
            # Parse and validate results
            df = self.enhanced_parse_markdown_table(response)
            if df.empty:
                logger.warning("❌ Groq agent returned no valid data")
                return pd.DataFrame()
            
            # Apply strict validation
            df_validated = self.validate_and_filter_results(df, api)
            logger.info(f"✅ Groq agent found {len(df_validated)} valid companies")
            
            return df_validated
            
        except Exception as e:
            logger.warning(f"❌ Groq Agent failed: {e}")
            return pd.DataFrame()

    def run_all_agents(self, api: str, country: str):
        """Enhanced main function with stricter validation and better reporting"""
        logger.info(f"\n🔍 Starting enhanced pharmaceutical research for {api} in {country}")
        logger.info("=" * 80)
        
        # Step 1: Fetch existing data
        existing_data_df = self.fetch_existing_data(api, country)
        
        # Ensure 'Company' column exists before proceeding
        if 'Company' not in existing_data_df.columns:
            logger.warning(f"'Company' column not found in existing_data_df for API: {api}, Country: {country}. Initializing empty existing_companies list.")
            existing_companies = []
        else:
            existing_companies = [c for c in existing_data_df['Company'].tolist()]
        
        if existing_data_df.empty:
            logger.info(f"\n📊 No existing companies found for {api} in {country}")
        else:
            logger.info(f"\n📊 Found {len(existing_data_df)} existing companies for {api} in {country}:")
            for i, company in enumerate(existing_data_df['Company'].tolist(), 1):
                logger.info(f"  {i}. {company}")
        
        # Step 2: Run agents with enhanced validation
        all_results = []
        
        # Run Groq Agent first (primary)
        logger.info(f"\n{'='*25} GROQ AGENT {'='*25}")
        df_groq = self.run_agent_groq(api, country, existing_companies)
        if not df_groq.empty:
            all_results.append(("Groq", df_groq))
        
        # Run OpenAI Agent as fallback
        logger.info(f"\n{'='*25} OPENAI AGENT {'='*25}")
        df_openai = self.run_agent_openai(api, country, existing_companies)
        if not df_openai.empty:
            all_results.append(("OpenAI", df_openai))
        
        # Step 3: Process results with enhanced validation
        logger.info(f"\n{'='*25} RESULTS SUMMARY {'='*25}")
        
        if not all_results:
            logger.warning("❌ No valid results from any agent")
            return pd.DataFrame()
        
        # Combine all results
        all_combined = pd.concat([df for _, df in all_results], ignore_index=True)
        logger.info(f"📊 Total companies found: {len(all_combined)}")
        
        # Remove duplicates based on company name
        all_combined = all_combined.drop_duplicates(subset=['Company'], keep='first')
        logger.info(f"📊 Unique companies after deduplication: {len(all_combined)}")
        
        # Apply final validation
        final_validated = self.validate_and_filter_results(all_combined, api)
        
        if final_validated.empty:
            logger.warning("❌ No companies passed final validation")
            logger.info("\n💡 This could mean:")
            logger.info("  - No suitable manufacturers found")
            logger.info("  - All candidates were API-only manufacturers")
            logger.info("  - All candidates were importers/distributors")
            logger.info("  - Confidence levels were too low")
            logger.info("  - No verifiable sources found")
            return pd.DataFrame()
        
        logger.info(f"✅ Found {len(final_validated)} validated companies:")
        
        # Display results
        display_cols = ['Company', 'Product Name', 'Form', 'Strength', 'Verification Source', 'Confidence (%)', 'URL']
        available_cols = [col for col in display_cols if col in final_validated.columns]
        
        if available_cols:
            display_df = final_validated[available_cols]
            logger.info("\n" + "="*80)
            logger.info(display_df.to_string(index=False))
            logger.info("="*80)
        
        # Insert validated results
        inserted_df = self.insert_into_viruj(final_validated, api, country)
        if not inserted_df.empty:
            logger.info(f"\n✅ Successfully inserted {len(inserted_df)} new companies into database")
        else:
            logger.info("\n⚠️ No new companies inserted (possibly duplicates)")
        
        # Summary
        logger.info(f"\n📋 FINAL SUMMARY:")
        logger.info(f"  - API: {api}")
        logger.info(f"  - Country: {country}")
        logger.info(f"  - Existing companies: {len(existing_companies)}")
        logger.info(f"  - New companies found: {len(final_validated)}")
        logger.info(f"  - Companies inserted: {len(inserted_df)}")
        return final_validated

    def find_api_buyers(self, api: str, country: str):
        """Main entry point for finding API buyers - compatible with Flask app"""
        logger.info(f"\n🔍 Starting pharmaceutical research for {api} in {country}")
        logger.info("=" * 80)
        
        # Step 1: Fetch existing data
        existing_data_df = self.fetch_existing_data(api, country)
        
        # Ensure 'Company' column exists before proceeding
        if 'Company' not in existing_data_df.columns:
            logger.warning(f"'Company' column not found in existing_data_df for API: {api}, Country: {country}. Initializing empty existing_companies list.")
            existing_companies = []
        else:
            existing_companies = [c for c in existing_data_df['Company'].tolist()]
        
        if existing_data_df.empty:
            logger.info(f"\n📊 No existing companies found for {api} in {country}")
        else:
            logger.info(f"\n📊 Found {len(existing_data_df)} existing companies for {api} in {country}:")
            for i, company in enumerate(existing_data_df['Company'].tolist(), 1):
                logger.info(f"  {i}. {company}")
        
        # Step 2: Run agents with enhanced validation
        all_results = []
        
        # Run Groq Agent first (primary)
        logger.info(f"\n{'='*25} GROQ AGENT {'='*25}")
        df_groq = self.run_agent_groq(api, country, existing_companies)
        if not df_groq.empty:
            all_results.append(("Groq", df_groq))
        
        # Run OpenAI Agent as fallback
        logger.info(f"\n{'='*25} OPENAI AGENT {'='*25}")
        df_openai = self.run_agent_openai(api, country, existing_companies)
        if not df_openai.empty:
            all_results.append(("OpenAI", df_openai))
        
        # Step 3: Process results with enhanced validation
        logger.info(f"\n{'='*25} RESULTS SUMMARY {'='*25}")
        
        if not all_results:
            logger.warning("❌ No valid results from any agent")
            return {
                "success": True,
                "existing_data": existing_data_df.to_dict(orient="records") if not existing_data_df.empty else [],
                "newly_found_companies": []
            }
        
        # Combine all results
        all_combined = pd.concat([df for _, df in all_results], ignore_index=True)
        logger.info(f"📊 Total companies found: {len(all_combined)}")
        
        # Remove duplicates based on company name
        all_combined = all_combined.drop_duplicates(subset=['Company'], keep='first')
        logger.info(f"📊 Unique companies after deduplication: {len(all_combined)}")
        
        # Apply final validation
        final_validated = self.validate_and_filter_results(all_combined, api)
        
        if final_validated.empty:
            logger.warning("❌ No companies passed final validation")
            logger.info("\n💡 This could mean:")
            logger.info("  - No suitable manufacturers found")
            logger.info("  - All candidates were API-only manufacturers")
            logger.info("  - All candidates were importers/distributors")
            logger.info("  - Confidence levels were too low")
            logger.info("  - No verifiable sources found")
            return {
                "success": True,
                "existing_data": existing_data_df.to_dict(orient="records") if not existing_data_df.empty else [],
                "newly_found_companies": []
            }
        
        logger.info(f"✅ Found {len(final_validated)} validated companies:")
        
        # Convert final_validated to API response format BEFORE insertion
        # This ensures frontend gets the data even if insertion fails
        discovered_companies = []
        if not final_validated.empty:
            # Prepare data for frontend display - convert to lowercase keys
            for _, row in final_validated.iterrows():
                company_dict = {
                    'company': str(row.get('Company', '')),
                    'form': str(row.get('Form', '')),
                    'strength': str(row.get('Strength', '')),
                    'additional_info': str(row.get('Additional Info', '')),
                    'url': str(row.get('URL', '')),
                    'api': api,
                    'country': country
                }
                discovered_companies.append(company_dict)
        
        # Insert validated results (try to insert, but don't fail if it doesn't work)
        inserted_df = self.insert_into_viruj(final_validated, api, country)
        
        # Convert inserted records to API response format (use discovered_companies if insertion failed)
        newly_inserted = discovered_companies if discovered_companies else []
        if not inserted_df.empty:
            newly_inserted = []
            for _, row in inserted_df.iterrows():
                company_dict = {
                    'company': str(row.get('company', '')),
                    'form': str(row.get('form', '')),
                    'strength': str(row.get('strength', '')),
                    'additional_info': str(row.get('additional_info', '')),
                    'api': str(row.get('api', api)),
                    'country': str(row.get('country', country))
                }
                newly_inserted.append(company_dict)
        
        # Summary
        logger.info(f"\n📋 FINAL SUMMARY:")
        logger.info(f"  - API: {api}")
        logger.info(f"  - Country: {country}")
        logger.info(f"  - Existing companies: {len(existing_companies)}")
        logger.info(f"  - New companies found: {len(final_validated)}")
        logger.info(f"  - Companies inserted: {len(inserted_df)}")
        
        # Convert existing_data to lowercase keys for frontend
        existing_data_formatted = []
        if not existing_data_df.empty:
            for _, row in existing_data_df.iterrows():
                existing_dict = {
                    'company': str(row.get('Company', '')),
                    'form': str(row.get('Form', '')),
                    'strength': str(row.get('Strength', '')),
                    'additional_info': str(row.get('Additional Info', '')),
                    'api': str(row.get('API', api)),
                    'country': str(row.get('Country', country))
                }
                existing_data_formatted.append(existing_dict)
        
        return {
            "success": True,
            "existing_data": existing_data_formatted,
            "newly_found_companies": newly_inserted,
            "discovered_companies": discovered_companies
        }
