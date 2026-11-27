"""
Data profiling module for auto-detecting column types, statistics, and generating
semantic mappings for source codes and group descriptions.
"""

import json
import re
from collections import Counter
from datetime import datetime
from typing import Any
import sqlite3


def detect_column_type(values: list[Any]) -> str:
    """Detect the type of a column based on its values."""
    non_null = [v for v in values if v is not None and v != '']
    if not non_null:
        return 'text'

    # Try numeric
    numeric_count = 0
    for v in non_null[:100]:
        try:
            float(str(v).replace(',', ''))
            numeric_count += 1
        except (ValueError, TypeError):
            pass

    if numeric_count / len(non_null[:100]) > 0.8:
        return 'number'

    # Try date (dd/mm/yyyy or similar)
    date_patterns = [
        r'^\d{2}/\d{2}/\d{4}$',  # dd/mm/yyyy
        r'^\d{4}-\d{2}-\d{2}$',  # yyyy-mm-dd
        r'^\d{2}-\d{2}-\d{4}$',  # dd-mm-yyyy
    ]
    date_count = 0
    for v in non_null[:100]:
        s = str(v).strip()
        if any(re.match(p, s) for p in date_patterns):
            date_count += 1

    if date_count / len(non_null[:100]) > 0.8:
        return 'date'

    return 'text'


def parse_amount(value: Any) -> float:
    """Parse amount string to float, handling commas and various formats."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(',', '').replace(' ', '')
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def parse_date(value: Any) -> str | None:
    """Parse date string to ISO format (YYYY-MM-DD)."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    # Try dd/mm/yyyy
    match = re.match(r'^(\d{2})[/-](\d{2})[/-](\d{4})$', s)
    if match:
        d, m, y = match.groups()
        return f"{y}-{m}-{d}"

    # Try yyyy-mm-dd
    match = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', s)
    if match:
        return s

    # Try dd-mm-yyyy
    match = re.match(r'^(\d{2})-(\d{2})-(\d{4})$', s)
    if match:
        d, m, y = match.groups()
        return f"{y}-{m}-{d}"

    return None


def profile_records(records: list[dict]) -> dict:
    """Generate comprehensive profile of records."""
    if not records:
        return {}

    columns = list(records[0].keys())
    profile = {
        'record_count': len(records),
        'columns': {},
        'date_range': {'min': None, 'max': None},
        'distinct_counts': {},
        'top_values': {},
        'numeric_stats': {}
    }

    # Collect values per column
    col_values = {col: [] for col in columns}
    for rec in records:
        for col in columns:
            col_values[col].append(rec.get(col))

    for col in columns:
        values = col_values[col]
        non_null = [v for v in values if v is not None and v != '']
        null_count = len(values) - len(non_null)

        col_type = detect_column_type(non_null)

        col_profile = {
            'type': col_type,
            'total': len(values),
            'null_count': null_count,
            'distinct_count': len(set(str(v) for v in non_null)),
        }

        if col_type == 'number':
            nums = [parse_amount(v) for v in non_null]
            if nums:
                col_profile.update({
                    'min': min(nums),
                    'max': max(nums),
                    'sum': sum(nums),
                    'avg': sum(nums) / len(nums),
                })
                profile['numeric_stats'][col] = col_profile

        elif col_type == 'date':
            dates = [parse_date(v) for v in non_null if parse_date(v)]
            if dates:
                col_profile['min'] = min(dates)
                col_profile['max'] = max(dates)
                if profile['date_range']['min'] is None or col_profile['min'] < profile['date_range']['min']:
                    profile['date_range']['min'] = col_profile['min']
                if profile['date_range']['max'] is None or col_profile['max'] > profile['date_range']['max']:
                    profile['date_range']['max'] = col_profile['max']

        # Top values (for categorical columns)
        if col_type == 'text' and non_null:
            counter = Counter(str(v) for v in non_null)
            profile['top_values'][col] = counter.most_common(10)
            profile['distinct_counts'][col] = col_profile['distinct_count']

        profile['columns'][col] = col_profile

    return profile


def infer_source_roles(records: list[dict], profile: dict) -> dict:
    """Infer semantic roles for Source codes using LLM-friendly analysis."""
    source_col = 'Source'
    if source_col not in profile['columns']:
        # Try to find source-like column
        for col in profile['columns']:
            if col.lower() in ('source', 'srctype', 'vtype', 'transtype'):
                source_col = col
                break

    if source_col not in profile['columns']:
        return {}

    top_sources = profile['top_values'].get(source_col, [])
    source_roles = {}

    # Heuristic mapping based on common patterns
    for source, count in top_sources:
        source_upper = str(source).upper().strip()

        # Get sample records for this source
        source_records = [r for r in records if str(r.get(source_col, '')).upper().strip() == source_upper]
        dr_total = sum(parse_amount(r.get('LocalDrAmt', r.get('DrAmt', 0))) for r in source_records)
        cr_total = sum(parse_amount(r.get('LocalCrAmt', r.get('CrAmt', 0))) for r in source_records)

        # Sample narrations
        narrations = [str(r.get('Narration', r.get('Nar', ''))) for r in source_records[:5] if r.get('Narration') or r.get('Nar')]

        # Canonical accounting transaction mapping. Keep screenshot/user-instructed codes exact.
        role = 'other'
        if source_upper in ('OB', 'OPENING', 'OPENING BALANCE', 'OBAL'):
            role = 'opening_balance'
        elif source_upper in ('SB', 'SALES', 'SALE', 'INV', 'INVOICE', 'SI'):
            role = 'sales'
        elif source_upper == 'PB' or source_upper in ('PURCHASE', 'PUR', 'PI', 'PURCHASE INVOICE'):
            role = 'purchase'
        elif source_upper == 'PR' or source_upper in ('PURCHASE RETURN', 'PURCHASE RETURNS', 'PRN'):
            role = 'purchase_return'
        elif source_upper == 'SR' or source_upper in ('SALES RETURN', 'SALES RETURNS', 'SRN'):
            role = 'sales_return'
        elif source_upper in ('CB', 'COLLECTION', 'RECEIPT', 'RECV', 'RCPT', 'CR'):
            role = 'cash_bank' if source_upper == 'CB' else 'collection'
        elif source_upper in ('PAY', 'PAYMENT', 'PMT', 'PYMT'):
            role = 'payment'
        elif source_upper in ('JV', 'JOURNAL', 'JRNL', 'JE'):
            role = 'journal'
        elif source_upper in ('CN', 'CREDIT NOTE', 'CRNOTE'):
            role = 'credit_note'
        elif source_upper in ('DN', 'DEBIT NOTE', 'DRNOTE'):
            role = 'debit_note'
        elif source_upper == 'ISB':
            role = 'branch_transfer'
        elif source_upper == 'IPB':
            role = 'branch_received'
        elif dr_total > 0 and cr_total == 0:
            role = 'debit_entry'
        elif cr_total > 0 and dr_total == 0:
            role = 'credit_entry'

        source_roles[source] = {
            'role': role,
            'count': count,
            'dr_total': dr_total,
            'cr_total': cr_total,
            'sample_narrations': narrations[:3]
        }

    return source_roles


def infer_group_kinds(records: list[dict], profile: dict) -> dict:
    """Infer semantic kinds for Group descriptions."""
    group_col = 'GrpDesc'
    if group_col not in profile['columns']:
        for col in profile['columns']:
            if col.lower() in ('grpdesc', 'group', 'groupdesc', 'acctgroup', 'accountgroup'):
                group_col = col
                break

    if group_col not in profile['columns']:
        return {}

    top_groups = profile['top_values'].get(group_col, [])
    group_kinds = {}

    for group, count in top_groups:
        group_upper = str(group).upper().strip()

        # Get sample records
        group_records = [r for r in records if str(r.get(group_col, '')).upper().strip() == group_upper]
        dr_total = sum(parse_amount(r.get('LocalDrAmt', r.get('DrAmt', 0))) for r in group_records)
        cr_total = sum(parse_amount(r.get('LocalCrAmt', r.get('CrAmt', 0))) for r in group_records)

        # Heuristic inference based on common accounting group names
        kind = 'other'
        group_lower = group_upper.lower()

        if any(kw in group_lower for kw in ['asset', 'current asset', 'fixed asset', 'cash', 'bank', 'debtor', 'receivable', 'inventory', 'stock']):
            kind = 'asset'
        elif any(kw in group_lower for kw in ['liability', 'liabilities', 'current liability', 'current liabilities', 'creditor', 'creditors', 'payable', 'payables', 'loan', 'borrowing']):
            kind = 'liability'
        elif any(kw in group_lower for kw in ['equity', 'capital', 'reserve', 'retained earning', "owner's equity"]):
            kind = 'equity'
        elif any(kw in group_lower for kw in ['income', 'revenue', 'sales', 'turnover', 'fee', 'interest income', 'other income']):
            kind = 'income'
        elif any(kw in group_lower for kw in ['expense', 'expenditure', 'cost', 'purchase', 'salary', 'rent', 'interest expense', 'depreciation']):
            kind = 'expense'
        elif dr_total > cr_total * 2:
            kind = 'asset'
        elif cr_total > dr_total * 2:
            kind = 'liability'

        group_kinds[group] = {
            'kind': kind,
            'count': count,
            'dr_total': dr_total,
            'cr_total': cr_total
        }

    return group_kinds


def build_schema_description(profile: dict, source_mapping: dict, group_mapping: dict) -> str:
    """Build a schema description string for the LLM system prompt."""
    lines = []
    lines.append("DATABASE SCHEMA: Table `data` with the following columns:")

    for col, info in profile['columns'].items():
        col_type = info['type']
        nullable = "NULLABLE" if info['null_count'] > 0 else "NOT NULL"
        distinct = info['distinct_count']

        extra = ""
        if col_type == 'number':
            extra = f" (min={info.get('min', 0):.2f}, max={info.get('max', 0):.2f}, sum={info.get('sum', 0):.2f})"
        elif col_type == 'date':
            extra = f" (range: {info.get('min', '')} to {info.get('max', '')})"

        # Add semantic info for known columns
        semantic = ""
        if col == 'Source':
            roles = {k: v['role'] for k, v in source_mapping.items()}
            if roles:
                semantic = f" [Source codes: {roles}]"
        elif col == 'GrpDesc':
            kinds = {k: v['kind'] for k, v in group_mapping.items()}
            if kinds:
                semantic = f" [Groups: {kinds}]"
        elif col in ('LocalDrAmt', 'DrAmt'):
            semantic = " [DEBIT amount]"
        elif col in ('LocalCrAmt', 'CrAmt'):
            semantic = " [CREDIT amount]"
        elif col in ('VDate', 'Date'):
            semantic = " [Voucher date, use VDate_iso for filtering]"
        elif col == 'VNo':
            semantic = " [Voucher number]"
        elif col == 'GlDesc':
            semantic = " [Party/Ledger name]"
        elif col == 'Catagory':
            semantic = " [Category]"
        elif col == 'Narration':
            semantic = " [Narration/Description]"

        lines.append(f"  - {col}: {col_type.upper()} {nullable}, distinct={distinct}{extra}{semantic}")

    lines.append("")
    lines.append("DERIVED COLUMNS (available for queries):")
    lines.append("  - VDate_iso: DATE (YYYY-MM-DD) - parsed from VDate")
    lines.append("  - net_amount: REAL (LocalDrAmt - LocalCrAmt)")
    lines.append("  - is_opening: BOOLEAN (Source = 'OB')")
    lines.append("  - year, month, week, day: INTEGER (from VDate_iso)")
    lines.append("")
    lines.append(f"Total records: {profile['record_count']}")
    if profile['date_range']['min'] and profile['date_range']['max']:
        lines.append(f"Date range: {profile['date_range']['min']} to {profile['date_range']['max']}")

    return "\n".join(lines)


async def analyze_data_with_llm(records: list[dict], profile: dict, openai_client, model: str) -> dict:
    """Use LLM to analyze data and infer semantic mappings."""
    # Prepare summary for LLM
    source_col = 'Source'
    group_col = 'GrpDesc'

    source_values = profile['top_values'].get(source_col, [])
    group_values = profile['top_values'].get(group_col, [])

    # Sample records per source
    source_samples = {}
    for source, _ in source_values[:10]:
        samples = [r for r in records if str(r.get(source_col, '')).strip() == str(source).strip()][:3]
        source_samples[str(source)] = [
            {
                'GlDesc': s.get('GlDesc', ''),
                'LocalDrAmt': s.get('LocalDrAmt', 0),
                'LocalCrAmt': s.get('LocalCrAmt', 0),
                'Narration': s.get('Narration', ''),
                'GrpDesc': s.get('GrpDesc', '')
            }
            for s in samples
        ]

    group_samples = {}
    for group, _ in group_values[:10]:
        samples = [r for r in records if str(r.get(group_col, '')).strip() == str(group).strip()][:3]
        group_samples[str(group)] = [
            {
                'GlDesc': s.get('GlDesc', ''),
                'LocalDrAmt': s.get('LocalDrAmt', 0),
                'LocalCrAmt': s.get('LocalCrAmt', 0),
                'Narration': s.get('Narration', ''),
                'Source': s.get('Source', '')
            }
            for s in samples
        ]

    prompt = f"""Analyze this accounting data and infer semantic meanings.

SOURCE CODES (with sample entries):
{json.dumps(source_samples, indent=2, ensure_ascii=False)}

GROUP DESCRIPTIONS (with sample entries):
{json.dumps(group_samples, indent=2, ensure_ascii=False)}

Return JSON with two mappings:
1. source_roles: {{source_code -> role}} where role is one of: opening_balance, sales, collection, purchase, payment, journal, credit_note, debit_note, other
2. group_kinds: {{group_name -> kind}} where kind is one of: asset, liability, equity, income, expense, other

Only include mappings you're confident about. For ambiguous ones, use "other"."""

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert accounting data analyst. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"LLM analysis failed: {e}")
        return {"source_roles": {}, "group_kinds": {}}


def create_sqlite_table(conn: sqlite3.Connection, records: list[dict], profile: dict) -> None:
    """Create SQLite table with appropriate columns and derived columns."""
    cursor = conn.cursor()

    # Determine column types
    columns = []
    for col, info in profile['columns'].items():
        col_type = info['type']
        if col_type == 'number':
            sql_type = 'REAL'
        elif col_type == 'date':
            sql_type = 'TEXT'  # Store as ISO string
        else:
            sql_type = 'TEXT'
        columns.append(f'"{col}" {sql_type}')

    # Add derived columns
    columns.append('"VDate_iso" TEXT')
    columns.append('"net_amount" REAL')
    columns.append('"is_opening" INTEGER')
    columns.append('"year" INTEGER')
    columns.append('"month" INTEGER')
    columns.append('"week" INTEGER')
    columns.append('"day" INTEGER')

    create_sql = f'CREATE TABLE IF NOT EXISTS data ({", ".join(columns)})'
    cursor.execute(create_sql)

    # Create indexes for common queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_vdate_iso ON data(VDate_iso)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_gldesc ON data(GlDesc)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON data(Source)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_grpdesc ON data(GrpDesc)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_catagory ON data(Catagory)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_vno ON data(VNo)')

    conn.commit()


def insert_records(conn: sqlite3.Connection, records: list[dict], profile: dict) -> int:
    """Insert records into SQLite table with derived columns."""
    cursor = conn.cursor()

    # Get column names
    cols = list(profile['columns'].keys())
    # VDate_iso, net_amount, is_opening, year, month, week, and day.
    placeholders = ', '.join(['?'] * (len(cols) + 7))
    quoted_cols = ', '.join(f'"{col}"' for col in cols)

    insert_sql = f'INSERT INTO data ({quoted_cols}, "VDate_iso", "net_amount", "is_opening", "year", "month", "week", "day") VALUES ({placeholders})'

    count = 0
    for rec in records:
        values = []
        for col in cols:
            val = rec.get(col)
            if val is None:
                values.append(None)
            elif profile['columns'][col]['type'] == 'number':
                values.append(parse_amount(val))
            else:
                values.append(str(val) if val is not None else None)

        # Derived columns
        vdate = rec.get('VDate', rec.get('Date', ''))
        vdate_iso = parse_date(vdate)
        values.append(vdate_iso)

        dr = parse_amount(rec.get('LocalDrAmt', rec.get('DrAmt', 0)))
        cr = parse_amount(rec.get('LocalCrAmt', rec.get('CrAmt', 0)))
        values.append(dr - cr)  # net_amount

        source = str(rec.get('Source', '')).upper().strip()
        values.append(1 if source == 'OB' else 0)  # is_opening

        if vdate_iso:
            try:
                dt = datetime.strptime(vdate_iso, '%Y-%m-%d')
                values.append(dt.year)
                values.append(dt.month)
                values.append(dt.isocalendar()[1])  # week number
                values.append(dt.day)
            except ValueError:
                values.extend([None, None, None, None])
        else:
            values.extend([None, None, None, None])

        cursor.execute(insert_sql, values)
        count += 1

    conn.commit()
    return count
