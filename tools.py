"""
Tools module for the Chat with Company Data application (FIXED VERSION).

Every ledger-level tool works for ANY GlDesc (party/ledger) in the data and uses the
same rules as the "Outstanding Bills Report" screenshot:

  * Bills        = debit-side entries of the party (receivable) / credit-side entries (payable)
  * Credits      = ALL opposite-side entries of the party (CB, PB, CN, JV ... every Source)
  * FIFO         = the credit pool is applied to the oldest bills first
  * Outstanding  = bills - credits applied   (reconciles with the ledger closing balance)
  * Turnover     = average daily net receivable / net sales * period days
                   (period = first entry date .. as-of date, inclusive)
                   ratio = 365 / that number
"""

import base64
import csv
import io
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _iso(value) -> Optional[str]:
    """Normalise yyyy-mm-dd / dd/mm/yyyy / dd-mm-yyyy / ISO datetime to yyyy-mm-dd."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})', s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.match(r'^(\d{2})[/-](\d{2})[/-](\d{4})$', s)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return s


def _d(s) -> date:
    return datetime.strptime(str(s)[:10], '%Y-%m-%d').date()


def _r2(x) -> float:
    return round(float(x or 0), 2)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: Any = None
    error: str = ""
    truncated: bool = False
    row_count: int = 0

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'truncated': self.truncated,
            'row_count': self.row_count
        }


# ----------------------------------------------------------------------------
# SQL executor
# ----------------------------------------------------------------------------

_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")


class SQLiteExecutor:
    """Safe read-only SQLite executor with authorization."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = None
        self.max_rows = 200
        self.timeout = 15  # seconds

    def connect(self, read_only: bool = True) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA busy_timeout = 15000")
        if read_only:
            self.conn.set_authorizer(self._authorizer)
        else:
            self.conn.set_authorizer(None)
        return self.conn

    def _authorizer(self, action, arg1, arg2, dbname, trigger):
        """Only allow what is needed to evaluate read-only queries."""
        allowed = {
            sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_PRAGMA,
            getattr(sqlite3, 'SQLITE_RECURSIVE', 33),  # WITH RECURSIVE
        }
        if action in allowed:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def execute(self, query: str, params: tuple = (), max_rows: Optional[int] = None) -> ToolResult:
        if not self._is_read_only(query):
            return ToolResult(success=False, error="Only SELECT queries are allowed")

        # single statement only (semicolons inside string literals are fine)
        body = _LITERAL_RE.sub("''", query).strip().rstrip(';')
        if ';' in body:
            return ToolResult(success=False, error="Multiple statements not allowed")

        limit = self.max_rows if max_rows is None else max_rows

        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchmany(limit + 1)
            truncated = len(rows) > limit
            if truncated:
                rows = rows[:limit]
            data = [dict(row) for row in rows]
            return ToolResult(success=True, data=data, row_count=len(data), truncated=truncated)
        except sqlite3.Error as e:
            return ToolResult(success=False, error=f"SQL error: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Execution error: {str(e)}")

    def _is_read_only(self, query: str) -> bool:
        # Strip string literals first, otherwise  LIKE '%delete%'  or a ledger name
        # containing a keyword is rejected. The SQLite authorizer is the real guard.
        q = _LITERAL_RE.sub("''", query).upper().strip()
        if not (q.startswith('SELECT') or q.startswith('WITH')):
            return False
        forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE',
                     'ATTACH', 'DETACH', 'VACUUM']
        for word in forbidden:
            if re.search(rf'\b{word}\b', q):
                return False
        # REPLACE(...) the string function is fine; REPLACE INTO is not.
        if re.search(r'\bREPLACE\b(?!\s*\()', q):
            return False
        return True

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


# ----------------------------------------------------------------------------
# analytical tools
# ----------------------------------------------------------------------------

BIG = 5_000_000


class DataTools:
    """Collection of all analytical tools for the chat application."""

    DEFAULT_SLABS = '0-30,31-60,61-90,91-120,121-150,151-180,181+'

    def __init__(self, executor: SQLiteExecutor, settings: dict, source_mapping: Optional[dict] = None):
        self.executor = executor
        self.settings = settings
        self.max_rows = settings.get('max_rows', 200)
        self.source_mapping = source_mapping or {}

    # ---------------- source / slab helpers ----------------

    def _roles(self, roles, defaults) -> set:
        roles = {roles} if isinstance(roles, str) else set(roles)
        codes = {str(k).upper().strip() for k, v in (self.source_mapping or {}).items()
                 if isinstance(v, dict) and v.get('role') in roles}
        return codes | {str(c).upper().strip() for c in defaults}

    def _sales_src(self):
        return self._roles('sales', ('SB', 'SI', 'INV', 'SALES'))

    def _purch_src(self):
        return self._roles('purchase', ('PB', 'PI', 'PURCHASE'))

    def _coll_src(self):
        return self._roles(('collection', 'payment', 'cash_bank', 'cash_bank_transaction'),
                           ('CB', 'RECEIPT', 'COLLECTION', 'PAY', 'PAYMENT'))

    def _sales_ret_src(self):
        return self._roles(('sales_return', 'credit_note'), ('SR', 'SRN', 'CN'))

    def _purch_ret_src(self):
        return self._roles(('purchase_return', 'debit_note'), ('PR', 'PRN', 'DN'))

    @staticmethod
    def _parse_slabs(s: str):
        out = []
        for part in str(s).split(','):
            part = part.strip()
            if not part:
                continue
            try:
                if part.endswith('+'):
                    out.append((int(part[:-1]), float('inf'), part))
                elif '-' in part:
                    a, b = part.split('-', 1)
                    out.append((int(a), int(b), part))
                else:
                    out.append((int(part), int(part), part))
            except ValueError:
                continue
        return out

    def _slabs(self):
        raw = self.settings.get('aging_slabs') or self.DEFAULT_SLABS
        # Preserve existing saved configs while matching the screenshot/Outstanding Bills Report defaults.
        if str(raw).replace(' ', '') == '0-30,31-60,61-90,91-180,181-365,365+':
            raw = self.DEFAULT_SLABS
        return self._parse_slabs(raw) or self._parse_slabs(self.DEFAULT_SLABS)

    def _get_aging_slab(self, days: int, slabs_str: Optional[str] = None) -> str:
        slabs = self._parse_slabs(slabs_str) if slabs_str else self._slabs()
        days = max(int(days), 0)
        for start, end, label in slabs:
            if start <= days <= end:
                return label
        return slabs[-1][2] if slabs else '181+'

    # ---------------- ledger name matching ----------------

    @staticmethod
    def _like_escape(s: str) -> str:
        return s.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')

    def _matching_ledgers(self, ledger: str, limit: int = 50) -> list:
        """GlDesc names matching `ledger`.
        1) exact (case-insensitive)  2) every word of the input contained in the name
        3) punctuation/space-insensitive contains  (works for 'abn enterprises' -> 'A.B.N.Enterprises - X')"""
        ledger = (ledger or '').strip()
        if not ledger:
            return []
        ex = self.executor
        r = ex.execute("SELECT DISTINCT GlDesc FROM data WHERE LOWER(TRIM(GlDesc)) = LOWER(?) LIMIT ?",
                       (ledger, limit))
        if r.success and r.data:
            return [row['GlDesc'] for row in r.data]

        tokens = [t for t in re.split(r'\s+', ledger) if t]
        if tokens:
            conds = " AND ".join(["LOWER(GlDesc) LIKE ? ESCAPE '\\'"] * len(tokens))
            params = tuple(f"%{self._like_escape(t.lower())}%" for t in tokens) + (limit,)
            r = ex.execute(f"SELECT DISTINCT GlDesc FROM data WHERE {conds} ORDER BY GlDesc LIMIT ?", params)
            if r.success and r.data:
                return [row['GlDesc'] for row in r.data]

        norm = re.sub(r'[^a-z0-9]', '', ledger.lower())
        if norm:
            r = ex.execute(
                "SELECT DISTINCT GlDesc FROM data WHERE "
                "REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(LOWER(GlDesc),'.',''),' ',''),'-',''),'/',''),',','') "
                "LIKE ? ORDER BY GlDesc LIMIT ?", (f"%{norm}%", limit))
            if r.success and r.data:
                return [row['GlDesc'] for row in r.data]
        return []

    def _matching_ledger(self, ledger: str):
        """Single unambiguous match or None."""
        names = self._matching_ledgers(ledger)
        return names[0] if len(names) == 1 else None

    def _resolve_ledger(self, ledger: str):
        names = self._matching_ledgers(ledger)
        if not names:
            return None, ToolResult(success=False, error=f"No ledger found matching '{ledger}'")
        if len(names) == 1:
            return names[0], None
        return None, ToolResult(
            success=False,
            error=f"Multiple ledgers match '{ledger}': {names[:20]}. Please specify the exact name.",
            data={'options': names[:20]})

    # ---------------- raw ledger rows ----------------

    def _load_rows(self, ledger_name: str, as_of: str) -> list:
        r = self.executor.execute(
            "SELECT rowid AS rid, VDate_iso AS d, VNo, Source, Narration, GrpDesc, Catagory, "
            "COALESCE(LocalDrAmt,0) AS dr, COALESCE(LocalCrAmt,0) AS cr "
            "FROM data WHERE GlDesc = ? AND VDate_iso <= ? ORDER BY VDate_iso, rid",
            (ledger_name, as_of), max_rows=BIG)
        return r.data if r.success and r.data else []

    def _party_rows_by_ledger(self, as_of: str):
        r = self.executor.execute(
            "SELECT rowid AS rid, GlDesc, VDate_iso AS d, VNo, Source, Narration, GrpDesc, Catagory, "
            "COALESCE(LocalDrAmt,0) AS dr, COALESCE(LocalCrAmt,0) AS cr FROM data "
            "WHERE (LOWER(Catagory) LIKE '%customer%' OR LOWER(Catagory) LIKE '%vendor%') "
            "AND VDate_iso <= ? ORDER BY GlDesc, VDate_iso, rid", (as_of,), max_rows=BIG)
        if not r.success:
            return r, None
        groups = defaultdict(list)
        for row in r.data:
            groups[row['GlDesc']].append(row)
        return None, groups

    def _party_mode(self, rows: list) -> str:
        """'receivable' (customer: bills are debits) or 'payable' (vendor: bills are credits)."""
        sales, purch = self._sales_src(), self._purch_src()
        s = sum(r['dr'] for r in rows if str(r['Source']).upper().strip() in sales)
        p = sum(r['cr'] for r in rows if str(r['Source']).upper().strip() in purch)
        if s > 0 or p > 0:
            return 'receivable' if s >= p else 'payable'
        net = sum(r['dr'] - r['cr'] for r in rows)
        return 'receivable' if net >= 0 else 'payable'

    # ---------------- FIFO core ----------------

    def _fifo_core(self, rows: list, mode: str, as_of: str) -> dict:
        sign = 1 if mode == 'receivable' else -1
        bills, credit_total = [], 0.0
        for r in rows:
            net = sign * (r['dr'] - r['cr'])
            if net > 0.004:
                bills.append({'vno': r['VNo'], 'date': r['d'], 'source': r['Source'],
                              'narration': r['Narration'] or '', 'amount': _r2(net),
                              'applied': 0.0, 'outstanding': 0.0})
            elif net < -0.004:
                credit_total += -net
        remaining = credit_total
        for b in bills:
            a = min(b['amount'], remaining)
            b['applied'] = _r2(a)
            remaining -= a
            b['outstanding'] = _r2(b['amount'] - a)
        as_of_d = _d(as_of)
        pending = []
        for b in bills:
            if b['outstanding'] > 0.004:
                b['days'] = (as_of_d - _d(b['date'])).days
                b['slab'] = self._get_aging_slab(b['days'])
                pending.append(b)
        return {
            'bills': bills, 'pending': pending,
            'credits_total': _r2(credit_total),
            'unapplied_credit': _r2(max(remaining, 0)),
            'total_bills': _r2(sum(b['amount'] for b in bills)),
            'total_outstanding': _r2(sum(b['outstanding'] for b in pending)),
        }

    def _slab_summary(self, pending: list) -> dict:
        summary = {label: {'count': 0, 'amount': 0.0} for _, _, label in self._slabs()}
        for b in pending:
            s = summary.setdefault(b['slab'], {'count': 0, 'amount': 0.0})
            s['count'] += 1
            s['amount'] = _r2(s['amount'] + b['outstanding'])
        return summary

    # ---------------- turnover core ----------------

    def _metrics_for(self, rows: list, mode: str, as_of: str) -> dict:
        sign = 1 if mode == 'receivable' else -1
        as_of_d = _d(as_of)
        start_d = _d(rows[0]['d'])
        period_days = (as_of_d - start_d).days + 1

        # average DAILY net balance: balance * (days it stayed) / period_days
        delta = defaultdict(float)
        for r in rows:
            delta[r['d']] += sign * (r['dr'] - r['cr'])
        days = sorted(delta)
        bal, area = 0.0, 0.0
        for i, ds in enumerate(days):
            bal += delta[ds]
            nxt = _d(days[i + 1]) if i + 1 < len(days) else as_of_d + timedelta(days=1)
            area += bal * (nxt - _d(ds)).days
        avg_recv = area / period_days if period_days > 0 else 0.0
        closing = bal

        bill_src = self._sales_src() if mode == 'receivable' else self._purch_src()
        ret_src = self._sales_ret_src() if mode == 'receivable' else self._purch_ret_src()
        gross = returns = 0.0
        for r in rows:
            src = str(r['Source']).upper().strip()
            net = sign * (r['dr'] - r['cr'])
            if src in bill_src and net > 0:
                gross += net
            if src in ret_src and net < 0:
                returns += -net
        net_sales = gross - returns

        turnover_days_basis = self.settings.get('turnover_days', 365) or 365
        avg_coll_days = ratio = None
        note = ""
        if net_sales > 0 and avg_recv > 0:
            avg_coll_days = avg_recv / net_sales * period_days
            ratio = turnover_days_basis / avg_coll_days
        else:
            note = "Turnover not applicable: no net sales/purchases or no positive average balance in the period."

        return {
            'mode': mode,
            'period_start': rows[0]['d'], 'period_end': as_of, 'period_days': period_days,
            'gross_sales': _r2(gross), 'returns': _r2(returns), 'net_sales': _r2(net_sales),
            'net_sales_invoiced': _r2(gross),
            'avg_receivable': _r2(avg_recv), 'closing_balance': _r2(closing),
            'avg_collection_days': round(avg_coll_days, 2) if avg_coll_days is not None else None,
            'turnover_ratio': round(ratio, 2) if ratio is not None else None,
            'turnover_days': round(avg_coll_days, 2) if avg_coll_days is not None else None,
            'note': note,
        }

    # ------------------------------------------------------------------
    # public tools
    # ------------------------------------------------------------------

    def run_sql(self, query: str) -> ToolResult:
        """Execute a read-only SQL query."""
        return self.executor.execute(query)

    # ---------------- ledger statement ----------------

    def ledger_statement(self, ledger: str, from_date: str, to_date: str) -> ToolResult:
        """Ledger statement with running balance (positive = Dr, negative = Cr)."""
        from_date, to_date = _iso(from_date), _iso(to_date)
        matched, err = self._resolve_ledger(ledger)
        if err:
            return err

        # Opening = EVERY entry before from_date (not only 'OB' source)
        ob = self.executor.execute(
            "SELECT COALESCE(SUM(LocalDrAmt),0) AS dr, COALESCE(SUM(LocalCrAmt),0) AS cr "
            "FROM data WHERE GlDesc = ? AND VDate_iso < ?", (matched, from_date))
        if not ob.success:
            return ob
        opening_balance = ob.data[0]['dr'] - ob.data[0]['cr']

        period = self.executor.execute(
            "SELECT VDate_iso AS date, VNo AS voucher, Source, Narration, "
            "COALESCE(LocalDrAmt,0) AS debit, COALESCE(LocalCrAmt,0) AS credit "
            "FROM data WHERE GlDesc = ? AND VDate_iso BETWEEN ? AND ? ORDER BY VDate_iso, rowid",
            (matched, from_date, to_date), max_rows=BIG)
        if not period.success:
            return period

        running = opening_balance
        entries = []
        for row in period.data:
            running += row['debit'] - row['credit']
            entries.append({'date': row['date'], 'voucher': row['voucher'], 'source': row['Source'],
                            'narration': row['Narration'], 'debit': row['debit'],
                            'credit': row['credit'], 'balance': _r2(running)})
        total_dr = sum(e['debit'] for e in entries)
        total_cr = sum(e['credit'] for e in entries)
        shown = entries[:self.max_rows]

        return ToolResult(success=True, data={
            'ledger': matched,
            'period': {'from': from_date, 'to': to_date},
            'opening_balance': _r2(opening_balance),
            'entries': shown,
            'entries_total_count': len(entries),
            'entries_truncated': len(entries) > len(shown),
            'closing_balance': _r2(running),
            'closing_side': 'Dr' if running >= 0 else 'Cr',
            'total_debit': _r2(total_dr),
            'total_credit': _r2(total_cr),
            'note': 'Balance = debit - credit (positive Dr, negative Cr). Opening includes all entries before from_date.',
        })

    # ---------------- trial balance ----------------

    def trial_balance(self, as_of_date: str, group_by: str = 'ledger') -> ToolResult:
        as_of_date = _iso(as_of_date)
        if group_by not in ('ledger', 'group', 'category'):
            group_by = 'ledger'
        col = {'ledger': 'GlDesc', 'group': 'GrpDesc', 'category': 'Catagory'}[group_by]

        res = self.executor.execute(f"""
            SELECT {col} AS name,
                COALESCE(SUM(CASE WHEN UPPER(TRIM(Source))='OB' THEN LocalDrAmt - LocalCrAmt ELSE 0 END),0) AS opening,
                COALESCE(SUM(CASE WHEN UPPER(TRIM(Source))<>'OB' THEN LocalDrAmt ELSE 0 END),0) AS period_dr,
                COALESCE(SUM(CASE WHEN UPPER(TRIM(Source))<>'OB' THEN LocalCrAmt ELSE 0 END),0) AS period_cr
            FROM data WHERE VDate_iso <= ? GROUP BY {col} ORDER BY {col}
        """, (as_of_date,), max_rows=BIG)
        if not res.success:
            return res

        rows, total_dr, total_cr = [], 0.0, 0.0
        for r in res.data:
            closing = r['opening'] + r['period_dr'] - r['period_cr']
            dr_bal = closing if closing >= 0 else 0.0
            cr_bal = -closing if closing < 0 else 0.0
            rows.append({'name': r['name'], 'opening': _r2(r['opening']), 'debit': _r2(r['period_dr']),
                         'credit': _r2(r['period_cr']), 'closing': _r2(closing),
                         'dr_balance': _r2(dr_bal), 'cr_balance': _r2(cr_bal)})
            total_dr += dr_bal
            total_cr += cr_bal

        totals = {'name': 'TOTAL',
                  'opening': _r2(sum(r['opening'] for r in rows)),
                  'debit': _r2(sum(r['debit'] for r in rows)),
                  'credit': _r2(sum(r['credit'] for r in rows)),
                  'closing': _r2(sum(r['closing'] for r in rows)),
                  'dr_balance': _r2(total_dr), 'cr_balance': _r2(total_cr)}
        shown = rows
        truncated = False
        if len(rows) > 300:  # totals above are computed on ALL rows
            shown = sorted(rows, key=lambda x: abs(x['closing']), reverse=True)[:300]
            truncated = True

        return ToolResult(success=True, data={
            'as_of': as_of_date, 'group_by': group_by,
            'rows': shown + [totals], 'rows_total_count': len(rows), 'rows_truncated': truncated,
            'balanced': abs(total_dr - total_cr) < 1.0,
            'difference': _r2(total_dr - total_cr),
        })

    # ---------------- balance sheet / P&L ----------------

    def balance_sheet_pnl(self, as_of_date: str = None, from_date: str = None, to_date: str = None,
                          group_mapping: dict = None) -> ToolResult:
        group_mapping = group_mapping or {}
        as_of_date, from_date, to_date = _iso(as_of_date), _iso(from_date), _iso(to_date)
        if as_of_date:
            date_filter, params = "VDate_iso <= ?", (as_of_date,)
        elif from_date and to_date:
            date_filter, params = "VDate_iso BETWEEN ? AND ?", (from_date, to_date)
        else:
            date_filter, params = "1=1", ()

        result = self.executor.execute(f"""
            SELECT GrpDesc AS group_name, COALESCE(SUM(LocalDrAmt),0) AS total_dr,
                   COALESCE(SUM(LocalCrAmt),0) AS total_cr
            FROM data WHERE {date_filter} GROUP BY GrpDesc
        """, params, max_rows=BIG)
        if not result.success:
            return result

        buckets = {k: [] for k in ('asset', 'liability', 'equity', 'income', 'expense')}
        for row in result.data:
            group = row['group_name']
            dr, cr = row['total_dr'], row['total_cr']
            net = dr - cr                               # Dr positive
            kind = (group_mapping.get(group) or {}).get('kind', 'other')
            entry = {'group': group, 'debit': _r2(dr), 'credit': _r2(cr), 'net': _r2(net)}
            if kind not in buckets:
                kind = 'asset' if net >= 0 else 'liability'
                entry['assumed'] = kind
            # natural-sign balance: assets/expenses = Dr-Cr, liabilities/equity/income = Cr-Dr
            entry['balance'] = _r2(net if kind in ('asset', 'expense') else -net)
            buckets[kind].append(entry)

        s = lambda k: sum(e['balance'] for e in buckets[k])
        total_assets, total_liab, total_equity = s('asset'), s('liability'), s('equity')
        total_income, total_exp = s('income'), s('expense')
        net_profit = total_income - total_exp
        mismatch = total_assets - (total_liab + total_equity + net_profit)

        warnings = []
        cumulative = bool(as_of_date) or not (from_date and to_date)
        if cumulative and abs(mismatch) > 1.0:
            warnings.append(f"Balance sheet mismatch: {mismatch:.2f} (Assets - Liabilities - Equity - Net profit)")
        if any('assumed' in e for k in buckets for e in buckets[k]):
            warnings.append("Some groups are not mapped to asset/liability/equity/income/expense; "
                            "they were classified by balance sign (see 'assumed').")

        return ToolResult(success=True, data={
            'as_of': as_of_date, 'period': {'from': from_date, 'to': to_date},
            'balance_sheet': {
                'assets': buckets['asset'], 'liabilities': buckets['liability'], 'equity': buckets['equity'],
                'total_assets': _r2(total_assets), 'total_liabilities': _r2(total_liab),
                'total_equity': _r2(total_equity + net_profit),
            },
            'profit_loss': {
                'income': buckets['income'], 'expenses': buckets['expense'],
                'total_income': _r2(total_income), 'total_expenses': _r2(total_exp),
                'net_profit': _r2(net_profit),
            },
            'balanced': (abs(mismatch) < 1.0) if cumulative else None,
            'mismatch': _r2(mismatch), 'warnings': warnings,
        })

    # ---------------- FIFO outstanding ----------------

    def fifo_outstanding(self, ledger: str = 'ALL', as_of_date: str = None) -> ToolResult:
        as_of = _iso(as_of_date) or date.today().isoformat()
        if (ledger or 'ALL').strip().upper() == 'ALL':
            return self._fifo_all(as_of)

        matched, err = self._resolve_ledger(ledger)
        if err:
            return err
        rows = self._load_rows(matched, as_of)
        if not rows:
            return ToolResult(success=False, error=f"No entries for '{matched}' up to {as_of}")
        mode = self._party_mode(rows)
        f = self._fifo_core(rows, mode, as_of)
        sign = 1 if mode == 'receivable' else -1
        closing = sign * sum(r['dr'] - r['cr'] for r in rows)

        outstanding = [{
            'ledger': matched, 'voucher': b['vno'], 'date': b['date'], 'source': b['source'],
            'original_amount': b['amount'], 'fifo_applied': b['applied'], 'outstanding': b['outstanding'],
            'days_outstanding': b['days'], 'aging_slab': b['slab'], 'narration': b['narration'],
        } for b in f['pending']]

        warnings = []
        cat = str(rows[0].get('Catagory') or '')
        if not re.search(r'customer|vendor', cat, re.I):
            warnings.append(f"Catagory is '{cat}', not Customer/Vendor: bill-wise outstanding may not be meaningful.")

        return ToolResult(success=True, data={
            'as_of': as_of, 'ledger': matched, 'mode': mode,
            'method': 'FIFO: all opposite-side entries of the ledger (every Source) are applied to the oldest bills first',
            'total_bills': f['total_bills'],
            'credits_applied': _r2(f['credits_total'] - f['unapplied_credit']),
            'unapplied_credit': f['unapplied_credit'],
            'outstanding': outstanding,
            'summary_by_slab': self._slab_summary(f['pending']),
            'total_outstanding': f['total_outstanding'],
            'total_count': len(outstanding),
            'closing_balance': _r2(closing),
            'reconciles': abs((f['total_outstanding'] - f['unapplied_credit']) - closing) < 0.05,
            'warnings': warnings,
        })

    def _fifo_all(self, as_of: str) -> ToolResult:
        err, groups = self._party_rows_by_ledger(as_of)
        if err:
            return err
        agg = {m: {'total_outstanding': 0.0, 'unapplied_credit': 0.0, 'ledgers_with_outstanding': 0,
                   'slabs': {label: {'count': 0, 'amount': 0.0} for _, _, label in self._slabs()}}
               for m in ('receivable', 'payable')}
        ledgers, bills = [], []
        for name, rows in groups.items():
            mode = self._party_mode(rows)
            f = self._fifo_core(rows, mode, as_of)
            a = agg[mode]
            a['total_outstanding'] += f['total_outstanding']
            a['unapplied_credit'] += f['unapplied_credit']
            if f['pending']:
                a['ledgers_with_outstanding'] += 1
            for b in f['pending']:
                s = a['slabs'].setdefault(b['slab'], {'count': 0, 'amount': 0.0})
                s['count'] += 1
                s['amount'] += b['outstanding']
                bills.append({'ledger': name, 'mode': mode, 'voucher': b['vno'], 'date': b['date'],
                              'outstanding': b['outstanding'], 'days_outstanding': b['days'],
                              'aging_slab': b['slab']})
            if f['pending'] or f['unapplied_credit']:
                ledgers.append({'ledger': name, 'mode': mode, 'outstanding': f['total_outstanding'],
                                'unapplied_credit': f['unapplied_credit'], 'pending_bills': len(f['pending']),
                                'oldest_days': max((b['days'] for b in f['pending']), default=0)})
        ledgers.sort(key=lambda x: x['outstanding'], reverse=True)
        bills.sort(key=lambda x: x['outstanding'], reverse=True)
        out = {}
        for m, a in agg.items():
            out[m] = {'total_outstanding': _r2(a['total_outstanding']),
                      'unapplied_credit': _r2(a['unapplied_credit']),
                      'ledgers_with_outstanding': a['ledgers_with_outstanding'],
                      'summary_by_slab': {k: {'count': v['count'], 'amount': _r2(v['amount'])}
                                          for k, v in a['slabs'].items()}}
        return ToolResult(success=True, data={
            'as_of': as_of, 'ledger': 'ALL',
            'scope': 'Ledgers whose Catagory is Customer/Vendor only (sales/purchase revenue, VAT, bank ledgers are excluded)',
            'receivable': out['receivable'], 'payable': out['payable'],
            'top_ledgers': ledgers[:200], 'ledgers_listed': min(len(ledgers), 200),
            'ledgers_total': len(ledgers),
            'outstanding': bills[:300], 'pending_bills_total': len(bills),
        })

    # ---------------- receivables metrics ----------------

    def receivables_metrics(self, ledger: str = 'ALL', as_of_date: str = None) -> ToolResult:
        as_of = _iso(as_of_date) or date.today().isoformat()
        formulas = {
            'net_sales': 'Sales/purchase-source bill entries minus returns/credit notes',
            'avg_receivable': 'Average of the END-OF-DAY net balance over every day from first entry to as-of (inclusive)',
            'avg_collection_days': 'avg_receivable / net_sales * period_days',
            'turnover_ratio': f"{self.settings.get('turnover_days', 365)} / avg_collection_days",
            'turnover_days': 'same as avg_collection_days',
        }
        if (ledger or 'ALL').strip().upper() == 'ALL':
            err, groups = self._party_rows_by_ledger(as_of)
            if err:
                return err
            metrics = []
            for name, rows in groups.items():
                m = self._metrics_for(rows, self._party_mode(rows), as_of)
                m['ledger'] = name
                metrics.append(m)
            metrics.sort(key=lambda x: abs(x['closing_balance']), reverse=True)
            return ToolResult(success=True, data={
                'as_of': as_of, 'metrics': metrics[:200], 'ledgers_total': len(metrics),
                'ledgers_listed': min(len(metrics), 200), 'formulas': formulas})

        matched, err = self._resolve_ledger(ledger)
        if err:
            return err
        rows = self._load_rows(matched, as_of)
        if not rows:
            return ToolResult(success=False, error=f"No entries for '{matched}' up to {as_of}")
        m = self._metrics_for(rows, self._party_mode(rows), as_of)
        m['ledger'] = matched
        return ToolResult(success=True, data={'as_of': as_of, 'metrics': [m], 'formulas': formulas})

    # ---------------- complete ledger report (dashboard-style) ----------------

    def ledger_report(self, ledger: str, as_of_date: str = None) -> ToolResult:
        """Everything in the 'Outstanding Bills Report' for ANY GlDesc, in one call."""
        as_of = _iso(as_of_date) or date.today().isoformat()
        matched, err = self._resolve_ledger(ledger)
        if err:
            return err
        rows = self._load_rows(matched, as_of)
        if not rows:
            return ToolResult(success=False, error=f"No entries for '{matched}' up to {as_of}")

        mode = self._party_mode(rows)
        sign = 1 if mode == 'receivable' else -1
        fifo = self._fifo_core(rows, mode, as_of)
        m = self._metrics_for(rows, mode, as_of)
        bill_src = self._sales_src() if mode == 'receivable' else self._purch_src()
        coll_src = self._coll_src()
        as_of_d = _d(as_of)

        def amounts(r):
            src = str(r['Source']).upper().strip()
            net = sign * (r['dr'] - r['cr'])
            bill = net if (src in bill_src and net > 0) else 0.0
            coll = -net if (src in coll_src and net < 0) else 0.0
            tx = 1 if (src in bill_src or src in coll_src) else 0
            return bill, coll, tx

        # monthly
        monthly = defaultdict(lambda: [0.0, 0.0, 0])
        for r in rows:
            b, c, t = amounts(r)
            k = r['d'][:7]
            monthly[k][0] += b
            monthly[k][1] += c
            monthly[k][2] += t
        months, cur, end_m = [], _d(rows[0]['d']).replace(day=1), as_of_d.replace(day=1)
        while cur <= end_m:
            k = cur.strftime('%Y-%m')
            v = monthly.get(k, [0.0, 0.0, 0])
            months.append({'month': cur.strftime('%b %Y'), 'sales': _r2(v[0]), 'collections': _r2(v[1]),
                           'transactions': v[2]})
            cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

        # 52 consecutive 7-day weeks ending on the as-of date
        weeks = [[0.0, 0.0, 0] for _ in range(52)]
        for r in rows:
            idx = (as_of_d - _d(r['d'])).days // 7
            if 0 <= idx < 52:
                b, c, t = amounts(r)
                w = weeks[51 - idx]
                w[0] += b
                w[1] += c
                w[2] += t
        weekly = []
        for i, w in enumerate(weeks):
            wend = as_of_d - timedelta(days=7 * (51 - i))
            wstart = wend - timedelta(days=6)
            weekly.append({'week_no': i + 1, 'from': wstart.isoformat(), 'to': wend.isoformat(),
                           'sales': _r2(w[0]), 'collections': _r2(w[1]), 'transactions': w[2]})

        # source breakup
        src = defaultdict(lambda: [0.0, 0.0])
        for r in rows:
            src[str(r['Source'])][0] += r['dr']
            src[str(r['Source'])][1] += r['cr']
        breakup = [{'source': k, 'debit_total': _r2(v[0]), 'credit_total': _r2(v[1])} for k, v in sorted(src.items())]

        slab_summary = self._slab_summary(fifo['pending'])
        aging = [{'slab': k, 'count': v['count'], 'amount': v['amount']} for k, v in slab_summary.items()]
        pending = [{'date': b['date'], 'voucher': b['vno'], 'source': b['source'], 'narration': b['narration'],
                    'bill_amount': b['amount'], 'fifo_applied': b['applied'], 'outstanding': b['outstanding'],
                    'age_days': b['days'], 'aging_slab': b['slab']} for b in fifo['pending']]

        highlights = {}
        if fifo['pending']:
            big = max(slab_summary.items(), key=lambda kv: kv[1]['amount'])
            oldest = fifo['pending'][0]
            highlights = {'largest_aging_slab': {'slab': big[0], 'amount': big[1]['amount']},
                          'oldest_pending_bill': {'voucher': oldest['vno'], 'age_days': oldest['days'],
                                                  'outstanding': oldest['outstanding']}}

        notes = []
        cat = str(rows[0].get('Catagory') or '')
        if not re.search(r'customer|vendor', cat, re.I):
            notes.append(f"Catagory is '{cat}', not Customer/Vendor: bill-wise report may not be meaningful.")
        if mode == 'payable':
            notes.append("Vendor-type ledger: 'sales' = purchases (credit side), 'collections' = payments (debit side).")
        if m['note']:
            notes.append(m['note'])

        closing = sign * sum(r['dr'] - r['cr'] for r in rows)
        return ToolResult(success=True, data={
            'ledger': matched, 'as_of': as_of, 'mode': mode,
            'group': rows[0].get('GrpDesc'), 'category': cat,
            'bill_sources': sorted(bill_src), 'collection_sources': sorted(coll_src),
            'kpis': {
                'sales': m['gross_sales'], 'returns_credit_notes': m['returns'], 'net_sales': m['net_sales'],
                'average_receivable': m['avg_receivable'], 'average_collection_days': m['avg_collection_days'],
                'receivables_turnover_ratio': m['turnover_ratio'], 'turnover_days': m['turnover_days'],
                'credits_applied': _r2(fifo['credits_total'] - fifo['unapplied_credit']),
                'outstanding': fifo['total_outstanding'], 'unapplied_credit': fifo['unapplied_credit'],
                'closing_balance': _r2(closing),
            },
            'turnover_period': {'from': m['period_start'], 'to': as_of, 'days': m['period_days']},
            'aging': aging, 'source_breakup': breakup,
            'monthly': months, 'weekly_52': weekly, 'pending_bills': pending,
            'pending_bill_total': {'bill_amount': _r2(sum(b['amount'] for b in fifo['pending'])),
                                   'fifo_applied': _r2(sum(b['applied'] for b in fifo['pending'])),
                                   'outstanding': fifo['total_outstanding']},
            'highlights': highlights,
            'checks': {'outstanding_minus_unapplied_equals_closing':
                       abs((fifo['total_outstanding'] - fifo['unapplied_credit']) - closing) < 0.05},
            'notes': notes,
        })

    # ---------------- period series ----------------

    def _generate_periods(self, from_date: str, to_date: str, grain: str) -> list:
        """Labels exactly as SQLite produces them; built day by day so year/quarter/week
        boundaries and 29/30/31st can never raise or mismatch."""
        cur, end = _d(from_date), _d(to_date)
        labels, last = [], None
        while cur <= end:
            if grain == 'day':
                lab = cur.strftime('%Y-%m-%d')
            elif grain == 'week':
                lab = cur.strftime('%Y-%W')
            elif grain == 'quarter':
                lab = f"{cur.year}-Q{(cur.month - 1) // 3 + 1}"
            elif grain == 'year':
                lab = cur.strftime('%Y')
            else:
                lab = cur.strftime('%Y-%m')
            if lab != last:
                labels.append(lab)
                last = lab
            cur += timedelta(days=1)
        return labels

    def period_series(self, metric: str, grain: str = 'month', from_date: str = None,
                      to_date: str = None, group_by: str = None,
                      source: str = None, ledger: str = None) -> ToolResult:
        """Time series. Optional filters: source ('SB' or 'SB,CB') and ledger (GlDesc)."""
        from_date, to_date = _iso(from_date), _iso(to_date)
        if metric not in ('debit', 'credit', 'net', 'balance'):
            return ToolResult(success=False, error=f"Unknown metric: {metric}")
        if not from_date:
            r = self.executor.execute("SELECT MIN(VDate_iso) AS v FROM data WHERE VDate_iso IS NOT NULL")
            from_date = r.data[0]['v'] if r.success and r.data and r.data[0]['v'] else '2020-01-01'
        if not to_date:
            r = self.executor.execute("SELECT MAX(VDate_iso) AS v FROM data WHERE VDate_iso IS NOT NULL")
            to_date = r.data[0]['v'] if r.success and r.data and r.data[0]['v'] else date.today().isoformat()

        grain_sql = {
            'day': "strftime('%Y-%m-%d', VDate_iso)",
            'week': "strftime('%Y-%W', VDate_iso)",
            'month': "strftime('%Y-%m', VDate_iso)",
            'quarter': "strftime('%Y', VDate_iso) || '-Q' || ((CAST(strftime('%m', VDate_iso) AS INTEGER) - 1) / 3 + 1)",
            'year': "strftime('%Y', VDate_iso)",
        }.get(grain)
        if grain_sql is None:
            grain = 'month'
            grain_sql = "strftime('%Y-%m', VDate_iso)"

        group_col = {'ledger': 'GlDesc', 'group': 'GrpDesc', 'category': 'Catagory', 'source': 'Source',
                     'GlDesc': 'GlDesc', 'GrpDesc': 'GrpDesc', 'Catagory': 'Catagory',
                     'Source': 'Source'}.get(group_by)

        extra, eparams = [], []
        if source:
            codes = [s.strip().upper() for s in str(source).split(',') if s.strip()]
            if codes:
                extra.append(f"UPPER(TRIM(Source)) IN ({','.join('?' * len(codes))})")
                eparams += codes
        if ledger:
            matched, err = self._resolve_ledger(ledger)
            if err:
                return err
            extra.append("GlDesc = ?")
            eparams.append(matched)
        extra_sql = (" AND " + " AND ".join(extra)) if extra else ""

        agg = {'debit': "SUM(LocalDrAmt)", 'credit': "SUM(LocalCrAmt)"}.get(metric, "SUM(LocalDrAmt - LocalCrAmt)")
        g_sel = f", {group_col} AS group_name" if group_col else ""
        g_by = ", group_name" if group_col else ""
        res = self.executor.execute(
            f"SELECT {grain_sql} AS period{g_sel}, COALESCE({agg},0) AS value FROM data "
            f"WHERE VDate_iso BETWEEN ? AND ?{extra_sql} GROUP BY period{g_by} ORDER BY period{g_by}",
            tuple([from_date, to_date] + eparams), max_rows=BIG)
        if not res.success:
            return res

        # opening balance for the cumulative 'balance' metric
        opening = defaultdict(float)
        if metric == 'balance':
            g_expr = group_col or "''"
            o = self.executor.execute(
                f"SELECT {g_expr} AS g, "
                f"COALESCE(SUM(LocalDrAmt - LocalCrAmt),0) AS v FROM data "
                f"WHERE VDate_iso < ?{extra_sql} GROUP BY g",
                tuple([from_date] + eparams), max_rows=BIG)
            if o.success:
                for row in o.data:
                    opening[row['g'] if group_col else None] = row['v']

        periods = self._generate_periods(from_date, to_date, grain)
        lookup = defaultdict(float)
        groups = set()
        for row in res.data:
            g = row['group_name'] if group_col else None
            groups.add(g)
            lookup[(row['period'], g)] = row['value']

        def build(g):
            out, running = [], opening.get(g, 0.0) if metric == 'balance' else 0.0
            for p in periods:
                v = lookup.get((p, g), 0.0)
                if metric == 'balance':
                    running += v
                    v = running
                out.append({'period': p, 'value': _r2(v)})
            return out

        base = {'metric': metric, 'grain': grain, 'from': from_date, 'to': to_date, 'periods': periods,
                'filters': {'source': source, 'ledger': ledger}}
        if metric == 'balance':
            base['note'] = 'balance = cumulative (debit - credit) including everything before from_date'
        if group_col:
            if metric == 'balance':
                groups |= {g for g in opening if g is not None}
            gl = sorted(g for g in groups if g is not None)
            if len(gl) > 50:  # keep the answer small; biggest groups only
                tot = {g: sum(abs(lookup.get((p, g), 0.0)) for p in periods) for g in gl}
                gl = sorted(sorted(gl, key=lambda x: tot[x], reverse=True)[:50])
                base['groups_truncated'] = True
            base.update({'groups': gl, 'series': {g: build(g) for g in gl}})
        else:
            base['series'] = build(None)
        return ToolResult(success=True, data=base)

    # ---------------- compare ----------------

    def compare(self, metric: str, items: list, as_of_date: str = None,
                from_date: str = None, to_date: str = None) -> ToolResult:
        """Compare a metric across ledgers (GlDesc) or, if no ledger matches, account groups (GrpDesc)."""
        as_of_date, from_date, to_date = _iso(as_of_date), _iso(from_date), _iso(to_date)
        as_of_date = as_of_date or date.today().isoformat()
        if not items:
            return ToolResult(success=False, error="No items to compare")

        results = {}
        for item in items:
            col, key = 'GlDesc', None
            names = self._matching_ledgers(item)
            if len(names) == 1:
                key = names[0]
            elif len(names) > 1:
                results[item] = {'error': f"Multiple ledgers match '{item}': {names[:10]}"}
                continue
            else:
                g = self.executor.execute(
                    "SELECT DISTINCT GrpDesc FROM data WHERE LOWER(GrpDesc) = LOWER(?)", (item.strip(),))
                if g.success and g.data:
                    col, key = 'GrpDesc', g.data[0]['GrpDesc']
            if key is None:
                results[item] = {'error': f"No match for '{item}'"}
                continue

            if from_date and to_date and metric != 'balance':
                where, params = f"{col} = ? AND VDate_iso BETWEEN ? AND ?", (key, from_date, to_date)
            else:
                where, params = f"{col} = ? AND VDate_iso <= ?", (key, to_date or as_of_date)
            r = self.executor.execute(
                f"SELECT COALESCE(SUM(LocalDrAmt),0) AS debit, COALESCE(SUM(LocalCrAmt),0) AS credit, "
                f"COALESCE(SUM(LocalDrAmt - LocalCrAmt),0) AS net FROM data WHERE {where}", params)
            if r.success and r.data:
                row = r.data[0]
                value = row['debit'] if metric == 'debit' else row['credit'] if metric == 'credit' else row['net']
                results[key] = {'matched_name': key, 'matched_on': col, 'value': _r2(value),
                                'debit': _r2(row['debit']), 'credit': _r2(row['credit']), 'net': _r2(row['net'])}
            else:
                results[key] = {'error': 'Query failed'}

        errors = {k: v['error'] for k, v in results.items() if 'error' in v}
        ranked = sorted([(k, v['value']) for k, v in results.items() if 'value' in v],
                        key=lambda x: x[1], reverse=True)
        comparison = [{'name': n, 'value': v, 'rank': i + 1} for i, (n, v) in enumerate(ranked)]
        if len(ranked) >= 2:
            diff = ranked[0][1] - ranked[1][1]
            pct = (diff / abs(ranked[1][1]) * 100) if ranked[1][1] != 0 else None
            comparison[0]['diff_vs_next'] = _r2(diff)
            comparison[0]['pct_vs_next'] = round(pct, 2) if pct is not None else None
            comparison[1]['diff_vs_prev'] = _r2(-diff)
            comparison[1]['pct_vs_prev'] = round(-pct, 2) if pct is not None else None
        return ToolResult(success=True, data={
            'metric': metric, 'items': comparison, 'errors': errors,
            'as_of': as_of_date, 'period': {'from': from_date, 'to': to_date}})

    # ---------------- chart ----------------

    def make_chart(self, chart_type: str, title: str, labels: list, series: list) -> ToolResult:
        if chart_type not in ['bar', 'line', 'stacked', 'pie', 'area']:
            chart_type = 'bar'
        validated = []
        for s in series:
            if isinstance(s, dict) and 'name' in s and 'data' in s:
                validated.append({'name': s['name'], 'data': s['data'], 'color': s.get('color')})
            elif isinstance(s, list):
                validated.append({'name': 'Series', 'data': s})
        return ToolResult(success=True, data={'type': chart_type, 'title': title,
                                              'labels': labels, 'series': validated})

    # ---------------- export ----------------

    def export_table(self, data: list, format: str = 'csv') -> ToolResult:
        if not data:
            return ToolResult(success=False, error="No data to export")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        headers = list(data[0].keys())

        def pack(fmt, raw):
            return ToolResult(success=True, data={
                'format': fmt, 'content': base64.b64encode(raw).decode('utf-8'),
                'filename': f'export_{stamp}.{fmt}'})

        if format == 'csv':
            out = io.StringIO()
            w = csv.DictWriter(out, fieldnames=headers)
            w.writeheader()
            w.writerows(data)
            return pack('csv', out.getvalue().encode('utf-8'))

        if format == 'xlsx':
            try:
                from openpyxl import Workbook
                from openpyxl.styles import Font, Alignment
                wb = Workbook()
                ws = wb.active
                for ci, h in enumerate(headers, 1):
                    c = ws.cell(row=1, column=ci, value=h)
                    c.font = Font(bold=True)
                    c.alignment = Alignment(horizontal='center')
                for ri, row in enumerate(data, 2):
                    for ci, h in enumerate(headers, 1):
                        c = ws.cell(row=ri, column=ci, value=row.get(h))
                        if isinstance(row.get(h), (int, float)):
                            c.number_format = '#,##0.00'
                            c.alignment = Alignment(horizontal='right')
                for ci in range(1, len(headers) + 1):
                    ws.column_dimensions[ws.cell(row=1, column=ci).column_letter].width = 18
                buf = io.BytesIO()
                wb.save(buf)
                return pack('xlsx', buf.getvalue())
            except ImportError:
                return ToolResult(success=False, error="openpyxl not installed, falling back to CSV")

        if format == 'docx':
            try:
                from docx import Document
                doc = Document()
                doc.add_heading('Data Export', 0)
                table = doc.add_table(rows=1, cols=len(headers))
                table.style = 'Table Grid'
                for i, h in enumerate(headers):
                    table.rows[0].cells[i].text = h
                for row_data in data:
                    row = table.add_row()
                    for i, h in enumerate(headers):
                        row.cells[i].text = str(row_data.get(h))
                buf = io.BytesIO()
                doc.save(buf)
                return pack('docx', buf.getvalue())
            except ImportError:
                return ToolResult(success=False, error="python-docx not installed, falling back to CSV")

        if format == 'pptx':
            try:
                from pptx import Presentation
                from pptx.util import Inches, Pt
                prs = Presentation()
                slide = prs.slides.add_slide(prs.slide_layouts[5])
                rows_count = min(len(data) + 1, 20)
                table = slide.shapes.add_table(rows_count, len(headers), Inches(0.5), Inches(0.5),
                                               Inches(9), Inches(6)).table
                for i, h in enumerate(headers):
                    cell = table.cell(0, i)
                    cell.text = h
                    for p in cell.text_frame.paragraphs:
                        p.font.size = Pt(10)
                        p.font.bold = True
                for ri, row_data in enumerate(data[:rows_count - 1], 1):
                    for ci, h in enumerate(headers):
                        cell = table.cell(ri, ci)
                        cell.text = str(row_data.get(h))
                        for p in cell.text_frame.paragraphs:
                            p.font.size = Pt(9)
                buf = io.BytesIO()
                prs.save(buf)
                return pack('pptx', buf.getvalue())
            except ImportError:
                return ToolResult(success=False, error="python-pptx not installed, falling back to CSV")

        if format == 'pdf':
            try:
                from reportlab.lib.pagesizes import A4, landscape
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
                from reportlab.lib import colors
                buf = io.BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=landscape(A4))
                table_data = [headers] + [[str(r.get(h)) for h in headers] for r in data]
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                doc.build([table])
                return pack('pdf', buf.getvalue())
            except ImportError:
                return ToolResult(success=False, error="reportlab not installed, falling back to CSV")

        return ToolResult(success=False, error=f"Unknown format: {format}")


# ----------------------------------------------------------------------------
# Tool definitions for OpenAI function calling
# ----------------------------------------------------------------------------

_LEDGER_DESC = "Ledger/party name (GlDesc). Exact match preferred; every-word and punctuation-insensitive match also works."

TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "ledger_report",
        "description": ("Complete dashboard-style report for one ledger/party (GlDesc): sales, FIFO outstanding, aging, "
                        "receivable metrics, source breakup, monthly/weekly activity, and pending bills. Use only when the user "
                        "asks for a complete/overall report. For a specific metric or detail, use the relevant focused tool instead."),
        "parameters": {"type": "object", "properties": {
            "ledger": {"type": "string", "description": _LEDGER_DESC},
            "as_of_date": {"type": "string", "description": "As of date (YYYY-MM-DD). Default today."}},
            "required": ["ledger"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Execute a read-only SQL query on the company data. Use for filtering, aggregation, sorting, custom calculations. Returns up to 200 rows - use SUM()/COUNT() in SQL for totals, never add rows yourself.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "SQL SELECT query (or WITH). Only one statement. Use VDate_iso for date filtering. Use net_amount = LocalDrAmt - LocalCrAmt."}},
            "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "ledger_statement",
        "description": "Ledger statement with running balance for a specific party/ledger. Opening balance = every entry before from_date.",
        "parameters": {"type": "object", "properties": {
            "ledger": {"type": "string", "description": _LEDGER_DESC},
            "from_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
            "to_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}},
            "required": ["ledger", "from_date", "to_date"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "trial_balance",
        "description": "Trial balance as of a date, grouped by ledger, group, or category.",
        "parameters": {"type": "object", "properties": {
            "as_of_date": {"type": "string", "description": "As of date (YYYY-MM-DD)"},
            "group_by": {"type": "string", "enum": ["ledger", "group", "category"], "description": "Grouping level"}},
            "required": ["as_of_date"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "balance_sheet_pnl",
        "description": "Balance Sheet and P&L using confirmed group mappings. Provide as_of_date for BS, or from_date/to_date for P&L.",
        "parameters": {"type": "object", "properties": {
            "as_of_date": {"type": "string", "description": "Balance sheet as of date (YYYY-MM-DD)"},
            "from_date": {"type": "string", "description": "P&L period start (YYYY-MM-DD)"},
            "to_date": {"type": "string", "description": "P&L period end (YYYY-MM-DD)"}},
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "fifo_outstanding",
        "description": "Outstanding bills by FIFO with aging for one ledger (any GlDesc) or 'ALL' customers/vendors. All credits of the party (every Source) are applied to the oldest bills first.",
        "parameters": {"type": "object", "properties": {
            "ledger": {"type": "string", "description": "Ledger name or 'ALL'"},
            "as_of_date": {"type": "string", "description": "As of date for aging (YYYY-MM-DD)"}},
            "required": ["ledger"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "receivables_metrics",
        "description": "Turnover ratio, average collection days, average (daily) receivable and net sales for one ledger (any GlDesc) or 'ALL'.",
        "parameters": {"type": "object", "properties": {
            "ledger": {"type": "string", "description": "Ledger name or 'ALL'"},
            "as_of_date": {"type": "string", "description": "As of date (YYYY-MM-DD)"}},
            "required": ["ledger"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "period_series",
        "description": "Time series of debit/credit/net/balance by day/week/month/quarter/year. Optional filters: source (e.g. 'SB' or 'SB,CB') and ledger (GlDesc). Example: monthly sales of one party = metric debit, source SB, ledger name.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": ["debit", "credit", "net", "balance"], "description": "balance is cumulative"},
            "grain": {"type": "string", "enum": ["day", "week", "month", "quarter", "year"], "description": "Time grain"},
            "from_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
            "to_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
            "group_by": {"type": "string", "enum": ["ledger", "group", "category", "source"], "description": "Optional grouping"},
            "source": {"type": "string", "description": "Optional Source code filter, comma separated"},
            "ledger": {"type": "string", "description": "Optional ledger (GlDesc) filter"}},
            "required": ["metric", "grain"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "compare",
        "description": "Compare a metric across multiple ledgers (GlDesc) or account groups (GrpDesc) side by side.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": ["debit", "credit", "net", "balance"], "description": "Metric to compare"},
            "items": {"type": "array", "items": {"type": "string"}, "description": "Ledger or group names"},
            "as_of_date": {"type": "string", "description": "As of date (YYYY-MM-DD)"},
            "from_date": {"type": "string", "description": "Period start (YYYY-MM-DD)"},
            "to_date": {"type": "string", "description": "Period end (YYYY-MM-DD)"}},
            "required": ["metric", "items"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "make_chart",
        "description": "Create chart configuration for frontend rendering.",
        "parameters": {"type": "object", "properties": {
            "chart_type": {"type": "string", "enum": ["bar", "line", "stacked", "pie", "area"], "description": "Chart type"},
            "title": {"type": "string", "description": "Chart title"},
            "labels": {"type": "array", "items": {"type": "string"}, "description": "X-axis labels"},
            "series": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "data": {"type": "array", "items": {"type": "number"}},
                "color": {"type": "string"}}, "required": ["name", "data"]}, "description": "Data series"}},
            "required": ["chart_type", "title", "labels", "series"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "export_table",
        "description": "Export table data in various formats.",
        "parameters": {"type": "object", "properties": {
            "data": {"type": "array", "items": {"type": "object"}, "description": "Table data as array of objects"},
            "format": {"type": "string", "enum": ["csv", "xlsx", "docx", "pptx", "pdf"], "description": "Export format"}},
            "required": ["data", "format"], "additionalProperties": False}}},
]