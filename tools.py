"""
Tools module for the Chat with Company Data application.
Provides all the SQL and analytical tools that the LLM can call.
"""

import json
import sqlite3
import re
from datetime import datetime, date
from typing import Any, Optional
from dataclasses import dataclass, asdict


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


class SQLiteExecutor:
    """Safe read-only SQLite executor with authorization."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = None
        self.max_rows = 200
        self.timeout = 15  # seconds

    def connect(self, read_only: bool = True) -> sqlite3.Connection:
        """Return the connection and enable the read-only guard when requested.

        Dataset setup needs one short, internal write phase before analytical
        queries are exposed. All normal callers (including ``execute``) retain
        the read-only authorizer.
        """
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            # Set timeout
            self.conn.execute("PRAGMA busy_timeout = 15000")
        if read_only:
            self.conn.set_authorizer(self._authorizer)
        else:
            # write phase (e.g. dataset reload): clear any read-only authorizer
            self.conn.set_authorizer(None)
        return self.conn

    def _authorizer(self, action, arg1, arg2, dbname, trigger):
        """SQLite authorizer callback - only allow SELECT."""
        # Allow the SQLite operations required to evaluate read-only queries.
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_PRAGMA):
            return sqlite3.SQLITE_OK
        # Deny all modifications
        return sqlite3.SQLITE_DENY

    def execute(self, query: str, params: tuple = (), max_rows: Optional[int] = None) -> ToolResult:
        """Execute a read-only query.

        `max_rows` optionally overrides the per-query cap; pass a larger value
        to fetch more rows. None keeps existing behavior.
        """
        # Validate query is read-only
        if not self._is_read_only(query):
            return ToolResult(success=False, error="Only SELECT queries are allowed")

        # Prevent multiple statements
        if query.count(';') > 1 or (query.count(';') == 1 and not query.strip().endswith(';')):
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

            # Convert to list of dicts
            data = [dict(row) for row in rows]

            return ToolResult(
                success=True,
                data=data,
                row_count=len(data),
                truncated=truncated
            )
        except sqlite3.Error as e:
            return ToolResult(success=False, error=f"SQL error: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Execution error: {str(e)}")

    def _is_read_only(self, query: str) -> bool:
        """Check if query is read-only."""
        query_upper = query.upper().strip()
        # Must start with SELECT or WITH
        if not (query_upper.startswith('SELECT') or query_upper.startswith('WITH')):
            return False
        # Check for forbidden keywords
        forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE',
                     'REPLACE', 'ATTACH', 'DETACH', 'VACUUM', 'PRAGMA']
        # Allow PRAGMA for read-only
        for word in forbidden:
            if word == 'PRAGMA':
                continue
            if re.search(rf'\b{word}\b', query_upper):
                return False
        return True

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


class DataTools:
    """Collection of all analytical tools for the chat application."""

    def __init__(self, executor: SQLiteExecutor, settings: dict):
        self.executor = executor
        self.settings = settings
        self.max_rows = settings.get('max_rows', 200)

    def run_sql(self, query: str) -> ToolResult:
        """Execute a read-only SQL query."""
        return self.executor.execute(query)

    def _matching_ledgers(self, ledger: str, limit: int = 50) -> list:
        """Return GlDesc names matching `ledger`, exact (case-insensitive) match first.

        Matching is additive convenience only: callers keep their existing
        fallback behavior when nothing matches.
        """
        ledger = (ledger or '').strip()
        if not ledger:
            return []
        match_result = self.executor.execute(
            "SELECT DISTINCT GlDesc FROM data WHERE LOWER(GlDesc) LIKE LOWER(?) LIMIT ?",
            (f"%{ledger}%", limit),
        )
        if not match_result.success or not match_result.data:
            return []
        names = [row['GlDesc'] for row in match_result.data]
        lowered = ledger.lower()
        exact = [n for n in names if n.lower() == lowered]
        return exact + [n for n in names if n.lower() != lowered]

    def _matching_ledger(self, ledger: str):
        """Return single best GlDesc match or None (exact case-insensitive preferred)."""
        names = self._matching_ledgers(ledger)
        return names[0] if names else None

    def ledger_statement(self, ledger: str, from_date: str, to_date: str) -> ToolResult:
        """Get ledger statement with running balance."""
        # Find matching ledger (fuzzy); exact case-insensitive match preferred
        names = self._matching_ledgers(ledger)
        if not names:
            return ToolResult(success=False, error=f"No ledger found matching '{ledger}'")

        lowered = (ledger or '').strip().lower()
        exact = [n for n in names if n.lower() == lowered]
        if exact:
            matched_ledger = exact[0]
        elif len(names) == 1:
            matched_ledger = names[0]
        else:
            return ToolResult(
                success=False,
                error=f"Multiple ledgers match '{ledger}': {names}. Please specify more precisely.",
                data={'options': names}
            )

        # Get opening balance (OB entries before from_date)
        ob_query = """
            SELECT
                COALESCE(SUM(LocalDrAmt), 0) as ob_dr,
                COALESCE(SUM(LocalCrAmt), 0) as ob_cr
            FROM data
            WHERE GlDesc = ? AND Source = 'OB' AND VDate_iso < ?
        """
        ob_result = self.executor.execute(ob_query, (matched_ledger, from_date))
        ob_dr = ob_result.data[0]['ob_dr'] if ob_result.success and ob_result.data else 0
        ob_cr = ob_result.data[0]['ob_cr'] if ob_result.success and ob_result.data else 0
        opening_balance = ob_dr - ob_cr

        # Get period entries
        period_query = """
            SELECT
                VDate_iso as date,
                VNo as voucher,
                Source,
                Narration,
                LocalDrAmt as debit,
                LocalCrAmt as credit,
                (LocalDrAmt - LocalCrAmt) as net
            FROM data
            WHERE GlDesc = ? AND VDate_iso BETWEEN ? AND ?
            ORDER BY VDate_iso, VNo
        """
        period_result = self.executor.execute(period_query, (matched_ledger, from_date, to_date))

        if not period_result.success:
            return period_result

        # Calculate running balance
        running_balance = opening_balance
        entries = []
        for row in period_result.data:
            running_balance += row['net']
            entries.append({
                'date': row['date'],
                'voucher': row['voucher'],
                'source': row.get('source') or row.get('Source'),
                'narration': row.get('narration') or row.get('Narration'),
                'debit': row['debit'],
                'credit': row['credit'],
                'balance': running_balance
            })

        closing_balance = running_balance

        return ToolResult(success=True, data={
            'ledger': matched_ledger,
            'period': {'from': from_date, 'to': to_date},
            'opening_balance': opening_balance,
            'entries': entries,
            'closing_balance': closing_balance,
            'total_debit': sum(e['debit'] for e in entries),
            'total_credit': sum(e['credit'] for e in entries)
        })

    def trial_balance(self, as_of_date: str, group_by: str = 'ledger') -> ToolResult:
        """Generate trial balance as of a date."""
        if group_by not in ('ledger', 'group', 'category'):
            group_by = 'ledger'

        group_col = {'ledger': 'GlDesc', 'group': 'GrpDesc', 'category': 'Catagory'}[group_by]

        # Opening balances (OB before as_of_date)
        ob_query = f"""
            SELECT
                {group_col} as name,
                COALESCE(SUM(LocalDrAmt), 0) as ob_dr,
                COALESCE(SUM(LocalCrAmt), 0) as ob_cr
            FROM data
            WHERE Source = 'OB' AND VDate_iso < ?
            GROUP BY {group_col}
        """
        ob_result = self.executor.execute(ob_query, (as_of_date,))
        ob_balances = {}
        if ob_result.success:
            for row in ob_result.data:
                ob_balances[row['name']] = row['ob_dr'] - row['ob_cr']

        # Period activity (non-OB entries up to as_of_date)
        period_query = f"""
            SELECT
                {group_col} as name,
                COALESCE(SUM(LocalDrAmt), 0) as period_dr,
                COALESCE(SUM(LocalCrAmt), 0) as period_cr
            FROM data
            WHERE Source != 'OB' AND VDate_iso <= ?
            GROUP BY {group_col}
        """
        period_result = self.executor.execute(period_query, (as_of_date,))

        # Combine
        all_names = set(ob_balances.keys())
        if period_result.success:
            for row in period_result.data:
                all_names.add(row['name'])

        rows = []
        total_dr = 0
        total_cr = 0

        for name in sorted(all_names):
            ob = ob_balances.get(name, 0)
            period_dr = 0
            period_cr = 0

            if period_result.success:
                for row in period_result.data:
                    if row['name'] == name:
                        period_dr = row['period_dr']
                        period_cr = row['period_cr']
                        break

            closing = ob + period_dr - period_cr

            # Determine debit/credit balance for trial balance
            if closing >= 0:
                dr_balance = closing
                cr_balance = 0
            else:
                dr_balance = 0
                cr_balance = -closing

            rows.append({
                'name': name,
                'opening': ob,
                'debit': period_dr,
                'credit': period_cr,
                'closing': closing,
                'dr_balance': dr_balance,
                'cr_balance': cr_balance
            })
            total_dr += dr_balance
            total_cr += cr_balance

        # Add totals row
        rows.append({
            'name': 'TOTAL',
            'opening': sum(r['opening'] for r in rows),
            'debit': sum(r['debit'] for r in rows),
            'credit': sum(r['credit'] for r in rows),
            'closing': sum(r['closing'] for r in rows),
            'dr_balance': total_dr,
            'cr_balance': total_cr
        })

        return ToolResult(success=True, data={
            'as_of': as_of_date,
            'group_by': group_by,
            'rows': rows,
            'balanced': abs(total_dr - total_cr) < 0.01,
            'difference': total_dr - total_cr
        })

    def balance_sheet_pnl(self, as_of_date: str = None, from_date: str = None, to_date: str = None,
                          group_mapping: dict = None) -> ToolResult:
        """Generate Balance Sheet and P&L using confirmed group mapping."""
        if not group_mapping:
            group_mapping = {}

        # Determine date range
        if as_of_date:
            # Balance sheet as of date
            date_filter = "VDate_iso <= ?"
            date_params = (as_of_date,)
        elif from_date and to_date:
            # P&L for period
            date_filter = "VDate_iso BETWEEN ? AND ?"
            date_params = (from_date, to_date)
        else:
            # Default to all data
            date_filter = "1=1"
            date_params = ()

        # Get all groups with their kinds
        query = f"""
            SELECT
                GrpDesc as group_name,
                COALESCE(SUM(LocalDrAmt), 0) as total_dr,
                COALESCE(SUM(LocalCrAmt), 0) as total_cr
            FROM data
            WHERE {date_filter}
            GROUP BY GrpDesc
        """
        result = self.executor.execute(query, date_params)

        if not result.success:
            return result

        # Classify groups
        assets = []
        liabilities = []
        equity = []
        income = []
        expenses = []
        unclassified = []

        for row in result.data:
            group = row['group_name']
            dr = row['total_dr']
            cr = row['total_cr']
            net = dr - cr

            kind_info = group_mapping.get(group, {})
            kind = kind_info.get('kind', 'other')

            entry = {
                'group': group,
                'debit': dr,
                'credit': cr,
                'net': net
            }

            if kind == 'asset':
                assets.append(entry)
            elif kind == 'liability':
                liabilities.append(entry)
            elif kind == 'equity':
                equity.append(entry)
            elif kind == 'income':
                income.append(entry)
            elif kind == 'expense':
                expenses.append(entry)
            else:
                # Heuristic: if net is typically debit, it's asset-like; if credit, liability-like
                if net >= 0:
                    assets.append({**entry, 'assumed': 'asset'})
                else:
                    liabilities.append({**entry, 'assumed': 'liability'})

        # Calculate totals
        total_assets = sum(a['net'] for a in assets)
        total_liabilities = sum(l['net'] for l in liabilities)
        total_equity = sum(e['net'] for e in equity)
        total_income = sum(i['net'] for i in income)
        total_expenses = sum(e['net'] for e in expenses)

        # For balance sheet: Assets = Liabilities + Equity
        # For P&L: Net Profit = Income - Expenses
        net_profit = total_income - total_expenses

        # Balance sheet check
        bs_balance = total_assets - (total_liabilities + total_equity + net_profit)

        warnings = []
        if abs(bs_balance) > 1.0:
            warnings.append(f"Balance sheet mismatch: {bs_balance:.2f} (Assets - Liabilities - Equity - P&L)")

        return ToolResult(success=True, data={
            'as_of': as_of_date,
            'period': {'from': from_date, 'to': to_date},
            'balance_sheet': {
                'assets': assets,
                'liabilities': liabilities,
                'equity': equity,
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'total_equity': total_equity + net_profit  # Equity includes current period profit
            },
            'profit_loss': {
                'income': income,
                'expenses': expenses,
                'total_income': total_income,
                'total_expenses': total_expenses,
                'net_profit': net_profit
            },
            'balanced': abs(bs_balance) < 1.0,
            'mismatch': bs_balance,
            'warnings': warnings
        })

    def fifo_outstanding(self, ledger: str = 'ALL', as_of_date: str = None) -> ToolResult:
        """Calculate outstanding bills using FIFO method."""
        if as_of_date is None:
            as_of_date = date.today().isoformat()

        # Find bills (typically Source = 'SB' or 'PB' for sales/purchase invoices)
        # and payments (Source = 'CB' or 'PAY' for collections/payments)

        if ledger == 'ALL':
            ledger_filter = ""
            ledger_params = (as_of_date,)
        else:
            # Fuzzy match ledger; exact case-insensitive match preferred
            matched = self._matching_ledger(ledger)
            if not matched:
                return ToolResult(success=False, error=f"No ledger found matching '{ledger}'")
            ledger_filter = "AND GlDesc = ?"
            # NOTE: SQL order is `VDate_iso <= ?` first, then `AND GlDesc = ?`
            ledger_params = (as_of_date, matched)

        # Get all invoices (debit entries for sales, credit for purchase)
        invoice_query = f"""
            SELECT
                VNo, VDate_iso, GlDesc, Source,
                LocalDrAmt as debit, LocalCrAmt as credit,
                Narration
            FROM data
            WHERE (Source IN ('SB', 'PB', 'SI', 'PI', 'INV', 'PURCHASE'))
            AND VDate_iso <= ?
            {ledger_filter}
            ORDER BY GlDesc, VDate_iso, VNo
        """
        invoice_result = self.executor.execute(invoice_query, ledger_params, max_rows=1_000_000)

        if not invoice_result.success:
            return invoice_result

        # Get all payments/collections
        payment_query = f"""
            SELECT
                VNo, VDate_iso, GlDesc, Source,
                LocalDrAmt as debit, LocalCrAmt as credit,
                Narration
            FROM data
            WHERE (Source IN ('CB', 'PAY', 'RECEIPT', 'PAYMENT', 'COLLECTION'))
            AND VDate_iso <= ?
            {ledger_filter}
            ORDER BY GlDesc, VDate_iso, VNo
        """
        payment_result = self.executor.execute(payment_query, ledger_params, max_rows=1_000_000)

        if not payment_result.success:
            return payment_result

        # Group by ledger and apply FIFO
        from collections import defaultdict

        invoices_by_ledger = defaultdict(list)
        for row in invoice_result.data:
            invoices_by_ledger[row['GlDesc']].append({
                'vno': row['VNo'],
                'date': row['VDate_iso'],
                'source': row['Source'],
                'amount': row['debit'] if row['debit'] > 0 else row['credit'],
                'narration': row['Narration'],
                'remaining': row['debit'] if row['debit'] > 0 else row['credit']
            })

        payments_by_ledger = defaultdict(list)
        for row in payment_result.data:
            payments_by_ledger[row['GlDesc']].append({
                'vno': row['VNo'],
                'date': row['VDate_iso'],
                'source': row['Source'],
                'amount': row['credit'] if row['credit'] > 0 else row['debit'],
                'narration': row['Narration'],
                'remaining': row['credit'] if row['credit'] > 0 else row['debit']
            })

        # FIFO matching
        outstanding = []
        for ledger_name, inv_list in invoices_by_ledger.items():
            pay_list = payments_by_ledger.get(ledger_name, [])
            pay_idx = 0

            for inv in inv_list:
                inv_remaining = inv['remaining']

                while inv_remaining > 0 and pay_idx < len(pay_list):
                    pay = pay_list[pay_idx]
                    pay_remaining = pay['remaining']

                    if pay_remaining <= inv_remaining:
                        inv_remaining -= pay_remaining
                        pay_list[pay_idx]['remaining'] = 0
                        pay_idx += 1
                    else:
                        pay_list[pay_idx]['remaining'] -= inv_remaining
                        inv_remaining = 0

                if inv_remaining > 0:
                    # Calculate aging
                    inv_date = datetime.strptime(inv['date'], '%Y-%m-%d').date()
                    as_of = datetime.strptime(as_of_date, '%Y-%m-%d').date()
                    days_outstanding = (as_of - inv_date).days

                    # Aging slab
                    slabs = self.settings.get('aging_slabs', '0-30,31-60,61-90,91-180,181-365,365+')
                    slab = self._get_aging_slab(days_outstanding, slabs)

                    outstanding.append({
                        'ledger': ledger_name,
                        'voucher': inv['vno'],
                        'date': inv['date'],
                        'original_amount': inv['amount'],
                        'outstanding': inv_remaining,
                        'days_outstanding': days_outstanding,
                        'aging_slab': slab,
                        'narration': inv['narration']
                    })

        # Summary by aging slab
        slab_summary = defaultdict(lambda: {'count': 0, 'amount': 0})
        for item in outstanding:
            slab_summary[item['aging_slab']]['count'] += 1
            slab_summary[item['aging_slab']]['amount'] += item['outstanding']

        return ToolResult(success=True, data={
            'as_of': as_of_date,
            'ledger': ledger,
            'outstanding': outstanding,
            'summary_by_slab': dict(slab_summary),
            'total_outstanding': sum(item['outstanding'] for item in outstanding),
            'total_count': len(outstanding)
        })

    def _get_aging_slab(self, days: int, slabs_str: str) -> str:
        """Determine aging slab for given days."""
        slabs = []
        for part in slabs_str.split(','):
            part = part.strip()
            if part.endswith('+'):
                # open-ended slab like "365+"
                slabs.append((int(part[:-1]), float('inf'), part))
            elif '-' in part:
                start, end = part.split('-')
                if end.endswith('+'):
                    slabs.append((int(start), float('inf'), part))
                else:
                    slabs.append((int(start), int(end), part))
            else:
                slabs.append((int(part), int(part), part))

        for start, end, label in slabs:
            if start <= days <= end:
                return label
        return slabs[-1][2] if slabs else '365+'

    def receivables_metrics(self, ledger: str = 'ALL', as_of_date: str = None) -> ToolResult:
        """Calculate receivables metrics: turnover, collection days, etc."""
        if as_of_date is None:
            as_of_date = date.today().isoformat()

        turnover_days = self.settings.get('turnover_days', 365)

        if ledger == 'ALL':
            ledger_filter = ""
            ledger_params = (as_of_date,)
        else:
            # Fuzzy match ledger; exact case-insensitive match preferred
            matched = self._matching_ledger(ledger)
            if not matched:
                return ToolResult(success=False, error=f"No ledger found matching '{ledger}'")
            ledger_filter = "AND GlDesc = ?"
            # NOTE: SQL order is `VDate_iso <= ?` first, then `AND GlDesc = ?`
            ledger_params = (as_of_date, matched)

        # Net sales (credit entries from sales sources)
        # Additive: also return sales_debit (SUM LocalDrAmt on sales sources) so
        # datasets that record customer sales as debit entries (SB on customer
        # ledger, revenue on SALES A/C) still get a usable net-sales figure.
        sales_query = f"""
            SELECT
                GlDesc,
                COALESCE(SUM(LocalCrAmt), 0) as sales,
                COALESCE(SUM(LocalDrAmt), 0) as sales_debit
            FROM data
            WHERE Source IN ('SB', 'SI', 'INV', 'SALES')
            AND VDate_iso <= ?
            {ledger_filter}
            GROUP BY GlDesc
        """
        sales_result = self.executor.execute(sales_query, ledger_params)

        # Average receivable (daily average balance)
        # This is complex - simplified: average of opening and closing
        # For a more accurate daily average, we'd need daily balances

        # Get current outstanding (closing balance for receivable ledgers)
        balance_query = f"""
            SELECT
                GlDesc,
                COALESCE(SUM(LocalDrAmt - LocalCrAmt), 0) as balance
            FROM data
            WHERE VDate_iso <= ?
            {ledger_filter}
            GROUP BY GlDesc
        """
        balance_result = self.executor.execute(balance_query, ledger_params)

        if not sales_result.success or not balance_result.success:
            return ToolResult(success=False, error="Failed to calculate metrics")

        # Build metrics per ledger
        sales_by_ledger = {row['GlDesc']: row['sales'] for row in sales_result.data}
        sales_debit_by_ledger = {row['GlDesc']: row.get('sales_debit', 0) for row in sales_result.data}
        balance_by_ledger = {row['GlDesc']: row['balance'] for row in balance_result.data}

        all_ledgers = set(sales_by_ledger.keys()) | set(balance_by_ledger.keys())

        specific_ledger = ledger != 'ALL'

        metrics = []
        for lg in all_ledgers:
            net_sales = sales_by_ledger.get(lg, 0)
            sales_debit = sales_debit_by_ledger.get(lg, 0)
            closing_balance = balance_by_ledger.get(lg, 0)

            # Simplified average receivable (opening + closing) / 2
            # Opening would need separate query - using 0 for simplicity
            avg_receivable = closing_balance / 2 if closing_balance > 0 else 0

            # Additive: net sales invoiced to this ledger (sales-source debits).
            # net_sales (credit-based) output is unchanged; turnover uses it when
            # present, else falls back to invoiced amount for specific ledgers.
            net_sales_invoiced = sales_debit
            sales_for_turnover = net_sales
            if sales_for_turnover <= 0 and specific_ledger and net_sales_invoiced > 0:
                sales_for_turnover = net_sales_invoiced

            turnover_ratio = sales_for_turnover / avg_receivable if avg_receivable > 0 else 0
            turnover_days_calc = turnover_days / turnover_ratio if turnover_ratio > 0 else 0

            metrics.append({
                'ledger': lg,
                'net_sales': net_sales,
                'net_sales_invoiced': net_sales_invoiced,
                'net_sales_effective': sales_for_turnover,
                'avg_receivable': avg_receivable,
                'turnover_ratio': round(turnover_ratio, 2),
                'turnover_days': round(turnover_days_calc, 1),
                'closing_balance': closing_balance
            })

        return ToolResult(success=True, data={
            'as_of': as_of_date,
            'metrics': metrics,
            'formulas': {
                'turnover_ratio': 'Net Sales / Average Receivable',
                'turnover_days': f'{turnover_days} / Turnover Ratio',
                'avg_collection_days': 'Approximated as Turnover Days'
            }
        })

    def period_series(self, metric: str, grain: str = 'month', from_date: str = None,
                      to_date: str = None, group_by: str = None) -> ToolResult:
        """Generate time series data for a metric."""
        # Default date range
        if not from_date:
            min_query = "SELECT MIN(VDate_iso) as min_date FROM data WHERE VDate_iso IS NOT NULL"
            min_result = self.executor.execute(min_query)
            from_date = min_result.data[0]['min_date'] if min_result.success and min_result.data else '2020-01-01'
        if not to_date:
            max_query = "SELECT MAX(VDate_iso) as max_date FROM data WHERE VDate_iso IS NOT NULL"
            max_result = self.executor.execute(max_query)
            to_date = max_result.data[0]['max_date'] if max_result.success and max_result.data else date.today().isoformat()

        # Date grouping
        grain_sql = {
            'day': "strftime('%Y-%m-%d', VDate_iso)",
            'week': "strftime('%Y-%W', VDate_iso)",
            'month': "strftime('%Y-%m', VDate_iso)",
            'quarter': "strftime('%Y-Q', VDate_iso) || ((strftime('%m', VDate_iso) - 1) / 3 + 1)",
            'year': "strftime('%Y', VDate_iso)"
        }.get(grain, "strftime('%Y-%m', VDate_iso)")

        group_col = group_by if group_by in ('GlDesc', 'GrpDesc', 'Catagory', 'Source') else None

        if metric in ('debit', 'credit', 'net', 'balance'):
            if metric == 'debit':
                agg = "SUM(LocalDrAmt)"
            elif metric == 'credit':
                agg = "SUM(LocalCrAmt)"
            elif metric == 'net':
                agg = "SUM(LocalDrAmt - LocalCrAmt)"
            else:  # balance
                agg = "SUM(LocalDrAmt - LocalCrAmt)"  # Running balance would need window function

            if group_col:
                query = f"""
                    SELECT
                        {grain_sql} as period,
                        {group_col} as group_name,
                        {agg} as value
                    FROM data
                    WHERE VDate_iso BETWEEN ? AND ?
                    GROUP BY period, group_name
                    ORDER BY period, group_name
                """
            else:
                query = f"""
                    SELECT
                        {grain_sql} as period,
                        {agg} as value
                    FROM data
                    WHERE VDate_iso BETWEEN ? AND ?
                    GROUP BY period
                    ORDER BY period
                """
        else:
            return ToolResult(success=False, error=f"Unknown metric: {metric}")

        result = self.executor.execute(query, (from_date, to_date))

        if not result.success:
            return result

        # Fill zero periods
        periods = self._generate_periods(from_date, to_date, grain)

        if group_col:
            groups = sorted(set(row['group_name'] for row in result.data))
            series = {}
            for g in groups:
                series[g] = []
                for p in periods:
                    val = 0
                    for row in result.data:
                        if row['period'] == p and row['group_name'] == g:
                            val = row['value']
                            break
                    series[g].append({'period': p, 'value': val})

            return ToolResult(success=True, data={
                'metric': metric,
                'grain': grain,
                'from': from_date,
                'to': to_date,
                'groups': groups,
                'series': series,
                'periods': periods
            })
        else:
            values = []
            for p in periods:
                val = 0
                for row in result.data:
                    if row['period'] == p:
                        val = row['value']
                        break
                values.append({'period': p, 'value': val})

            return ToolResult(success=True, data={
                'metric': metric,
                'grain': grain,
                'from': from_date,
                'to': to_date,
                'series': values,
                'periods': periods
            })

    def _generate_periods(self, from_date: str, to_date: str, grain: str) -> list[str]:
        """Generate list of all periods in range."""
        from_dt = datetime.strptime(from_date, '%Y-%m-%d')
        to_dt = datetime.strptime(to_date, '%Y-%m-%d')
        periods = []

        current = from_dt
        while current <= to_dt:
            if grain == 'day':
                periods.append(current.strftime('%Y-%m-%d'))
                from datetime import timedelta
                current += timedelta(days=1)
            elif grain == 'week':
                periods.append(current.strftime('%Y-%W'))
                from datetime import timedelta
                current += timedelta(weeks=1)
            elif grain == 'month':
                periods.append(current.strftime('%Y-%m'))
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            elif grain == 'quarter':
                q = (current.month - 1) // 3 + 1
                periods.append(f"{current.year}-Q{q}")
                current = current.replace(month=((q) * 3) + 1)
                if current.month > 12:
                    current = current.replace(year=current.year + 1, month=1)
            elif grain == 'year':
                periods.append(current.strftime('%Y'))
                current = current.replace(year=current.year + 1)
            else:
                break

        return periods

    def compare(self, metric: str, items: list[str], as_of_date: str = None,
                from_date: str = None, to_date: str = None) -> ToolResult:
        """Compare metric across multiple items (ledgers, groups, periods)."""
        if as_of_date is None:
            as_of_date = date.today().isoformat()

        if not items:
            return ToolResult(success=False, error="No items to compare")

        # Build query for each item
        results = {}
        for item in items:
            # Fuzzy match; exact case-insensitive match preferred
            matched = self._matching_ledger(item)

            if not matched:
                results[item] = {'error': f"No match for '{item}'"}
                continue

            if from_date and to_date:
                query = """
                    SELECT
                        COALESCE(SUM(LocalDrAmt), 0) as debit,
                        COALESCE(SUM(LocalCrAmt), 0) as credit,
                        COALESCE(SUM(LocalDrAmt - LocalCrAmt), 0) as net
                    FROM data
                    WHERE GlDesc = ? AND VDate_iso BETWEEN ? AND ?
                """
                params = (matched, from_date, to_date)
            else:
                query = """
                    SELECT
                        COALESCE(SUM(LocalDrAmt), 0) as debit,
                        COALESCE(SUM(LocalCrAmt), 0) as credit,
                        COALESCE(SUM(LocalDrAmt - LocalCrAmt), 0) as net
                    FROM data
                    WHERE GlDesc = ? AND VDate_iso <= ?
                """
                params = (matched, as_of_date)

            result = self.executor.execute(query, params)
            if result.success and result.data:
                row = result.data[0]
                if metric == 'debit':
                    value = row['debit']
                elif metric == 'credit':
                    value = row['credit']
                elif metric == 'net':
                    value = row['net']
                elif metric == 'balance':
                    value = row['net']
                else:
                    value = row['net']

                results[matched] = {
                    'matched_name': matched,
                    'value': value,
                    'debit': row['debit'],
                    'credit': row['credit'],
                    'net': row['net']
                }
            else:
                results[matched] = {'error': 'Query failed'}

        # Calculate comparisons
        comparison = []
        item_values = [(k, v.get('value', 0)) for k, v in results.items() if 'value' in v]
        item_values.sort(key=lambda x: x[1], reverse=True)

        for i, (name, value) in enumerate(item_values):
            comparison.append({
                'name': name,
                'value': value,
                'rank': i + 1
            })

        # Add diff and % change for first two if available
        if len(item_values) >= 2:
            first_val = item_values[0][1]
            second_val = item_values[1][1]
            diff = first_val - second_val
            pct = (diff / abs(second_val) * 100) if second_val != 0 else 0
            comparison[0]['diff_vs_next'] = diff
            comparison[0]['pct_vs_next'] = round(pct, 2)
            comparison[1]['diff_vs_prev'] = -diff
            comparison[1]['pct_vs_prev'] = round(-pct, 2)

        return ToolResult(success=True, data={
            'metric': metric,
            'items': comparison,
            'as_of': as_of_date,
            'period': {'from': from_date, 'to': to_date}
        })

    def make_chart(self, chart_type: str, title: str, labels: list, series: list[dict]) -> ToolResult:
        """Generate chart data for frontend rendering."""
        # Validate chart type
        valid_types = ['bar', 'line', 'stacked', 'pie', 'area']
        if chart_type not in valid_types:
            chart_type = 'bar'

        # Validate series format
        validated_series = []
        for s in series:
            if isinstance(s, dict) and 'name' in s and 'data' in s:
                validated_series.append({
                    'name': s['name'],
                    'data': s['data'],
                    'color': s.get('color')
                })
            elif isinstance(s, list):
                validated_series.append({'name': 'Series', 'data': s})

        return ToolResult(success=True, data={
            'type': chart_type,
            'title': title,
            'labels': labels,
            'series': validated_series
        })

    def export_table(self, data: list[dict], format: str = 'csv') -> ToolResult:
        """Export table data in various formats."""
        if not data:
            return ToolResult(success=False, error="No data to export")

        if format == 'csv':
            import csv
            import io
            output = io.StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            return ToolResult(success=True, data={
                'format': 'csv',
                'content': output.getvalue(),
                'filename': f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            })

        elif format == 'xlsx':
            try:
                from openpyxl import Workbook
                from openpyxl.styles import Font, Alignment, numbers
                import io

                wb = Workbook()
                ws = wb.active

                if data:
                    headers = list(data[0].keys())
                    # Write headers
                    for col_idx, header in enumerate(headers, 1):
                        cell = ws.cell(row=1, column=col_idx, value=header)
                        cell.font = Font(bold=True)
                        cell.alignment = Alignment(horizontal='center')

                    # Write data
                    for row_idx, row in enumerate(data, 2):
                        for col_idx, header in enumerate(headers, 1):
                            cell = ws.cell(row=row_idx, column=col_idx, value=row[header])
                            # Format numbers
                            if isinstance(row[header], (int, float)):
                                cell.number_format = '#,##0.00'
                                cell.alignment = Alignment(horizontal='right')

                    # Auto-fit columns
                    for col_idx in range(1, len(headers) + 1):
                        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = 18

                output = io.BytesIO()
                wb.save(output)
                output.seek(0)

                import base64
                return ToolResult(success=True, data={
                    'format': 'xlsx',
                    'content': base64.b64encode(output.read()).decode('utf-8'),
                    'filename': f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
                })
            except ImportError:
                return ToolResult(success=False, error="openpyxl not installed, falling back to CSV")

        elif format == 'docx':
            try:
                from docx import Document
                from docx.shared import Inches
                import io

                doc = Document()
                doc.add_heading('Data Export', 0)

                if data:
                    headers = list(data[0].keys())
                    table = doc.add_table(rows=1, cols=len(headers))
                    table.style = 'Table Grid'

                    # Headers
                    for i, header in enumerate(headers):
                        table.rows[0].cells[i].text = header

                    # Data
                    for row_data in data:
                        row = table.add_row()
                        for i, header in enumerate(headers):
                            row.cells[i].text = str(row_data[header])

                output = io.BytesIO()
                doc.save(output)
                output.seek(0)

                import base64
                return ToolResult(success=True, data={
                    'format': 'docx',
                    'content': base64.b64encode(output.read()).decode('utf-8'),
                    'filename': f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.docx'
                })
            except ImportError:
                return ToolResult(success=False, error="python-docx not installed, falling back to CSV")

        elif format == 'pptx':
            try:
                from pptx import Presentation
                from pptx.util import Inches, Pt
                import io

                prs = Presentation()
                slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank layout

                if data:
                    headers = list(data[0].keys())
                    rows_count = min(len(data) + 1, 20)  # Limit rows for slide
                    cols_count = len(headers)

                    table_shape = slide.shapes.add_table(
                        rows_count, cols_count,
                        Inches(0.5), Inches(0.5),
                        Inches(9), Inches(6)
                    )
                    table = table_shape.table

                    # Headers
                    for i, header in enumerate(headers):
                        cell = table.cell(0, i)
                        cell.text = header
                        for paragraph in cell.text_frame.paragraphs:
                            paragraph.font.size = Pt(10)
                            paragraph.font.bold = True

                    # Data
                    for row_idx, row_data in enumerate(data[:rows_count - 1], 1):
                        for col_idx, header in enumerate(headers):
                            cell = table.cell(row_idx, col_idx)
                            cell.text = str(row_data[header])
                            for paragraph in cell.text_frame.paragraphs:
                                paragraph.font.size = Pt(9)

                output = io.BytesIO()
                prs.save(output)
                output.seek(0)

                import base64
                return ToolResult(success=True, data={
                    'format': 'pptx',
                    'content': base64.b64encode(output.read()).decode('utf-8'),
                    'filename': f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pptx'
                })
            except ImportError:
                return ToolResult(success=False, error="python-pptx not installed, falling back to CSV")

        elif format == 'pdf':
            try:
                from reportlab.lib.pagesizes import A4, landscape
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
                from reportlab.lib import colors
                import io

                output = io.BytesIO()
                doc = SimpleDocTemplate(output, pagesize=landscape(A4))

                if data:
                    headers = list(data[0].keys())
                    table_data = [headers]
                    for row_data in data:
                        table_data.append([str(row_data[h]) for h in headers])

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

                output.seek(0)
                import base64
                return ToolResult(success=True, data={
                    'format': 'pdf',
                    'content': base64.b64encode(output.read()).decode('utf-8'),
                    'filename': f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
                })
            except ImportError:
                return ToolResult(success=False, error="reportlab not installed, falling back to CSV")

        return ToolResult(success=False, error=f"Unknown format: {format}")


# Tool definitions for OpenAI function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a read-only SQL query on the company data. Use for filtering, aggregation, sorting, custom calculations. Returns up to 200 rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query (or WITH). Only one statement. Use VDate_iso for date filtering. Use net_amount = LocalDrAmt - LocalCrAmt."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ledger_statement",
            "description": "Get detailed ledger statement with running balance for a specific party/ledger. Includes opening balance from OB entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ledger": {"type": "string", "description": "Ledger/party name (fuzzy match supported)"},
                    "from_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "to_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                },
                "required": ["ledger", "from_date", "to_date"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trial_balance",
            "description": "Generate trial balance as of a specific date, grouped by ledger, group, or category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "as_of_date": {"type": "string", "description": "As of date (YYYY-MM-DD)"},
                    "group_by": {"type": "string", "enum": ["ledger", "group", "category"], "description": "Grouping level"}
                },
                "required": ["as_of_date"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "balance_sheet_pnl",
            "description": "Generate Balance Sheet and P&L using confirmed group mappings. Provide as_of_date for BS, or from_date/to_date for P&L.",
            "parameters": {
                "type": "object",
                "properties": {
                    "as_of_date": {"type": "string", "description": "Balance sheet as of date (YYYY-MM-DD)"},
                    "from_date": {"type": "string", "description": "P&L period start (YYYY-MM-DD)"},
                    "to_date": {"type": "string", "description": "P&L period end (YYYY-MM-DD)"}
                },
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fifo_outstanding",
            "description": "Calculate outstanding bills using FIFO method with aging analysis. Requires bill-like data structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ledger": {"type": "string", "description": "Ledger name or 'ALL' for all parties"},
                    "as_of_date": {"type": "string", "description": "As of date for aging (YYYY-MM-DD)"}
                },
                "required": ["ledger"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "receivables_metrics",
            "description": "Calculate receivables metrics: turnover ratio, collection days, average receivable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ledger": {"type": "string", "description": "Ledger name or 'ALL'"},
                    "as_of_date": {"type": "string", "description": "As of date (YYYY-MM-DD)"}
                },
                "required": ["ledger"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "period_series",
            "description": "Generate time series data for a metric (debit, credit, net, balance) by period (day/week/month/quarter/year).",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["debit", "credit", "net", "balance"], "description": "Metric to chart"},
                    "grain": {"type": "string", "enum": ["day", "week", "month", "quarter", "year"], "description": "Time grain"},
                    "from_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "to_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                    "group_by": {"type": "string", "enum": ["ledger", "group", "category", "source"], "description": "Optional grouping"}
                },
                "required": ["metric", "grain"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare",
            "description": "Compare a metric across multiple items (ledgers, groups, periods) side by side with differences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["debit", "credit", "net", "balance"], "description": "Metric to compare"},
                    "items": {"type": "array", "items": {"type": "string"}, "description": "List of items to compare (ledger names, etc.)"},
                    "as_of_date": {"type": "string", "description": "As of date (YYYY-MM-DD)"},
                    "from_date": {"type": "string", "description": "Period start (YYYY-MM-DD)"},
                    "to_date": {"type": "string", "description": "Period end (YYYY-MM-DD)"}
                },
                "required": ["metric", "items"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": "Create chart configuration for frontend rendering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {"type": "string", "enum": ["bar", "line", "stacked", "pie", "area"], "description": "Chart type"},
                    "title": {"type": "string", "description": "Chart title"},
                    "labels": {"type": "array", "items": {"type": "string"}, "description": "X-axis labels"},
                    "series": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "data": {"type": "array", "items": {"type": "number"}}, "color": {"type": "string"}}, "required": ["name", "data"]}, "description": "Data series"}
                },
                "required": ["chart_type", "title", "labels", "series"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "export_table",
            "description": "Export table data in various formats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "array", "items": {"type": "object"}, "description": "Table data as array of objects"},
                    "format": {"type": "string", "enum": ["csv", "xlsx", "docx", "pptx", "pdf"], "description": "Export format"}
                },
                "required": ["data", "format"],
                "additionalProperties": False
            }
        }
    }
]
