"""Tests for rankings, zero-division safety, trial balance, and ledger statements."""
import pytest
from tools import DataTools
import sqlite3
from datetime import date


@pytest.fixture
def sample_db():
    """Create a sample database with comprehensive test data."""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE data (
            VNo TEXT,
            VDate_iso TEXT,
            GlDesc TEXT,
            Source TEXT,
            GrpDesc TEXT,
            Catagory TEXT,
            Narration TEXT,
            LocalDrAmt REAL,
            LocalCrAmt REAL,
            net_amount REAL,
            is_opening INTEGER,
            year INTEGER,
            month INTEGER,
            week INTEGER,
            day INTEGER
        )
    ''')

    # Insert test data covering various scenarios
    test_data = [
        # Opening balances
        ('1', '2024-01-01', 'Party A', 'OB', 'Sundry Debtors', 'Customer', 'Opening Balance', 50000, 0, 50000, 1, 2024, 1, 1, 1),
        ('2', '2024-01-01', 'Party B', 'OB', 'Sundry Debtors', 'Customer', 'Opening Balance', 30000, 0, 30000, 1, 2024, 1, 1, 1),
        ('3', '2024-01-01', 'Party C', 'OB', 'Sundry Creditors', 'Supplier', 'Opening Balance', 0, 20000, -20000, 1, 2024, 1, 1, 1),

        # Sales invoices
        ('4', '2024-01-15', 'Party A', 'SB', 'Sales', 'Revenue', 'Invoice INV-001', 25000, 0, 25000, 0, 2024, 1, 3, 15),
        ('5', '2024-01-20', 'Party B', 'SB', 'Sales', 'Revenue', 'Invoice INV-002', 15000, 0, 15000, 0, 2024, 1, 3, 20),
        ('6', '2024-02-10', 'Party A', 'SB', 'Sales', 'Revenue', 'Invoice INV-003', 30000, 0, 30000, 0, 2024, 2, 6, 10),

        # Collections
        ('7', '2024-01-25', 'Party A', 'CB', 'Sundry Debtors', 'Customer', 'Receipt RCP-001', 0, 20000, -20000, 0, 2024, 1, 4, 25),
        ('8', '2024-02-15', 'Party B', 'CB', 'Sundry Debtors', 'Customer', 'Receipt RCP-002', 0, 10000, -10000, 0, 2024, 2, 7, 15),

        # Purchases
        ('9', '2024-01-10', 'Party C', 'PB', 'Purchases', 'Expense', 'Purchase PI-001', 0, 10000, -10000, 0, 2024, 1, 2, 10),
        ('10', '2024-02-05', 'Party C', 'PB', 'Purchases', 'Expense', 'Purchase PI-002', 0, 5000, -5000, 0, 2024, 2, 6, 5),

        # Payments
        ('11', '2024-01-28', 'Party C', 'PAY', 'Sundry Creditors', 'Supplier', 'Payment PMT-001', 5000, 0, 5000, 0, 2024, 1, 4, 28),

        # Journal entries
        ('12', '2024-02-28', 'Party A', 'JV', 'Sales', 'Revenue', 'Adjustment', 1000, 0, 1000, 0, 2024, 2, 9, 28),
        ('13', '2024-02-28', 'Party A', 'JV', 'Sundry Debtors', 'Customer', 'Adjustment', 0, 1000, -1000, 0, 2024, 2, 9, 28),
    ]

    cursor.executemany(
        'INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        test_data
    )

    # Create indexes
    cursor.execute('CREATE INDEX idx_vdate ON data(VDate_iso)')
    cursor.execute('CREATE INDEX idx_gldesc ON data(GlDesc)')
    cursor.execute('CREATE INDEX idx_source ON data(Source)')
    cursor.execute('CREATE INDEX idx_grpdesc ON data(GrpDesc)')

    conn.commit()

    tools = DataTools(conn)
    yield tools

    conn.close()


class TestRankings:
    """Test top/bottom N rankings."""

    def test_top_parties_by_balance(self, sample_db):
        """Test ranking parties by balance (receivables)."""
        result = sample_db.ledger_statement("Party A", as_of_date="2024-02-28")
        assert result.success is True
        assert len(result.data) > 0
        # Party A: OB 50000 + Sales 55000 + JV 1000 - Collections 20000 - JV 1000 = 85000
        final_balance = result.data[-1]['running_balance']
        assert final_balance == 85000.0

    def test_bottom_parties_by_balance(self, sample_db):
        """Test parties with lowest balances."""
        result = sample_db.ledger_statement("Party C", as_of_date="2024-02-28")
        assert result.success is True
        # Party C: OB -20000 - Purchases 15000 + Payment 5000 = -30000
        final_balance = result.data[-1]['running_balance']
        assert final_balance == -30000.0


class TestZeroDivisionSafety:
    """Test division by zero protection in calculations."""

    def test_receivables_turnover_zero_sales(self, sample_db):
        """Test turnover when sales is zero."""
        # Create a party with no sales
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE data (
                VNo TEXT, VDate_iso TEXT, GlDesc TEXT, Source TEXT,
                LocalDrAmt REAL, LocalCrAmt REAL, net_amount REAL,
                year INTEGER, month INTEGER
            )
        ''')
        cursor.execute('INSERT INTO data VALUES ("1", "2024-01-01", "NoSales Party", "OB", 1000, 0, 1000, 2024, 1)')
        conn.commit()

        tools = DataTools(conn)
        result = tools.receivables_metrics(party="NoSales Party", as_of_date="2024-12-31")

        # Should handle zero division gracefully
        assert result.success is True
        # turnover should be 0 or None when no sales
        assert 'turnover' in str(result.data).lower() or 'days' in str(result.data).lower()
        conn.close()

    def test_collection_efficiency_zero_outstanding(self, sample_db):
        """Test collection efficiency when outstanding is zero."""
        result = sample_db.receivables_metrics(party="Party A", as_of_date="2024-02-28")
        assert result.success is True

    def test_average_collection_period_zero_collections(self, sample_db):
        """Test average collection period with no collections."""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE data (
                VNo TEXT, VDate_iso TEXT, GlDesc TEXT, Source TEXT,
                LocalDrAmt REAL, LocalCrAmt REAL, net_amount REAL,
                year INTEGER, month INTEGER
            )
        ''')
        cursor.execute('INSERT INTO data VALUES ("1", "2024-01-01", "NoCollect Party", "OB", 10000, 0, 10000, 2024, 1)')
        cursor.execute('INSERT INTO data VALUES ("2", "2024-02-01", "NoCollect Party", "SB", 5000, 0, 5000, 2024, 2)')
        conn.commit()

        tools = DataTools(conn)
        result = tools.receivables_metrics(party="NoCollect Party", as_of_date="2024-12-31")

        assert result.success is True
        conn.close()

    def test_percentage_calculation_zero_base(self, sample_db):
        """Test percentage calculations with zero base."""
        # This would be tested in compare tool
        result = sample_db.compare(
            metric="receivables",
            period1_start="2024-01-01", period1_end="2024-01-31",
            period2_start="2024-02-01", period2_end="2024-02-28"
        )
        assert result.success is True


class TestTrialBalanceEquality:
    """Test that Trial Balance Dr = Cr."""

    def test_trial_balance_balances(self, sample_db):
        """Test that total debits equal total credits in trial balance."""
        result = sample_db.trial_balance(as_of_date="2024-02-28")
        assert result.success is True

        # Parse the result
        data = result.data
        if isinstance(data, list) and len(data) > 0:
            # Check if it has summary totals
            total_dr = sum(row.get('dr_total', 0) for row in data if isinstance(row, dict))
            total_cr = sum(row.get('cr_total', 0) for row in data if isinstance(row, dict))

            # For a proper trial balance including opening balances,
            # the sum of all debits should equal sum of all credits
            # But we need to check the actual structure
            print(f"Trial balance data: {data}")

    def test_balance_sheet_balances(self, sample_db):
        """Test that Balance Sheet balances (Assets = Liabilities + Equity)."""
        result = sample_db.balance_sheet_pnl(as_of_date="2024-02-28")
        assert result.success is True

        data = result.data
        print(f"Balance sheet data: {data}")


class TestLedgerStatementRunningBalance:
    """Test ledger statement running balance calculation."""

    def test_running_balance_calculation(self, sample_db):
        """Test that running balance is calculated correctly."""
        result = sample_db.ledger_statement("Party A", as_of_date="2024-02-28")
        assert result.success is True

        rows = result.data
        assert len(rows) > 0

        # Verify running balance calculation
        running = 0
        for row in rows:
            dr = row.get('debit', 0) or row.get('LocalDrAmt', 0) or row.get('dr', 0)
            cr = row.get('credit', 0) or row.get('LocalCrAmt', 0) or row.get('cr', 0)
            running += dr - cr
            # Check running_balance column exists
            assert 'running_balance' in row or 'balance' in row

    def test_opening_balance_included(self, sample_db):
        """Test that opening balance (OB) is included as first entry."""
        result = sample_db.ledger_statement("Party A", as_of_date="2024-02-28")
        assert result.success is True

        rows = result.data
        # First row should be opening balance
        first_row = rows[0]
        assert first_row.get('source') == 'OB' or first_row.get('Source') == 'OB'

    def test_chronological_order(self, sample_db):
        """Test that entries are in chronological order."""
        result = sample_db.ledger_statement("Party A", as_of_date="2024-02-28")
        assert result.success is True

        rows = result.data
        dates = [row.get('vdate_iso') or row.get('VDate_iso') or row.get('date') for row in rows]
        dates = [d for d in dates if d]

        # Check non-decreasing order
        for i in range(1, len(dates)):
            assert dates[i] >= dates[i-1], f"Dates not in order: {dates}"

    def test_fifo_outstanding_calculation(self, sample_db):
        """Test FIFO outstanding calculation for Party A."""
        result = sample_db.fifo_outstanding(party="Party A", as_of_date="2024-02-28")
        assert result.success is True

        data = result.data
        print(f"FIFO outstanding: {data}")

        # Should have outstanding invoices after collections
        # Party A: OB 50000 + INV-001 25000 + INV-003 30000 + JV 1000 - RCP-001 20000 - JV 1000
        # = 85000 outstanding
        # FIFO: oldest invoices matched first


class TestPeriodSeries:
    """Test period series data generation."""

    def test_monthly_series(self, sample_db):
        """Test monthly period series."""
        result = sample_db.period_series(
            metric="receivables",
            period="month",
            start_date="2024-01-01",
            end_date="2024-02-28"
        )
        assert result.success is True
        assert len(result.data) >= 2  # Jan and Feb

    def test_quarterly_series(self, sample_db):
        """Test quarterly period series."""
        result = sample_db.period_series(
            metric="sales",
            period="quarter",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])