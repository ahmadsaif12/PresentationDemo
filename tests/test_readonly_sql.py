"""Tests for read-only SQLite enforcement."""
import pytest
import sqlite3
import tempfile
import os
from tools import SQLiteExecutor, ToolResult


class TestReadOnlySQL:
    """Test that SQLite authorizer blocks write operations."""

    @pytest.fixture
    def db_with_data(self):
        """Create a temporary database with test data."""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE data (
                VNo TEXT,
                VDate_iso TEXT,
                GlDesc TEXT,
                Source TEXT,
                LocalDrAmt REAL,
                LocalCrAmt REAL,
                net_amount REAL
            )
        ''')

        cursor.executemany(
            'INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?)',
            [
                ('1', '2024-01-01', 'Party A', 'OB', 1000, 0, 1000),
                ('2', '2024-01-02', 'Party B', 'SB', 500, 0, 500),
                ('3', '2024-01-03', 'Party A', 'CB', 0, 300, -300),
            ]
        )
        conn.commit()

        executor = SQLiteExecutor(conn)
        yield executor
        conn.close()

    def test_select_allowed(self, db_with_data):
        """Test that SELECT queries are allowed."""
        result = db_with_data.execute("SELECT * FROM data")
        assert result.success is True
        assert len(result.data) == 3

    def test_select_with_where(self, db_with_data):
        """Test SELECT with WHERE clause."""
        result = db_with_data.execute("SELECT * FROM data WHERE Source = 'OB'")
        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]['Source'] == 'OB'

    def test_insert_blocked(self, db_with_data):
        """Test that INSERT is blocked."""
        result = db_with_data.execute("INSERT INTO data VALUES ('4', '2024-01-04', 'Party C', 'SB', 200, 0, 200)")
        assert result.success is False
        assert "not allowed" in result.error.lower() or "read-only" in result.error.lower() or "unauthorized" in result.error.lower()

    def test_update_blocked(self, db_with_data):
        """Test that UPDATE is blocked."""
        result = db_with_data.execute("UPDATE data SET LocalDrAmt = 2000 WHERE VNo = '1'")
        assert result.success is False
        assert "not allowed" in result.error.lower() or "read-only" in result.error.lower() or "unauthorized" in result.error.lower()

    def test_delete_blocked(self, db_with_data):
        """Test that DELETE is blocked."""
        result = db_with_data.execute("DELETE FROM data WHERE VNo = '1'")
        assert result.success is False
        assert "not allowed" in result.error.lower() or "read-only" in result.error.lower() or "unauthorized" in result.error.lower()

    def test_create_table_blocked(self, db_with_data):
        """Test that CREATE TABLE is blocked."""
        result = db_with_data.execute("CREATE TABLE test (id INTEGER)")
        assert result.success is False
        assert "not allowed" in result.error.lower() or "read-only" in result.error.lower() or "unauthorized" in result.error.lower()

    def test_drop_table_blocked(self, db_with_data):
        """Test that DROP TABLE is blocked."""
        result = db_with_data.execute("DROP TABLE data")
        assert result.success is False
        assert "not allowed" in result.error.lower() or "read-only" in result.error.lower() or "unauthorized" in result.error.lower()

    def test_alter_table_blocked(self, db_with_data):
        """Test that ALTER TABLE is blocked."""
        result = db_with_data.execute("ALTER TABLE data ADD COLUMN test_col TEXT")
        assert result.success is False
        assert "not allowed" in result.error.lower() or "read-only" in result.error.lower() or "unauthorized" in result.error.lower()

    def test_pragma_allowed(self, db_with_data):
        """Test that PRAGMA queries are allowed (read-only)."""
        result = db_with_data.execute("PRAGMA table_info(data)")
        assert result.success is True
        assert len(result.data) > 0

    def test_with_cte_allowed(self, db_with_data):
        """Test that WITH (CTE) queries are allowed."""
        result = db_with_data.execute("""
            WITH cte AS (SELECT * FROM data WHERE Source = 'OB')
            SELECT * FROM cte
        """)
        assert result.success is True
        assert len(result.data) == 1

    def test_subquery_allowed(self, db_with_data):
        """Test that subqueries are allowed."""
        result = db_with_data.execute("""
            SELECT * FROM data
            WHERE VNo IN (SELECT VNo FROM data WHERE Source = 'OB')
        """)
        assert result.success is True
        assert len(result.data) == 1

    def test_join_allowed(self, db_with_data):
        """Test that JOINs are allowed."""
        result = db_with_data.execute("""
            SELECT d1.VNo, d1.GlDesc, d2.VNo as VNo2
            FROM data d1
            JOIN data d2 ON d1.GlDesc = d2.GlDesc
        """)
        assert result.success is True

    def test_union_allowed(self, db_with_data):
        """Test that UNION is allowed."""
        result = db_with_data.execute("""
            SELECT VNo, GlDesc FROM data WHERE Source = 'OB'
            UNION
            SELECT VNo, GlDesc FROM data WHERE Source = 'SB'
        """)
        assert result.success is True
        assert len(result.data) == 2

    def test_case_insensitive_block(self, db_with_data):
        """Test that case variations of blocked commands are blocked."""
        for cmd in ['insert', 'Insert', 'INSERT', 'update', 'Update', 'UPDATE', 'delete', 'Delete', 'DELETE']:
            result = db_with_data.execute(f"{cmd} INTO data VALUES ('test', '2024-01-01', 'Test', 'SB', 100, 0, 100)")
            assert result.success is False, f"Command '{cmd}' should be blocked"


class TestSQLiteExecutorAuthorizer:
    """Test the authorizer callback directly."""

    def test_authorizer_blocks_write(self):
        """Test that authorizer returns SQLITE_DENY for write operations."""
        conn = sqlite3.connect(':memory:')

        # Track authorizer calls
        auth_calls = []

        def authorizer(action, arg1, arg2, dbname, trigger):
            auth_calls.append((action, arg1, arg2, dbname, trigger))
            # Deny all write operations
            if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                         sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_DROP_TABLE,
                         sqlite3.SQLITE_ALTER_TABLE, sqlite3.SQLITE_CREATE_INDEX,
                         sqlite3.SQLITE_DROP_INDEX):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(authorizer)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE test (id INTEGER, val TEXT)')
        cursor.execute('INSERT INTO test VALUES (1, "a")')
        conn.commit()

        # Try insert - should fail
        with pytest.raises(sqlite3.DatabaseError):
            cursor.execute('INSERT INTO test VALUES (2, "b")')

        # Try select - should work
        cursor.execute('SELECT * FROM test')
        rows = cursor.fetchall()
        assert len(rows) == 1

        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])