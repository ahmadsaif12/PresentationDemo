"""
Settings management for the Chat with Company Data application.
Handles multiple saved configurations, persistence to settings.json, and environment variables.
"""

import json
import os
import re
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseModel):
    """API connection settings for a single company/database."""
    base_url: str = Field(default="", description="Base API URL")
    database: str = Field(default="", description="Database name")
    param_name: str = Field(default="dbname", description="Query parameter name for database")
    api_key: str = Field(default="", description="API key (optional)")
    api_key_header: str = Field(default="X-API-Key", description="Header name for API key")
    api_key_prefix: str = Field(default="", description="Prefix for API key (e.g., 'Bearer ')")
    timeout: int = Field(default=30, description="Request timeout in seconds")

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        if v and not v.startswith(('http://', 'https://')):
            return 'http://' + v
        return v


class OpenAISettings(BaseModel):
    """OpenAI API settings."""
    api_key: str = Field(default="", description="OpenAI API key")
    model: str = Field(default="gpt-4o-mini", description="Model to use")
    base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    strong_model: str = Field(default="gpt-4o", description="Stronger model for complex tasks")


class AppSettings(BaseModel):
    """Application-level settings."""
    week_start: str = Field(default="Monday", description="Week start day")
    aging_slabs: str = Field(default="0-30,31-60,61-90,91-180,181-365,365+", description="Aging slabs for receivables")
    turnover_days: int = Field(default=365, description="Days in year for turnover calculation")
    refresh_interval: int = Field(default=300, description="Auto-refresh interval in seconds")
    max_rows: int = Field(default=200, description="Maximum rows per query")
    max_tool_steps: int = Field(default=8, description="Maximum tool calling steps")


class CompanyConfig(BaseModel):
    """Complete configuration for a single company/database."""
    id: str = Field(default="", description="Unique identifier")
    name: str = Field(default="", description="Display name")
    api: ApiSettings = Field(default_factory=ApiSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    app: AppSettings = Field(default_factory=AppSettings)
    # Detected mappings (populated after analysis)
    source_mapping: dict = Field(default_factory=dict, description="Source code -> role mapping")
    group_mapping: dict = Field(default_factory=dict, description="Group -> kind mapping")
    # Data profile (populated after loading)
    data_profile: dict = Field(default_factory=dict, description="Column profiles and statistics")
    schema_info: str = Field(default="", description="Schema description for LLM")
    last_refreshed: str = Field(default="", description="Last data refresh timestamp")
    record_count: int = Field(default=0, description="Number of records loaded")
    date_range: dict = Field(default_factory=dict, description="Min/max dates in data")
    distinct_counts: dict = Field(default_factory=dict, description="Distinct counts for key columns")


class SettingsManager:
    """Manages multiple company configurations with persistence."""

    def __init__(self, settings_file: str = "settings.json"):
        self.settings_file = Path(settings_file)
        self.configs: dict[str, CompanyConfig] = {}
        self.current_config_id: str = ""
        load_dotenv()
        self.load()

    def load(self) -> None:
        """Load settings from file."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for config_data in data.get('configs', []):
                    config = CompanyConfig(**config_data)
                    self.configs[config.id] = config

                self.current_config_id = data.get('current_config_id', '')

                # Migrate from .env if no configs exist
                if not self.configs:
                    self._migrate_from_env()

            except Exception as e:
                print(f"Warning: Failed to load settings: {e}")
                self._migrate_from_env()
        else:
            self._migrate_from_env()

    def _migrate_from_env(self) -> None:
        """Migrate settings from .env file."""
        base_url = os.getenv('DATA_API_URL', '')
        api_key = os.getenv('DATA_API_KEY', '')
        api_key_header = os.getenv('DATA_API_KEY_HEADER', 'X-API-Key')
        api_key_prefix = os.getenv('DATA_API_KEY_PREFIX', '')
        openai_key = os.getenv('OPENAI_API_KEY', '')
        openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        openai_base = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        refresh_interval = int(os.getenv('REFRESH_INTERVAL', '300'))

        # Extract database from URL if present
        database = ""
        param_name = "dbname"
        if base_url:
            match = re.search(r'[?&](\w+)=([^&]+)', base_url)
            if match:
                param_name = match.group(1)
                database = match.group(2)
                # Remove the param from base_url
                base_url = re.sub(r'[?&]' + re.escape(param_name) + r'=[^&]*', '', base_url)
                base_url = base_url.rstrip('?&')

        config = CompanyConfig(
            id="default",
            name="Default Company",
            api=ApiSettings(
                base_url=base_url,
                database=database,
                param_name=param_name,
                api_key=api_key,
                api_key_header=api_key_header,
                api_key_prefix=api_key_prefix,
            ),
            openai=OpenAISettings(
                api_key=openai_key,
                model=openai_model,
                base_url=openai_base,
            ),
            app=AppSettings(refresh_interval=refresh_interval)
        )
        self.configs["default"] = config
        self.current_config_id = "default"
        self.save()

    def save(self) -> None:
        """Save settings to file."""
        data = {
            'configs': [config.model_dump() for config in self.configs.values()],
            'current_config_id': self.current_config_id
        }
        # Ensure parent directory exists
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.settings_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_current(self) -> CompanyConfig:
        """Get the currently selected configuration."""
        if self.current_config_id and self.current_config_id in self.configs:
            return self.configs[self.current_config_id]
        # Return first config or create default
        if self.configs:
            first_id = next(iter(self.configs))
            self.current_config_id = first_id
            return self.configs[first_id]
        # Create minimal default
        config = CompanyConfig(id="default", name="Default")
        self.configs["default"] = config
        self.current_config_id = "default"
        return config

    def set_current(self, config_id: str) -> bool:
        """Set the current configuration."""
        if config_id in self.configs:
            self.current_config_id = config_id
            self.save()
            return True
        return False

    def add_config(self, config: CompanyConfig) -> str:
        """Add a new configuration."""
        if not config.id:
            config.id = self._generate_id(config.name)
        # Ensure unique ID
        base_id = config.id
        counter = 1
        while config.id in self.configs:
            config.id = f"{base_id}_{counter}"
            counter += 1
        self.configs[config.id] = config
        self.save()
        return config.id

    def update_config(self, config_id: str, updates: dict) -> bool:
        """Update a configuration."""
        if config_id not in self.configs:
            return False
        config = self.configs[config_id]
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self.save()
        return True

    def delete_config(self, config_id: str) -> bool:
        """Delete a configuration."""
        if config_id in self.configs and len(self.configs) > 1:
            del self.configs[config_id]
            if self.current_config_id == config_id:
                self.current_config_id = next(iter(self.configs))
            self.save()
            return True
        return False

    def list_configs(self) -> list[dict]:
        """List all configurations for UI."""
        return [
            {
                'id': config.id,
                'name': config.name or config.id,
                'api_base_url': config.api.base_url,
                'database': config.api.database,
                'record_count': config.record_count,
                'last_refreshed': config.last_refreshed,
                'is_current': config.id == self.current_config_id
            }
            for config in self.configs.values()
        ]

    def _generate_id(self, name: str) -> str:
        """Generate a unique ID from name."""
        base = re.sub(r'[^a-zA-Z0-9]+', '_', name.lower()).strip('_')
        return base or "company"

    def build_final_url(self, config: Optional[CompanyConfig] = None) -> str:
        """Build the final API URL with database parameter."""
        if config is None:
            config = self.get_current()

        base = config.api.base_url.rstrip('/')
        db = config.api.database
        param = config.api.param_name.strip() or "dbname"

        if not base or not db:
            return base

        # A placeholder may appear in either the path or query string.
        if '{db}' in base:
            return base.replace('{db}', quote(db, safe=''))

        # Parse and re-encode the query rather than doing a string replacement.
        # This keeps an existing parameter exactly once and safely handles spaces
        # and other characters in a database name.
        parsed = urlsplit(base)
        query = parse_qsl(parsed.query, keep_blank_values=True)
        replaced = False
        final_query = []
        for key, value in query:
            if key == param:
                if not replaced:
                    final_query.append((key, db))
                    replaced = True
                continue
            final_query.append((key, value))
        if not replaced:
            final_query.append((param, db))
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(final_query), parsed.fragment))

    def mask_api_key(self, key: str) -> str:
        """Mask API key for display."""
        if not key:
            return ""
        if len(key) <= 8:
            return "•" * len(key)
        return key[:4] + "•" * (len(key) - 8) + key[-4:]


# Global settings manager instance
settings_manager = SettingsManager()
