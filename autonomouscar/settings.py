from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    CARLA_HOST: str = "localhost"
    CARLA_PORT: int = 2000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HLNA_API_",
        env_file_encoding="utf-8",
    )


settings = Settings()
