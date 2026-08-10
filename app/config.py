from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: str = "carbon-x-dev-key"
    database_url: str = "sqlite:///./carbon_x.db"
    grid_emission_factor: float = 0.82
    baseline_days: int = 30
    ml_model_path: str = "models/fraud_model.joblib"
    fraud_score_flag_threshold: float = 0.45
    fraud_score_reject_threshold: float = 0.75

    class Config:
        env_file = ".env"


settings = Settings()
