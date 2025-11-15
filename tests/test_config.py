from pmrisk.config import settings


def test_settings_basic_values():
    assert settings.horizon_n > 0
    assert settings.window_l > 0
    assert settings.data_raw_dir
    assert settings.data_processed_dir
    assert settings.models_production_dir
