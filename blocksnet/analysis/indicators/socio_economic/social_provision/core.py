import pandas as pd
from tqdm import tqdm
from loguru import logger
from .indicator import SocialProvisionIndicator
from blocksnet.config import service_types_config, log_config
from blocksnet.analysis.provision import competitive_provision, provision_strong_total


def calculate_social_provision_indicators(
    blocks_df: pd.DataFrame, acc_mx: pd.DataFrame
) -> tuple[dict[SocialProvisionIndicator, float], list[SocialProvisionIndicator]]:

    result = {}
    missing = []

    disable_tqdm = log_config.disable_tqdm
    logger_level = log_config.logger_level
    log_config.set_disable_tqdm(True)
    log_config.set_logger_level("ERROR")

    for indicator in tqdm(list(SocialProvisionIndicator), disable=disable_tqdm):

        name = indicator.meta.name
        if not name in service_types_config:
            logger.warning(f"{name} not found in config. The indicator is skipped")
            missing.append(indicator)
            continue

        column = f"capacity_{indicator.meta.name}"
        if not column in blocks_df.columns:
            logger.warning(f"{column} is missing. The indicator is skipped")
            missing.append(indicator)
            continue

        _, demand, accessibility = service_types_config[name].values()
        df = blocks_df.rename(columns={column: "capacity"})
        prov_df, _ = competitive_provision(df, acc_mx, accessibility, demand)
        result[indicator] = provision_strong_total(prov_df)

    log_config.set_disable_tqdm(disable_tqdm)
    log_config.set_logger_level(logger_level)

    return result, missing
