"""Aggregate pre-computed provision results into grouped city profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:  # geopandas is optional; we only need to read GeoDataFrame inputs.
    import geopandas as gpd  # type: ignore
except ImportError:  # pragma: no cover
    gpd = None  # type: ignore

# Public mapping of services by thematic group.
SERVICE_GROUPS: Dict[str, List[str]] = {
    "Образование": [
        "Детский сад",
        "Школа",
        "Дом детского творчества",
        "Среднее специальное учебное заведение",
        "Высшее учебное заведение",
    ],
    "Здравоохранение": [
        "Детская поликлиника",
        "Больница",
        "Роддом",
        "Станция скорой медицинской помощи",
        "Ветеринарная клиника",
    ],
    "Спорт": [
        "Спортивная площадка",
        "Бассейн",
        "Скейт-парк",
        "Детская площадка",
    ],
    "Социальная помощь": [
        "Детские дома-интернаты",
        "Комплексный центр социального обслуживания населения",
    ],
    "Услуги": [
        "Отделение банка",
        "Многофункциональные центры предоставления государственных и муниципальных услуг",
        "Банкомат",
        "Пункт выдачи",
        "Спортивный магазин",
        "Книжный магазин",
        "Хозяйственные товары",
        "Автозаправка",
        "Выход метро",
        "Супермаркет",
        "Продукты (магазин у дома)",
    ],
    "Культура и отдых": [
        "Театр",
        "Концертный зал",
        "Памятник",
        "Религиозный объект",
        "Библиотека",
        "Пляж",
        "Ботанический сад",
        "Сад",
        "Луг",
    ],
    "Безопасность": [
        "Пожарная станция",
        "Адвокат",
        "Электрическая подстанция",
        "Котельная",
        "Атомная электростанция",
        "Гидро-электростанция",
        "Тепловая электростанция",
        "Ветрогенератор",
        "ООПТ",
        "Промышленная зона",
        "Выпуски сточных вод в водоем",
    ],
    "Туризм": [
        "Гостиница",
        "Хостел",
        "Кафе",
        "Бар/Паб",
    ],
}

CityInfoSource = Union[str, Path, pd.DataFrame]
ServiceInput = Union[str, Path, pd.DataFrame]

REQUIRED_SERVICE_COLUMNS: Tuple[str, ...] = (
    "demand",
    "demand_within",
    "demand_without",
    "capacity",
    "capacity_left",
)


def _as_path(value: Union[str, Path]) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq", ".pqt"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".geojson":
        return gpd.read_file(path)
    raise ValueError(f"Unsupported file format: {path}")


def _ensure_numeric(series: pd.Series, *, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _drop_geometry(df: pd.DataFrame) -> pd.DataFrame:
    if gpd is not None and isinstance(df, gpd.GeoDataFrame):
        return pd.DataFrame(df.drop(columns=["geometry"], errors="ignore"))
    return df.drop(columns=["geometry"], errors="ignore")


@dataclass
class MigrationFlowModel:
    """Combine per-service results into grouped city provision profiles."""

    city_info: Optional[pd.DataFrame] = field(default=None, init=False)
    service_results: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    group_aggregates: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    service_aggregates: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    city_json: Dict[int, Dict[str, object]] = field(default_factory=dict, init=False)

    _service_results_lower: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False, repr=False)
    _external_supply_total: Optional[pd.Series] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def load_city_info(self, source: CityInfoSource) -> None:
        """Load city metadata (id, name, anchor flag, optional population)."""
        if isinstance(source, pd.DataFrame):
            data = source.copy()
        else:
            data = _read_table(_as_path(source))

        if data.empty:
            raise ValueError("City info table is empty.")

        data = _drop_geometry(data)
        data.columns = [str(col).strip() for col in data.columns]
        lower_map: Dict[str, str] = {col.lower(): col for col in data.columns}

        id_column = next(
            (lower_map[key] for key in ("city_id", "id", "town_id") if key in lower_map),
            None,
        )

        if id_column is not None:
            index = pd.to_numeric(data[id_column], errors="coerce")
        else:
            # fall back to index if column is missing
            index = pd.to_numeric(data.index, errors="coerce")

        if index.isna().any():
            raise ValueError("City identifiers must be numeric.")

        data.index = pd.Index(index.astype(int), name="city_id")

        name_column = next(
            (lower_map[key] for key in ("city_name", "name", "town_name") if key in lower_map),
            None,
        )
        if name_column is None:
            raise KeyError("City info must contain a column with city names (city_name, name, or town_name).")

        anchor_column = next(
            (
                lower_map[key]
                for key in ("is_anchor", "is_anchor_settlement", "anchor", "oporny", "is_city")
                if key in lower_map
            ),
            None,
        )
        if anchor_column is None:
            raise KeyError(
                "City info must contain a column with anchor flags "
                "(is_anchor, is_anchor_settlement, anchor, oporny, or is_city)."
            )

        population_column = next(
            (lower_map[key] for key in ("population", "pop", "inhabitants") if key in lower_map),
            None,
        )

        prepared = pd.DataFrame(index=data.index)
        prepared["city_name"] = data[name_column].astype(str)
        prepared["is_anchor"] = data[anchor_column].astype(bool)
        if population_column is not None:
            prepared["population"] = _ensure_numeric(data[population_column])
        else:
            prepared["population"] = 0.0

        self.city_info = prepared
        self.service_results.clear()
        self._service_results_lower.clear()
        self.group_aggregates.clear()
        self.service_aggregates.clear()
        self.city_json.clear()
        self._external_supply_total = None

    def load_service_results(
        self,
        services: Union[Mapping[str, ServiceInput], Sequence[ServiceInput]],
    ) -> None:
        """Load already calculated provision outputs for individual services."""
        if self.city_info is None:
            raise RuntimeError("Load city info before service results.")

        if isinstance(services, Mapping):
            items: Iterable[Tuple[str, ServiceInput]] = services.items()
        else:
            items = ((_infer_service_name(item), item) for item in services)

        loaded = 0
        for service_name, payload in items:
            if not service_name:
                raise ValueError("Failed to determine service name. Pass a mapping {service_name: path}.")
            frame = self._load_service_frame(payload)
            prepared = self._prepare_service_frame(frame)
            canonical_name = service_name.strip()
            self.service_results[canonical_name] = prepared
            self._service_results_lower[canonical_name.lower()] = prepared
            loaded += 1

        if loaded == 0:
            raise ValueError("No service results were loaded.")

        self.group_aggregates.clear()
        self.service_aggregates.clear()
        self.city_json.clear()
        self._external_supply_total = None

    # ------------------------------------------------------------------ #
    # Aggregation & export
    # ------------------------------------------------------------------ #

    def build_profiles(self) -> Dict[int, Dict[str, object]]:
        """Group services, compute metrics, and build JSON profiles."""
        if self.city_info is None:
            raise RuntimeError("City info is not loaded.")
        if not self.service_results:
            raise RuntimeError("Service results are not loaded.")

        external_supply = pd.Series(0.0, index=self.city_info.index, dtype=float)
        aggregates: Dict[str, pd.DataFrame] = {}
        has_matching_service = False

        for group_name, service_names in SERVICE_GROUPS.items():
            if any(
                service_name in self.service_results
                or service_name.lower() in self._service_results_lower
                for service_name in service_names
            ):
                has_matching_service = True
            aggregates[group_name] = self._aggregate_group(group_name, service_names, external_supply)

        if not has_matching_service:
            raise RuntimeError("No services match the configured service groups.")

        self.group_aggregates = aggregates
        self.service_aggregates = self._build_service_metrics()
        self._external_supply_total = external_supply
        self.city_json = self._assemble_city_json()
        return self.city_json

    def save_city_json(self, path: Union[str, Path], *, by: str = "id") -> None:
        """Save the assembled profiles to disk."""
        if not self.city_json:
            self.build_profiles()

        if by == "name":
            payload: MutableMapping[str, Dict[str, object]] = {
                profile["Название"]: profile for profile in self.city_json.values()
            }
        elif by == "id":
            payload = {str(city_id): profile for city_id, profile in self.city_json.items()}
        else:
            raise ValueError("Parameter 'by' must be either 'id' or 'name'.")

        target = _as_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_service_frame(self, payload: ServiceInput) -> pd.DataFrame:
        if isinstance(payload, pd.DataFrame):
            frame = payload.copy()
        else:
            frame = _read_table(_as_path(payload))
        if frame.empty:
            raise ValueError("Service result table is empty.")
        return _drop_geometry(frame)

    def _prepare_service_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.city_info is None:
            raise RuntimeError("City info must be loaded before services.")

        frame.columns = [str(col).strip() for col in frame.columns]
        lower_map = {col.lower(): col for col in frame.columns}

        if "city_id" in lower_map:
            frame = frame.set_index(lower_map["city_id"])
        elif "id" in lower_map:
            frame = frame.set_index(lower_map["id"])
        elif "town_id" in lower_map:
            frame = frame.set_index(lower_map["town_id"])
        elif frame.index.name and str(frame.index.name).lower() in {"city_id", "id", "town_id"}:
            pass
        else:
            frame.index.name = "city_id"

        frame.index = pd.Index(pd.to_numeric(frame.index, errors="coerce"), name="city_id")
        if frame.index.hasnans:
            raise ValueError("Service result index must be numeric (city ids).")

        numeric: Dict[str, pd.Series] = {}
        for column in REQUIRED_SERVICE_COLUMNS + ("population",):
            if column in frame.columns:
                numeric[column] = _ensure_numeric(frame[column])

        for column in REQUIRED_SERVICE_COLUMNS:
            if column not in numeric:
                raise KeyError(f"Service result must contain column '{column}'.")

        prepared = pd.DataFrame(numeric, index=frame.index)
        prepared = prepared.reindex(self.city_info.index).fillna(0.0)

        if "population" in prepared.columns:
            self._update_population(prepared["population"])

        return prepared

    def _update_population(self, population: pd.Series) -> None:
        if self.city_info is None:
            return
        aligned = population.reindex(self.city_info.index).fillna(0.0)
        mask = (self.city_info["population"] <= 0) & (aligned > 0)
        if mask.any():
            self.city_info.loc[mask, "population"] = aligned[mask]

    def _get_service_frame(self, service_name: str) -> Optional[pd.DataFrame]:
        frame = self.service_results.get(service_name)
        if frame is not None:
            return frame
        return self._service_results_lower.get(service_name.lower())

    def _build_service_metrics(self) -> Dict[str, pd.DataFrame]:
        if self.city_info is None:
            raise RuntimeError("City info is not loaded.")

        city_ids = self.city_info.index
        metrics: Dict[str, pd.DataFrame] = {}

        for service_names in SERVICE_GROUPS.values():
            for service_name in service_names:
                frame = self._get_service_frame(service_name)
                if frame is None:
                    demand = pd.Series(0.0, index=city_ids, dtype=float)
                    served = pd.Series(0.0, index=city_ids, dtype=float)
                    external = pd.Series(0.0, index=city_ids, dtype=float)
                else:
                    aligned = frame.reindex(city_ids).fillna(0.0)
                    demand = aligned["demand"]
                    served = aligned["demand_within"]
                    external = aligned["demand_without"]

                mask = demand > 0
                provision_pct = pd.Series(0.0, index=city_ids, dtype=float)
                provision_pct.loc[mask] = (
                    (served.loc[mask] / demand.loc[mask]).clip(0.0, 1.0) * 100.0
                )

                external_pct = pd.Series(0.0, index=city_ids, dtype=float)
                external_pct.loc[mask] = (
                    (external.loc[mask] / demand.loc[mask]).clip(lower=0.0) * 100.0
                )

                metrics[service_name] = pd.DataFrame(
                    {
                        "provision_pct": provision_pct,
                        "served_population": served,
                        "external_demand": external,
                        "external_pct": external_pct,
                    },
                    index=city_ids,
                )

        return metrics

    def _aggregate_group(
        self,
        group_name: str,
        service_names: Sequence[str],
        external_supply_acc: pd.Series,
    ) -> pd.DataFrame:
        if self.city_info is None:
            raise RuntimeError("City info is not loaded.")

        city_ids = self.city_info.index
        demand = pd.Series(0.0, index=city_ids, dtype=float)
        served = pd.Series(0.0, index=city_ids, dtype=float)
        external_demand = pd.Series(0.0, index=city_ids, dtype=float)
        capacity_used = pd.Series(0.0, index=city_ids, dtype=float)

        for service_name in service_names:
            frame = self._get_service_frame(service_name)
            if frame is None:
                continue

            aligned = frame.reindex(city_ids).fillna(0.0)
            demand += aligned["demand"]
            served += aligned["demand_within"]
            external_demand += aligned["demand_without"]

            used_capacity = aligned["capacity"] - aligned["capacity_left"]
            capacity_used += used_capacity

            supplied_to_others = (used_capacity - aligned["demand_within"]).clip(lower=0.0)
            external_supply_acc += supplied_to_others

        result = pd.DataFrame(
            {
                "city_id": city_ids,
                "group_name": group_name,
                "demand": demand,
                "served": served,
                "external_demand": external_demand,
            }
        ).set_index("city_id")

        mask = demand > 0

        result["provision_pct"] = 0.0
        result.loc[mask, "provision_pct"] = (
            (served.loc[mask] / demand.loc[mask]).clip(0.0, 1.0) * 100.0
        )

        result["served_population"] = served
        result["external_pct"] = 0.0
        result.loc[mask, "external_pct"] = (
            (external_demand.loc[mask] / demand.loc[mask]).clip(lower=0.0) * 100.0
        )

        result["capacity_used"] = capacity_used
        return result

    def _assemble_city_json(self) -> Dict[int, Dict[str, object]]:
        if self.city_info is None:
            raise RuntimeError("City info is not loaded.")
        if not self.group_aggregates:
            raise RuntimeError("Group aggregates are not available.")
        if not self.service_aggregates:
            raise RuntimeError("Service aggregates are not available.")

        profiles: Dict[int, Dict[str, object]] = {}
        if self._external_supply_total is None:
            external_supply = pd.Series(0.0, index=self.city_info.index, dtype=float)
        else:
            external_supply = self._external_supply_total.reindex(self.city_info.index).fillna(0.0)

        for city_id, city_row in self.city_info.iterrows():
            group_provision: Dict[str, Dict[str, float | int]] = {}
            group_mobility: Dict[str, Dict[str, float]] = {}
            service_provision: Dict[str, Dict[str, float | int]] = {}
            service_mobility: Dict[str, Dict[str, float]] = {}

            top_group = None
            top_value = 0.0
            has_served_group = False

            for group_name, service_names in SERVICE_GROUPS.items():
                group_df = self.group_aggregates.get(group_name)
                if group_df is not None and city_id in group_df.index:
                    metrics = group_df.loc[city_id]
                    provision_pct = float(metrics.get("provision_pct", 0.0))
                    served_population = float(metrics.get("served_population", 0.0))
                    external_demand = float(metrics.get("external_demand", 0.0))
                    external_pct = float(metrics.get("external_pct", 0.0))
                else:
                    provision_pct = 0.0
                    served_population = 0.0
                    external_demand = 0.0
                    external_pct = 0.0

                group_provision[group_name] = {
                    "Обеспеченность, %": round(provision_pct, 2),
                    "Обслуженное население": int(round(served_population)),
                }
                group_mobility[group_name] = {
                    "Внешний спрос": round(external_demand, 2),
                    "Доля внешнего спроса, %": round(external_pct, 2),
                }

                if served_population > 0:
                    has_served_group = True
                    if provision_pct > top_value or top_group is None:
                        top_value = provision_pct
                        top_group = group_name

            seen_services: set[str] = set()
            for service_names in SERVICE_GROUPS.values():
                for service_name in service_names:
                    if service_name in seen_services:
                        continue
                    seen_services.add(service_name)

                    metrics_df = self.service_aggregates.get(service_name)
                    if metrics_df is not None and city_id in metrics_df.index:
                        metrics = metrics_df.loc[city_id]
                        provision_pct = float(metrics.get("provision_pct", 0.0))
                        served_population = float(metrics.get("served_population", 0.0))
                        external_demand = float(metrics.get("external_demand", 0.0))
                        external_pct = float(metrics.get("external_pct", 0.0))
                    else:
                        provision_pct = 0.0
                        served_population = 0.0
                        external_demand = 0.0
                        external_pct = 0.0

                    service_provision[service_name] = {
                        "Обеспеченность, %": round(provision_pct, 2),
                        "Обслуженное население": int(round(served_population)),
                    }
                    service_mobility[service_name] = {
                        "Внешний спрос": round(external_demand, 2),
                        "Доля внешнего спроса, %": round(external_pct, 2),
                    }

            best_provision = round(top_value, 2) if has_served_group else None
            best_group = top_group if has_served_group else None

            population = int(round(float(city_row.get("population", 0.0) or 0.0)))
            is_anchor = bool(city_row.get("is_anchor", False))
            potential_anchor = bool(not is_anchor and external_supply.get(city_id, 0.0) > 0.0)

            profiles[int(city_id)] = {
                "Название": str(city_row.get("city_name", city_id)),
                "Опорный город": is_anchor,
                "Потенциал опорного": potential_anchor,
                "Население": population,
                "Лучшая обеспеченность, %": best_provision,
                "Сервисы: обеспеченность": service_provision,
                "Сервисы: мобильность": service_mobility,
                "Группы: обеспеченность": group_provision,
                "Группы: мобильность": group_mobility,
                "Лучшая группа": best_group,
            }

        return profiles


def _infer_service_name(payload: ServiceInput) -> Optional[str]:
    if isinstance(payload, pd.DataFrame):
        df = payload
        if "service_name" in df.columns:
            values = df["service_name"].dropna().unique()
            if len(values) == 1:
                return str(values[0])
        return None
    return _as_path(payload).stem or None


__all__ = ["MigrationFlowModel", "SERVICE_GROUPS"]
