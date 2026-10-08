"""Offline reconstruction of a city world using reviewed CrimeMind classes.

Only four full-file hash-pinned class definitions are compiled. Upstream imports,
module statements, trajectory files and pickle caches are never executed/read.
Geography, ACS missingness and OSM entity IDs come from the supplied public data;
this adapter does not reconstruct or invent the unavailable visit-flow cache.
"""

import ast
import builtins
from collections import Counter
import copy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pyproj import Transformer
from shapely import get_coordinates
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform

SOURCE_FILES = {
    "ResidentAgent": (
        "src/agents/resident.py",
        "49b4d58d820e9ab7a39b891c52964d6cf2e99cd9b60587c37941e48d7b87755d",
    ),
    "CriminalAgent": (
        "src/agents/criminal.py",
        "1bbad660cfa9cc1a623835e4d24f0863e3252f2733adaba693f0454fb09e1e9a",
    ),
    "PoliceAgent": (
        "src/agents/police.py",
        "99e264cbf5a7d1fb7f180b71423c1259f8ef1144162369837fb3a18a81e30aad",
    ),
    "EPRModel": (
        "src/models/EPR.py",
        "3b787ebb7ce981c6c2120047e0175f4f3140a372900ed66f4eadcf58223e1063",
    ),
}
_ACS_COLUMNS = {
    "Total population": "population",
    "average_income": "per_capita_income",
    "poverty_ratio": "poverty_ratio",
    "housing_value": "housing_value",
}


def _identifier(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (str, int))
        or not str(value).strip()
    ):
        raise ValueError(f"Invalid {label}")
    return str(value)


def _finite_numbers(value):
    if isinstance(value, Real) and not math.isfinite(value):
        raise ValueError("Inputs must contain finite numbers")
    if isinstance(value, dict):
        for item in value.values():
            _finite_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_numbers(item)


def _number(value, label, *, nullable=False, minimum=0, maximum=None):
    if value is None and nullable:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"Invalid {label}: expected finite number in range")


class RebuiltMap:
    """The small public-data Map interface consumed by upstream EPR and prompts.

    ``environments`` and ``pois`` retain independent copies of the audited rows,
    indexed by their original identifiers. ``aois`` exposes full WGS84 polygons,
    full EPSG:32616 polygons in metres, actual OSM IDs and source-named ACS columns.
    """

    def __init__(self, environments, pois, geometries):
        if not isinstance(environments, list) or not environments:
            raise ValueError("Need nonempty environment rows")
        self.environments = {}
        for original in environments:
            if not isinstance(original, dict):
                raise ValueError("Environment must be a row mapping")
            row = copy.deepcopy(original)
            _finite_numbers(row)
            key = _identifier(row.get("cbg_id"), "CBG ID")
            if key in self.environments:
                raise ValueError("Duplicate CBG ID")
            row["cbg_id"] = key
            if (
                not isinstance(row.get("description"), str)
                or not row["description"].strip()
            ):
                raise ValueError("Environment description is required")
            _number(row.get("safety_score"), "safety_score", maximum=1)
            for field in _ACS_COLUMNS.values():
                if field not in row:
                    raise ValueError(f"Missing ACS field: {field}")
                _number(
                    row[field],
                    field,
                    nullable=True,
                    maximum=1 if field == "poverty_ratio" else None,
                )
            _number(row.get("poi_count"), "poi_count")
            if row["poi_count"] % 1:
                raise ValueError("POI count must be an integer")
            self.environments[key] = row
        self.environments = dict(sorted(self.environments.items()))
        if not isinstance(geometries, dict):
            raise ValueError("Geometries must be keyed by CBG ID")
        normalized_geometries = {
            _identifier(key, "geometry CBG ID"): geom
            for key, geom in geometries.items()
        }
        if (
            len(normalized_geometries) != len(geometries)
            or normalized_geometries.keys() != self.environments.keys()
        ):
            raise ValueError("Geometry and environment CBG IDs must match exactly")
        if isinstance(pois, dict):
            poi_rows = []
            for key, row in pois.items():
                if not isinstance(row, dict) or str(key) != str(row.get("osm_id")):
                    raise ValueError("POI mapping keys must match OSM IDs")
                poi_rows.append(row)
        elif isinstance(pois, list):
            poi_rows = pois
        else:
            raise ValueError("POIs must be rows or an OSM-ID mapping")
        self.pois = {}
        by_cbg = {key: [] for key in self.environments}
        for original in poi_rows:
            if not isinstance(original, dict):
                raise ValueError("POI must be a row mapping")
            row = copy.deepcopy(original)
            _finite_numbers(row)
            key = _identifier(row.get("osm_id"), "OSM ID")
            cbg = _identifier(row.get("cbg_id"), "POI CBG ID")
            if key in self.pois or cbg not in by_cbg:
                raise ValueError("Duplicate OSM ID or unknown POI CBG ID")
            _number(row.get("lon"), "POI longitude", minimum=-180, maximum=180)
            _number(row.get("lat"), "POI latitude", minimum=-90, maximum=90)
            row.update(osm_id=key, cbg_id=cbg)
            self.pois[key] = row
            by_cbg[cbg].append(key)
        self.pois = dict(sorted(self.pois.items()))
        projector = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
        self.aois = {}
        for key, row in self.environments.items():
            geom = normalized_geometries[key]
            if (
                not isinstance(geom, (Polygon, MultiPolygon))
                or geom.is_empty
                or not geom.is_valid
                or any(not math.isfinite(v) for xy in get_coordinates(geom) for v in xy)
            ):
                raise ValueError(f"Invalid polygon geometry for {key}")
            projected = transform(projector.transform, geom)
            if (
                projected.is_empty
                or not projected.is_valid
                or any(
                    not math.isfinite(v)
                    for xy in get_coordinates(projected)
                    for v in xy
                )
            ):
                raise ValueError(f"Invalid projected geometry for {key}")
            if len(by_cbg[key]) != row["poi_count"]:
                raise ValueError(f"POI count differs from actual entities for {key}")
            self.aois[key] = {
                "shapely_lnglat": copy.deepcopy(geom),
                "shapely_xy": projected,
                "poi": sorted(by_cbg[key]),
                "data": pd.DataFrame(
                    [{column: row[field] for column, field in _ACS_COLUMNS.items()}],
                    dtype=object,
                ),
            }

    def get_aoi(self, aoi_id):
        """Return a known area, with the upstream ``None`` contract for misses."""
        return self.aois.get(str(aoi_id))


class _StableSet(set):
    """Retain source set membership/sampling while making iteration lexical.

    The reviewed stratified sampler converts sets (and their difference) to lists.
    Binding only its class namespace's ``set`` removes PYTHONHASHSEED dependence
    without disabling stratification, changing its counts, or editing source AST.
    """

    def __iter__(self):
        return iter(sorted(super().__iter__()))

    def __sub__(self, other):
        return type(self)(super().__sub__(other))


class _SourceClasses(dict):
    def __init__(self, definitions):
        self.definitions = definitions
        names = (
            "__build_class__",
            "super",
            "str",
            "int",
            "float",
            "bool",
            "len",
            "id",
            "hasattr",
            "max",
            "min",
            "sum",
            "range",
            "list",
            "dict",
            "tuple",
            "sorted",
            "staticmethod",
            "TypeError",
            "ValueError",
            "IndexError",
        )
        namespace = {
            "__name__": __name__ + "._reviewed",
            "__builtins__": {name: getattr(builtins, name) for name in names},
            "List": List,
            "Dict": Dict,
            "Any": Any,
            "Optional": Optional,
            "Tuple": Tuple,
            "Map": RebuiltMap,
            "math": math,
            "random": random.Random(0),
            "set": _StableSet,
        }
        for name, filename, source in definitions:
            module = ast.parse(source, filename=filename)
            exec(compile(module, filename, "exec"), namespace)
        super().__init__((name, namespace[name]) for name, _, _ in definitions)


def load_source_classes(upstream):
    """Verify every full file before compiling only its reviewed class AST."""
    upstream = Path(upstream)
    sources = {}
    for name, (filename, digest) in SOURCE_FILES.items():
        try:
            blob = (upstream / filename).read_bytes()
        except OSError as exc:
            raise ValueError(f"Unable to read reviewed source: {filename}") from exc
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError(f"Reviewed source hash mismatch: {filename}")
        sources[name] = blob.decode("utf-8")
    definitions = []
    for name, (filename, _) in SOURCE_FILES.items():
        tree = ast.parse(sources[name], filename=filename)
        candidates = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == name
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one reviewed class: {name}")
        definitions.append((name, filename, ast.unparse(candidates[0])))
    return _SourceClasses(tuple(definitions))


def _checked_epr_config(config, city_map):
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    _finite_numbers(config)
    settings = copy.deepcopy(config.get("epr_model", config))
    if not isinstance(settings, dict):
        raise ValueError("EPR config must be a mapping")
    for key in ("candidate_pool_size", "destination_band_size"):
        value = settings.get(key, 100 if key == "candidate_pool_size" else 500)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "rho",
        "gamma",
        "beta",
        "alpha",
        "target_density_lambda",
        "criminal_alpha_boost",
        "score_lambda",
        "activity_lambda",
        "return_score_lambda",
        "success_return_lambda",
        "utility_temperature",
    ):
        if key in settings:
            _number(settings[key], key, maximum=1 if key == "rho" else None)
    for key in (
        "distance_utility_weight",
        "unsafe_utility_weight",
        "activity_utility_weight",
    ):
        if key in settings:
            _number(settings[key], key, minimum=-math.inf)
    for key in ("stratified_candidates", "utility_selection"):
        if key in settings and not isinstance(settings[key], bool):
            raise ValueError(f"{key} must be boolean")
    if "opportunity_scores" not in settings:
        settings["opportunity_scores"] = {
            key: row["safety_score"] for key, row in city_map.environments.items()
        }
    scores = settings["opportunity_scores"]
    if not isinstance(scores, dict):
        raise ValueError("opportunity_scores must be a CBG mapping")
    normalized = {}
    for raw_key, value in scores.items():
        key = _identifier(raw_key, "opportunity score CBG ID")
        if key not in city_map.aois or key in normalized:
            raise ValueError("Unknown or duplicate opportunity score CBG ID")
        _number(value, "opportunity score", maximum=1)
        normalized[key] = value
    settings["opportunity_scores"] = normalized
    return settings


class CityWorld:
    """An independent world running the source agent and EPR class definitions.

    ``config`` accepts either the upstream full config or its ``epr_model`` mapping.
    ``move(0), move(1), ...`` follows upstream ordering: residents from their current
    location, refreshed resident density, HRIs from their residence, and uniform-CBG
    police movement. Upstream appends the *trajectory length* as each movement index;
    its police update does not append a trajectory. Those details are retained.

    Documented reproducibility adaptations: movement executes serially, and each
    (seed, role, source agent ID, step) gets a private random stream. Each world
    compiles its own classes/globals from the already-reviewed class definitions.
    The source stratified sampler uses a locally bound deterministically ordered set.
    Neither intervention decisions nor simulated crimes are supplied by this class.
    """

    def __init__(self, map, classes, residents, hris, police, config, seed):
        if not isinstance(map, RebuiltMap):
            raise ValueError("CityWorld needs a RebuiltMap")
        if not isinstance(classes, _SourceClasses):
            raise ValueError("Classes must come from load_source_classes")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Seed must be an integer")
        self.map = copy.deepcopy(map)
        self.config = copy.deepcopy(config)
        settings = _checked_epr_config(self.config, self.map)
        self.seed = seed
        self.current_step = -1
        self.classes = _SourceClasses(classes.definitions)
        self.epr_model = self.classes["EPRModel"](settings)
        self.residents, self.criminals, self.police = [], [], []
        self.source_rows = {}
        seen = set()
        for role, rows in (
            ("residents", residents),
            ("hris", hris),
            ("police", police),
        ):
            if not isinstance(rows, list):
                raise ValueError(f"{role} must be a list of source rows")
            self.source_rows[role] = copy.deepcopy(rows)
            for row in self.source_rows[role]:
                if not isinstance(row, dict):
                    raise ValueError(f"{role} must contain source row mappings")
                _finite_numbers(row)
                agent_id = _identifier(row.get("agent_id"), "source agent ID")
                residence = _identifier(row.get("residence"), "source residence")
                if agent_id in seen or residence not in self.map.aois:
                    raise ValueError("Duplicate source agent ID or unknown residence")
                seen.add(agent_id)
                if role == "police":
                    self.police.append(
                        self.classes["PoliceAgent"](
                            agent_id=agent_id,
                            police_station=residence,
                            historical_trajectory=[],
                        )
                    )
                    continue
                for field in ("gender", "race"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f"Missing source {field}")
                attributes = dict(
                    agent_id=agent_id,
                    gender=row["gender"],
                    race=row["race"],
                    residence=residence,
                    current_location=None,
                    historical_trajectory=None,
                    visited_locations=None,
                    total_trajectory=None,
                )
                if role == "residents":
                    income = row.get("income_level")
                    # Actual citizens.json uses category strings despite the
                    # upstream constructor's int annotation. Keep source values.
                    categories = {
                        "Low",
                        "Lower_Middle",
                        "Middle",
                        "Upper_Middle",
                        "High",
                    }
                    numeric_level = (
                        isinstance(income, int)
                        and not isinstance(income, bool)
                        and 1 <= income <= 5
                    )
                    category_level = isinstance(income, str) and income in categories
                    if not numeric_level and not category_level:
                        raise ValueError("Invalid source resident income_level")
                    self.residents.append(
                        self.classes["ResidentAgent"](income_level=income, **attributes)
                    )
                else:
                    self.criminals.append(
                        self.classes["CriminalAgent"](
                            criminal_record=[(-1, residence)], **attributes
                        )
                    )
        self._criminals_by_id = {agent.agent_id: agent for agent in self.criminals}

    def _stream(self, role, agent_id, step):
        label = json.dumps(
            [self.seed, role, agent_id, step], ensure_ascii=True, separators=(",", ":")
        )
        return random.Random(
            int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest(), "big")
        )

    def _check_step(self, step, *, next_step=False):
        expected = self.current_step + 1 if next_step else self.current_step
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or step != expected
        ):
            raise ValueError(
                "Step must be the next consecutive integer"
                if next_step
                else "Step must match the current moved step"
            )

    def move(self, step):
        """Advance every source agent exactly once at this zero-based step."""
        self._check_step(step, next_step=True)
        aois = list(self.map.aois)
        namespace = self.epr_model.generate_new_loc.__globals__
        for role, population in (
            ("resident", self.residents),
            ("criminal", self.criminals),
        ):
            if role == "criminal":
                self.epr_model.set_target_density(
                    Counter(a.current_location for a in self.residents)
                )
            for agent in population:
                namespace["random"] = self._stream(role, agent.agent_id, step)
                location = self.epr_model.generate_new_loc(
                    resident=agent,
                    aois=aois,
                    start_place=(
                        agent.current_location
                        if role == "resident"
                        else agent.residence
                    ),
                    map=self.map,
                    current_step=step,
                )
                if location not in self.map.aois:
                    raise ValueError("Source EPR returned an unknown location")
                agent.visited_locations[location] = (
                    agent.visited_locations.get(location, 0) + 1
                )
                agent.historical_trajectory.append(
                    (len(agent.historical_trajectory), location)
                )
                agent.current_location = location
        for agent in self.police:
            agent.current_location = self._stream(
                "police", agent.agent_id, step
            ).choice(aois)
        self.current_step = step

    def encounters(self, step):
        """Copy factual co-locations, including HRIs with zero resident targets."""
        self._check_step(step)
        residents_by_location = {}
        for resident in self.residents:
            residents_by_location.setdefault(resident.current_location, []).append(
                resident
            )
        police_counts = Counter(agent.current_location for agent in self.police)
        return [
            {
                "step": step,
                "criminal": copy.deepcopy(agent.get_attributes()),
                "targets": [
                    copy.deepcopy(target.get_attributes())
                    for target in residents_by_location.get(agent.current_location, [])
                ],
                "police_count": police_counts[agent.current_location],
                "current_location": agent.current_location,
            }
            for agent in self.criminals
        ]

    def record_crime(self, agent_id, step, location):
        """Append a simulated source record only at the HRI's actual location."""
        self._check_step(step)
        agent_id = _identifier(agent_id, "source HRI ID")
        location = _identifier(location, "crime location")
        agent = self._criminals_by_id.get(agent_id)
        if (
            agent is None
            or location not in self.map.aois
            or agent.current_location != location
        ):
            raise ValueError(
                "Unknown HRI ID or crime location differs from actual location"
            )
        agent.add_criminal_record(step, location)
