"""Load explicit city inputs. No private path defaults or pickle deserialization."""

from pathlib import Path
from shapely.geometry import shape
from simulation.io import read
from simulation.world import RebuiltMap, CityWorld, load_source_classes


class Inputs:
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.spec = read(self.manifest)
        if self.spec["city"] not in ("chicago", "dallas", "los_angeles"):
            raise ValueError("Unsupported city")
        self.paths = {
            k: (self.manifest.parent / v).resolve()
            for k, v in self.spec["files"].items()
        }
        required = {
            "environments",
            "pois",
            "geometries",
            "residents",
            "actors",
            "police",
            "config",
        }
        if not required <= self.paths.keys():
            raise ValueError(
                "Missing input files: " + str(required - self.paths.keys())
            )
        data = {k: read(p) for k, p in self.paths.items()}
        self.map = RebuiltMap(
            data["environments"],
            data["pois"],
            {k: shape(v) for k, v in data["geometries"].items()},
        )
        self.source = (self.manifest.parent / self.spec["upstream"]).resolve()
        self.classes = load_source_classes(self.source)
        self.residents, self.actors, self.police = (
            data["residents"],
            data["actors"],
            data["police"],
        )
        self.config, self.patterns = data["config"], data.get("patterns")
        self.observed, self.domain = data.get("observed"), data.get("domain")
        self.excluded = set(data.get("excluded_actors", []))

    @property
    def city(self):
        return self.spec["city"]

    def world(self, seed, actor_ids=None):
        ids = set(actor_ids) if actor_ids is not None else None
        actors = [r for r in self.actors if ids is None or r["agent_id"] in ids]
        if ids is not None and len(ids) != len(actors):
            raise ValueError("Unknown or duplicate requested actor IDs")
        return CityWorld(
            self.map,
            self.classes,
            self.residents,
            actors,
            self.police,
            self.config,
            seed,
        )
