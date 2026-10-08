"""Frozen exposure targeting and bounded policies over the source SAT world.

All step indices are zero based. Targeting reads only the half-open selection
window. Resident participation uses registered residence and a keyed identity
draw; P labels are used only for descriptive coverage. No policy selects crime,
changes source safety caches, or changes resident/actor mobility rules.
"""

from copy import deepcopy
import hashlib
import json
import math

from .geography import ENV_FIELDS

ARMS = ("control", "hotspot", "collective", "environment")
DEFAULTS = dict(
    selection_start=5,
    selection_end=20,
    activation_step=25,
    exposure_target=0.30,
    max_target_cbgs=100,
    safety_increment=0.20,
    participation_fraction=0.60,
    baseline_visibility=0.40,
    baseline_willingness=0.20,
    baseline_coordination=0.25,
    collective_willingness=0.60,
    collective_coordination=0.70,
    hotspot_officers=100,
    help_response_cap=25,
)


def _json(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _draw(seed, *parts):
    digest = hashlib.sha256(_json([seed, *parts]).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return seed


def _identifier(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (str, int))
        or not str(value).strip()
    ):
        raise ValueError(f"Invalid {label}")
    return str(value)


def _parameters(parameters):
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("parameters must be a mapping")
    unknown = set(parameters or {}) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown policy parameters: {sorted(unknown)}")
    result = dict(DEFAULTS, **(parameters or {}))
    integer_keys = (
        "selection_start",
        "selection_end",
        "activation_step",
        "max_target_cbgs",
        "hotspot_officers",
        "help_response_cap",
    )
    for key in integer_keys:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if (
        not result["selection_start"]
        < result["selection_end"]
        <= result["activation_step"]
    ):
        raise ValueError("selection window must end no later than activation_step")
    if result["max_target_cbgs"] < 1 or result["hotspot_officers"] < 1:
        raise ValueError("max_target_cbgs and hotspot_officers must be positive")
    for key in DEFAULTS.keys() - set(integer_keys):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{key} must be a finite fraction in [0, 1]")
    if result["exposure_target"] == 0:
        raise ValueError("exposure_target must be positive")
    return result


def _group(row):
    values = [str(row[key]) for key in ("p", "P", "p_group", "P_group") if key in row]
    if values and (not values[0].strip() or len(set(values)) != 1):
        raise ValueError("Missing or conflicting P group label")
    return values[0] if values else "unassigned"


def _resident_rows(residents):
    if not isinstance(residents, list):
        raise ValueError("residents must be source row dictionaries in a list")
    result = {}
    for row in residents:
        if not isinstance(row, dict):
            raise ValueError("residents must contain source row dictionaries")
        key = _identifier(row.get("agent_id"), "resident agent_id")
        residence = _identifier(row.get("residence"), "resident residence")
        if key in result:
            raise ValueError("duplicate resident agent_id")
        result[key] = residence
    return result


def _coverage(rows, targets):
    target_rows = [row for row in rows if str(row["location"]) in targets]
    opportunities = sum(row["opportunity"] for row in rows)
    target_opportunities = sum(row["opportunity"] for row in target_rows)
    return dict(
        actor_steps=len(rows),
        target_actor_steps=len(target_rows),
        actor_step_coverage=len(target_rows) / len(rows) if rows else None,
        opportunities=opportunities,
        target_opportunities=target_opportunities,
        opportunity_coverage=(
            target_opportunities / opportunities if opportunities else None
        ),
        crimes=sum(row["crime"] for row in rows),
        target_crimes=sum(row["crime"] for row in target_rows),
    )


def build_target_plan(records, residents, seed, parameters=None):
    """Freeze targets from actor-step rows and participants from source residents.

    ``records`` is a list of row mappings with step, actor (or actor_id),
    location, boolean opportunity/crime, and optional p/P/p_group/P_group.
    Out-of-window rows do not enter counts, ranking or source_sha256. The hash
    covers the full selected source rows sorted by (step, actor), including any
    receipt evidence supplied by the runner. Zero-opportunity windows fail.
    """
    seed, parameters = _seed(seed), _parameters(parameters)
    if not isinstance(records, list):
        raise ValueError("records must be a list of actor-step row mappings")
    resident_homes = _resident_rows(residents)
    selected, seen, counts, groups = [], set(), {}, {}
    for original in records:
        if not isinstance(original, dict):
            raise ValueError("record must be a row mapping")
        step = original.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("record step must be a nonnegative integer")
        if not parameters["selection_start"] <= step < parameters["selection_end"]:
            continue
        actor = _identifier(
            original.get("actor", original.get("actor_id")), "record actor"
        )
        location = _identifier(original.get("location"), "record location")
        if (step, actor) in seen:
            raise ValueError("duplicate actor-step in target selection window")
        seen.add((step, actor))
        if (
            type(original.get("opportunity")) is not bool
            or type(original.get("crime")) is not bool
        ):
            raise ValueError("opportunity and crime must be boolean")
        if original["crime"] and not original["opportunity"]:
            raise ValueError("crime requires an opportunity")
        group = _group(original)
        groups.setdefault(group, []).append(original)
        selected.append((step, actor, deepcopy(original)))
        count = counts.setdefault(
            location, dict(actor_steps=0, opportunities=0, crimes=0)
        )
        count["actor_steps"] += 1
        count["opportunities"] += int(original["opportunity"])
        count["crimes"] += int(original["crime"])
    selected.sort(key=lambda item: (item[0], item[1]))
    selected_rows = [item[2] for item in selected]
    total_opportunities = sum(count["opportunities"] for count in counts.values())
    if not total_opportunities:
        raise ValueError("Target selection window contains zero opportunities")
    ranked = sorted(
        (key for key in counts if counts[key]["opportunities"]),
        key=lambda key: (-counts[key]["crimes"], -counts[key]["opportunities"], key),
    )
    targets, covered = [], 0
    for key in ranked[: parameters["max_target_cbgs"]]:
        targets.append(key)
        covered += counts[key]["opportunities"]
        if covered >= parameters["exposure_target"] * total_opportunities:
            break
    target_set = set(targets)
    eligible = sorted(
        key for key, residence in resident_homes.items() if residence in target_set
    )
    participant_count = math.floor(parameters["participation_fraction"] * len(eligible))
    participants = sorted(
        sorted(
            eligible,
            key=lambda key: (_draw(seed, "collective_participation", key), key),
        )[:participant_count]
    )
    plan = dict(
        targets=targets,
        participant_ids=participants,
        eligible_resident_ids=eligible,
        selection_range=dict(
            start=parameters["selection_start"],
            end=parameters["selection_end"],
            semantics="zero_based_half_open",
        ),
        source_sha256=_sha(selected_rows),
        counts_by_cbg=dict(sorted(counts.items())),
        ranked_cbg_ids=ranked,
        coverage=dict(
            total=_coverage(selected_rows, target_set),
            by_p={
                key: _coverage(rows, target_set) for key, rows in sorted(groups.items())
            },
        ),
        parameters=parameters,
        seed=seed,
    )
    plan["plan_id"] = _sha(plan)
    return plan


class PolicyWorld:
    """Own one unmoved source SAT world; call move, observe, then finish_step.

    configure(plan) may be called during baseline and is idempotent for the exact
    same plan. Configuration is mandatory when entering the activation step.
    All exogenous draws omit the arm and P label. No global monkeypatch is used.
    """

    def __init__(self, ready, arm, seed, parameters=None):
        if arm not in ARMS:
            raise ValueError("Unknown policy arm")
        self.arm, self.seed, self.parameters = arm, _seed(seed), _parameters(parameters)
        self.world = ready["worlds"]["sat"]
        if self.world.current_step != -1:
            raise ValueError("PolicyWorld requires an unmoved source world")
        self._targets, self._participant_ids = (), ()
        self.logs, self.pending_help = [], []
        self._plan, self._plan_json, self._finished = None, None, True
        self._participant_set = set()
        self._residents = {str(a.agent_id): a for a in self.world.residents}
        self._actors = {str(a.agent_id): a for a in self.world.criminals}
        self._officers = {str(a.agent_id): a for a in self.world.police}
        self._centroids = {
            key: aoi["shapely_xy"].centroid for key, aoi in self.world.map.aois.items()
        }
        self._source_safety = {
            key: row["safety_score"]
            for key, row in sorted(self.world.map.environments.items())
        }
        self._hotspot_ids = sorted(
            self._officers, key=lambda key: (self._draw("patrol", key), key)
        )[: self.parameters["hotspot_officers"]]

    def _draw(self, *parts):
        return _draw(self.seed, *parts)

    @property
    def targets(self):
        return list(self._targets)

    @property
    def participant_ids(self):
        return list(self._participant_ids)

    def configure(self, plan):
        encoded = _json(plan)
        if self._plan_json is not None:
            if encoded != self._plan_json:
                raise ValueError("Target plan is frozen; cannot change configuration")
            return
        if not isinstance(plan, dict):
            raise ValueError("Target plan must be a mapping")
        if plan.get("seed") != self.seed or plan.get("parameters") != self.parameters:
            raise ValueError("Target plan seed or parameters do not match this world")
        if plan.get("plan_id") != _sha(
            {key: value for key, value in plan.items() if key != "plan_id"}
        ):
            raise ValueError("Target plan integrity mismatch")
        targets, participants = plan.get("targets"), plan.get("participant_ids")
        if (
            not isinstance(targets, list)
            or not targets
            or len(targets) != len(set(targets))
            or not set(targets) <= self.world.map.aois.keys()
            or len(targets) > self.parameters["max_target_cbgs"]
        ):
            raise ValueError("Target plan has invalid or unknown target locations")
        eligible = sorted(
            key
            for key, resident in self._residents.items()
            if resident.residence in targets
        )
        count = math.floor(self.parameters["participation_fraction"] * len(eligible))
        expected_participants = sorted(
            sorted(
                eligible,
                key=lambda key: (self._draw("collective_participation", key), key),
            )[:count]
        )
        if (
            plan.get("eligible_resident_ids") != eligible
            or participants != expected_participants
        ):
            raise ValueError(
                "Target plan participants differ from registered resident identities"
            )
        self._plan, self._plan_json = deepcopy(plan), encoded
        self._targets, self._participant_ids = tuple(targets), tuple(participants)
        self._participant_set = set(participants)

    def _active(self, location=None):
        return self.world.current_step >= self.parameters["activation_step"] and (
            location is None or location in self.targets
        )

    def _response_parameters(self, resident_id):
        collective = (
            self.arm == "collective"
            and self._active()
            and str(resident_id) in self._participant_set
        )
        prefix = "collective" if collective else "baseline"
        return (
            self.parameters[prefix + "_willingness"],
            self.parameters[prefix + "_coordination"],
        )

    def move(self, step):
        if not self._finished:
            raise ValueError("finish_step must complete before the next move")
        if step >= self.parameters["activation_step"]:
            if self._plan is None:
                raise ValueError("configure a frozen target plan before activation")
            if (
                self.arm == "hotspot"
                and len(self._officers) < self.parameters["hotspot_officers"]
            ):
                raise ValueError(
                    "Insufficient source police for the registered hotspot dose"
                )
        self.world.move(step)
        self._finished = False
        if self._active():
            if self.arm == "hotspot":
                for index, officer_id in enumerate(self._hotspot_ids):
                    officer = self._officers[officer_id]
                    destination = self.targets[index % len(self.targets)]
                    self.logs.append(
                        dict(
                            kind="hotspot_assignment",
                            step=step,
                            officer_id=officer_id,
                            source_location=officer.current_location,
                            destination=destination,
                        )
                    )
                    officer.current_location = destination
            if step == self.parameters["activation_step"]:
                if self.arm == "collective":
                    for kind in ("community_meeting", "project_start"):
                        self.logs.append(
                            dict(
                                kind=kind,
                                step=step,
                                targets=list(self.targets),
                                participant_ids=list(self.participant_ids),
                                willingness=self.parameters["collective_willingness"],
                                coordination=self.parameters["collective_coordination"],
                            )
                        )
                elif self.arm == "environment":
                    for location in self.targets:
                        self.logs.append(
                            dict(
                                kind="environment_activation",
                                step=step,
                                location=location,
                                original_safety_score=self._source_safety[location],
                                effective_safety_score=self.environment(location)[
                                    "safety_score"
                                ],
                            )
                        )
        self._respond_to_help(step)

    def _respond_to_help(self, step):
        due = [item for item in self.pending_help if item["respond_at_step"] <= step]
        self.pending_help = [
            item for item in self.pending_help if item["respond_at_step"] > step
        ]
        reserved = (
            set(self._hotspot_ids)
            if self.arm == "hotspot" and self._active()
            else set()
        )
        available, allocated = set(self._officers) - reserved, 0
        for request in sorted(due, key=lambda item: item["event_id"]):
            if allocated >= self.parameters["help_response_cap"] or not available:
                self.logs.append(
                    dict(
                        kind="police_response_unserved",
                        step=step,
                        event_step=request["event_step"],
                        event_id=request["event_id"],
                        location=request["location"],
                        reason="fixed_response_cap",
                    )
                )
                continue
            location = request["location"]
            officer_id = min(
                available,
                key=lambda key: (
                    self._centroids[self._officers[key].current_location].distance(
                        self._centroids[location]
                    ),
                    self._draw("police_response", step, request["event_id"], key),
                    key,
                ),
            )
            available.remove(officer_id)
            officer = self._officers[officer_id]
            self.logs.append(
                dict(
                    kind="police_response",
                    step=step,
                    event_step=request["event_step"],
                    event_id=request["event_id"],
                    officer_id=officer_id,
                    source_location=officer.current_location,
                    destination=location,
                )
            )
            officer.current_location = location
            allocated += 1

    def environment(self, location):
        """Return the original prompt schema, replacing only current safety_score."""
        location = str(location)
        if location not in self.world.map.environments:
            raise ValueError("Unknown environment location")
        source = self.world.map.environments[location]
        row = {key: deepcopy(source[key]) for key in ENV_FIELDS}
        if self.arm == "environment" and self._active(location):
            row["safety_score"] = min(
                1.0, self._source_safety[location] + self.parameters["safety_increment"]
            )
        return row

    def observations(self, location):
        location = str(location)
        if self.world.current_step < 0 or location not in self.world.map.aois:
            raise ValueError("Observations require a moved world and a known location")
        residents = [a for a in self.world.residents if a.current_location == location]
        visibility, step = (
            self.parameters["baseline_visibility"],
            self.world.current_step,
        )
        visible = [
            a
            for a in residents
            if self._draw("visible_resident", step, a.agent_id) < visibility
        ]
        proactive = sum(
            self._draw("proactive_activity", step, a.agent_id)
            < self._response_parameters(a.agent_id)[0]
            for a in visible
        )
        return dict(
            location=location,
            resident_count=len(residents),
            police_count=sum(a.current_location == location for a in self.world.police),
            visible_resident_count=len(visible),
            visible_proactive_resident_count=proactive,
            proactive_activity_description=(
                "Visible residents checking shared spaces and offering help; "
                "possible victims may be among them, so this count "
                "does not identify only third-party guardians."
            ),
            visibility_fraction=visibility,
            synthetic_physical_attributes=False,
            physical_cues=[],
            synthetic_park_count=0,
        )

    def finish_step(self, records):
        if self._finished or self.world.current_step < 0:
            raise ValueError("Step already finished or not yet moved")
        normalized, seen = [], set()
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("Event record must be a mapping")
            actor = _identifier(row.get("actor", row.get("actor_id")), "event actor")
            target = row.get("target", row.get("victim_id"))
            target = str(target) if target is not None else None
            location = _identifier(row.get("location"), "event location")
            if (
                row.get("step") != self.world.current_step
                or type(row.get("step")) is not int
            ):
                raise ValueError("Event step must match the current moved step")
            if actor in seen:
                raise ValueError("duplicate actor in finish_step records")
            seen.add(actor)
            if (
                actor not in self._actors
                or self._actors[actor].current_location != location
            ):
                raise ValueError("Unknown actor or actor location mismatch")
            if type(row.get("crime")) is not bool:
                raise ValueError("crime must be boolean")
            if "opportunity" in row and (
                type(row["opportunity"]) is not bool
                or (row["crime"] and not row["opportunity"])
            ):
                raise ValueError("Actual crime requires a boolean true opportunity")
            if row["crime"] and (
                target not in self._residents
                or self._residents[target].current_location != location
            ):
                raise ValueError("Actual event requires a co-located source victim")
            normalized.append((actor, location, row["crime"], target))
        for actor, location, crime, target in sorted(normalized):
            if crime:
                self._record_event(actor, location, target)
        self._finished = True

    def _record_event(self, actor, location, target):
        step = self.world.current_step
        event_id = f"{step}:{actor}"
        self.world.record_crime(actor, step, location)
        self.logs.append(
            dict(
                kind="actual_event",
                step=step,
                event_id=event_id,
                actor=actor,
                location=location,
                victim_id=target,
            )
        )
        visibility = self.parameters["baseline_visibility"]
        reporting, coordinating = [], []
        for resident_id, resident in sorted(self._residents.items()):
            if resident.current_location != location or resident_id == target:
                continue
            willingness, coordination = self._response_parameters(resident_id)
            draws = {
                rule: self._draw(rule, step, actor, resident_id)
                for rule in ("witness_observation", "help_willingness", "coordination")
            }
            observed = draws["witness_observation"] < visibility
            willing = observed and draws["help_willingness"] < willingness
            coordinated = willing and draws["coordination"] < coordination
            self.logs.append(
                dict(
                    kind="witness_response",
                    step=step,
                    event_id=event_id,
                    resident_id=resident_id,
                    location=location,
                    observed=observed,
                    requested_help=willing,
                    coordinated=coordinated,
                    draws=draws,
                    visibility=visibility,
                    willingness=willingness,
                    coordination=coordination,
                )
            )
            if willing:
                reporting.append(resident_id)
            if coordinated:
                coordinating.append(resident_id)
        request = dict(
            event_id=event_id,
            event_step=step,
            location=location,
            respond_at_step=step + 1,
        )
        if reporting:
            self.pending_help.append(dict(request, resident_ids=reporting))
            self.logs.append(
                dict(kind="help_request", step=step, **request, resident_ids=reporting)
            )
        if coordinating:
            self.logs.append(
                dict(
                    kind="resident_coordination",
                    step=step,
                    **request,
                    resident_ids=coordinating,
                )
            )

    def snapshot(self):
        """JSON replay state with identical baseline structure and no arm label."""
        populations = {
            role: getattr(self.world, role)
            for role in ("residents", "criminals", "police")
        }
        positions, histories = {}, {}
        for role, population in populations.items():
            ordered = sorted(population, key=lambda a: a.agent_id)
            positions[role] = {a.agent_id: a.current_location for a in ordered}
            histories[role] = {
                a.agent_id: {
                    "historical_trajectory": [
                        list(item) for item in a.historical_trajectory
                    ],
                    **(
                        {"visited_locations": dict(sorted(a.visited_locations.items()))}
                        if hasattr(a, "visited_locations")
                        else {}
                    ),
                    **(
                        {"criminal_record": [list(item) for item in a.criminal_record]}
                        if hasattr(a, "criminal_record")
                        else {}
                    ),
                }
                for a in ordered
            }
        effective = {
            key: self.environment(key)["safety_score"] for key in self._source_safety
        }
        return dict(
            step=self.world.current_step,
            finished=self._finished,
            plan_id=self._plan["plan_id"] if self._plan else None,
            targets=list(self.targets),
            participant_ids=list(self.participant_ids),
            population_ids={
                role: sorted(a.agent_id for a in population)
                for role, population in populations.items()
            },
            positions=positions,
            histories=histories,
            physical_attributes={
                key: dict(
                    visibility_fraction=self.parameters["baseline_visibility"],
                    cleared_refuse=False,
                    open_sightlines=False,
                    synthetic_park_count=0,
                )
                for key in self._source_safety
            },
            source_safety_scores=dict(self._source_safety),
            effective_safety_scores=effective,
            policy_dose=dict(
                hotspot_officers=(
                    len(self._hotspot_ids)
                    if self.arm == "hotspot" and self._active()
                    else 0
                ),
                active_participant_ids=(
                    list(self.participant_ids)
                    if self.arm == "collective" and self._active()
                    else []
                ),
                safety_increments=(
                    {
                        key: effective[key] - self._source_safety[key]
                        for key in self.targets
                    }
                    if self.arm == "environment" and self._active()
                    else {}
                ),
            ),
            pending_help=deepcopy(self.pending_help),
            log_count=len(self.logs),
            logs_sha256=_sha(self.logs),
        )
