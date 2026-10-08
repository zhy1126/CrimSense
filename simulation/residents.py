"""Direct, lagged resident assistance on top of the frozen SAT city world.

Only the resident policy changes. Existing police reporting always uses its
baseline dose. Extra coordination creates real next-step resident presence.
The preperiod retains the old state, observation and log representation exactly.
"""

from copy import deepcopy
import math

from .policies import PolicyWorld

RESPONSE_DEFAULTS = dict(radius_m=1000.0, max_helpers=2, duration_steps=1)


def response_parameters(values=None):
    if values is not None and not isinstance(values, dict):
        raise ValueError("Response parameters must be a mapping")
    if set(values or {}) - RESPONSE_DEFAULTS.keys():
        raise ValueError("Unknown resident response parameter")
    result = dict(RESPONSE_DEFAULTS, **(values or {}))
    radius = result["radius_m"]
    if (
        isinstance(radius, bool)
        or not isinstance(radius, (int, float))
        or not math.isfinite(radius)
        or radius < 0
    ):
        raise ValueError("Response radius must be finite and nonnegative")
    helpers = result["max_helpers"]
    if isinstance(helpers, bool) or not isinstance(helpers, int) or helpers < 0:
        raise ValueError("max_helpers must be a nonnegative integer")
    if type(result["duration_steps"]) is not int or result["duration_steps"] != 1:
        raise ValueError("Only the approved one-step response is implemented")
    return result


class ResidentResponseWorld(PolicyWorld):
    def __init__(self, ready, arm, seed, parameters=None, response_parameters=None):
        if arm != "collective":
            raise ValueError("This revision runs only the resident collective arm")
        super().__init__(ready, arm, seed, parameters)
        self.response_parameters = globals()["response_parameters"](response_parameters)
        self.pending_resident_response = []
        self.active_resident_responses = {}

    def _response_parameters(self, resident_id):
        # This hook belongs to old police reporting and generic ambient activity.
        # New resident cooperation has an independent queue and does not raise it.
        return (
            self.parameters["baseline_willingness"],
            self.parameters["baseline_coordination"],
        )

    def _record_event(self, actor, location, target):
        super()._record_event(actor, location, target)
        if not self._active():
            return
        step = self.world.current_step
        event_id = f"{step}:{actor}"
        visibility = self.parameters["baseline_visibility"]
        willingness = self.parameters["collective_willingness"]
        coordination = self.parameters["collective_coordination"]
        initiators = []
        for resident_id in self.participant_ids:
            resident = self._residents[resident_id]
            if resident.current_location != location or resident_id == target:
                continue
            draws = {
                rule: self._draw(rule, step, actor, resident_id)
                for rule in ("witness_observation", "help_willingness", "coordination")
            }
            observed = draws["witness_observation"] < visibility
            willing = observed and draws["help_willingness"] < willingness
            coordinated = willing and draws["coordination"] < coordination
            self.logs.append(
                dict(
                    kind="resident_direct_witness",
                    step=step,
                    event_id=event_id,
                    resident_id=resident_id,
                    location=location,
                    observed=observed,
                    willing=willing,
                    coordinated=coordinated,
                    draws=draws,
                    visibility=visibility,
                    willingness=willingness,
                    coordination=coordination,
                )
            )
            if coordinated:
                initiators.append(resident_id)
        if not initiators:
            return
        initiator = min(
            initiators,
            key=lambda rid: (self._draw("resident_initiator", step, actor, rid), rid),
        )
        candidates = []
        for resident_id in self.participant_ids:
            if resident_id in (initiator, target):
                continue
            origin = self._residents[resident_id].current_location
            distance = self._centroids[origin].distance(self._centroids[location])
            if distance > self.response_parameters["radius_m"]:
                continue
            draw = self._draw("neighbor_help", step, actor, resident_id)
            willing = draw < willingness
            self.logs.append(
                dict(
                    kind="resident_invitation",
                    step=step,
                    event_id=event_id,
                    resident_id=resident_id,
                    origin_at_event=origin,
                    location=location,
                    distance_m=distance,
                    willing=willing,
                    draw=draw,
                    willingness=willingness,
                )
            )
            if willing:
                candidates.append(
                    (
                        distance,
                        self._draw("neighbor_priority", step, actor, resident_id),
                        resident_id,
                    )
                )
        helpers = [
            rid
            for _, _, rid in sorted(candidates)[
                : self.response_parameters["max_helpers"]
            ]
        ]
        resident_ids = [initiator] + helpers
        origins = {rid: self._residents[rid].current_location for rid in resident_ids}
        request = dict(
            event_id=event_id,
            event_step=step,
            location=location,
            respond_at_step=step + 1,
            initiator_id=initiator,
            resident_ids=resident_ids,
            origins_at_event=origins,
        )
        self.pending_resident_response.append(request)
        self.logs.append(
            dict(kind="resident_response_request", step=step, **deepcopy(request))
        )

    def move(self, step):
        super().move(step)
        self.active_resident_responses = {}
        if not self._active():
            return
        due = [
            r for r in self.pending_resident_response if r["respond_at_step"] <= step
        ]
        self.pending_resident_response = [
            r for r in self.pending_resident_response if r["respond_at_step"] > step
        ]
        for request in sorted(due, key=lambda r: r["event_id"]):
            if request["respond_at_step"] != step:
                raise RuntimeError("Missed resident-response step")
            for resident_id in request["resident_ids"]:
                if resident_id in self.active_resident_responses:
                    self.logs.append(
                        dict(
                            kind="resident_response_unserved",
                            step=step,
                            event_step=request["event_step"],
                            event_id=request["event_id"],
                            resident_id=resident_id,
                            location=request["location"],
                            reason="already_responding",
                        )
                    )
                    continue
                resident = self._residents[resident_id]
                destination = request["location"]
                origin = request["origins_at_event"][resident_id]
                distance = self._centroids[origin].distance(
                    self._centroids[destination]
                )
                if (
                    resident_id not in self._participant_set
                    or distance > self.response_parameters["radius_m"]
                ):
                    raise RuntimeError(
                        "Resident response outside registered identity or distance"
                    )
                scheduled = resident.current_location
                self._redirect_resident(resident, destination)
                entry = dict(
                    kind="resident_response",
                    step=step,
                    event_step=request["event_step"],
                    event_id=request["event_id"],
                    resident_id=resident_id,
                    source_location=origin,
                    origin_at_event=origin,
                    scheduled_location=scheduled,
                    destination=destination,
                    distance_m=distance,
                    activity="Watching the shared space and assisting neighbors after an earlier incident.",
                )
                self.active_resident_responses[resident_id] = entry
                self.logs.append(deepcopy(entry))

    def _redirect_resident(self, resident, destination):
        scheduled = resident.current_location
        if destination not in self.world.map.aois:
            raise ValueError("Unknown resident-response destination")
        if scheduled == destination:
            return
        history = resident.historical_trajectory
        if (
            not history
            or history[-1][1] != scheduled
            or resident.visited_locations.get(scheduled, 0) < 1
        ):
            raise RuntimeError("Resident scheduled movement history is inconsistent")
        count = resident.visited_locations[scheduled] - 1
        if count:
            resident.visited_locations[scheduled] = count
        else:
            del resident.visited_locations[scheduled]
        resident.visited_locations[destination] = (
            resident.visited_locations.get(destination, 0) + 1
        )
        history[-1] = (history[-1][0], destination)
        resident.current_location = destination

    def observations(self, location):
        result = super().observations(location)
        responders = {
            rid
            for rid, e in self.active_resident_responses.items()
            if e["destination"] == str(location)
        }
        if not responders:
            return result
        local = {
            a.agent_id
            for a in self.world.residents
            if a.current_location == str(location)
        }
        if not responders <= local:
            raise RuntimeError("Visible responders are not physically present")
        step = self.world.current_step
        ambient = {
            rid
            for rid in local
            if self._draw("visible_resident", step, rid)
            < self.parameters["baseline_visibility"]
        }
        proactive = {
            rid
            for rid in ambient
            if self._draw("proactive_activity", step, rid)
            < self.parameters["baseline_willingness"]
        }
        result.update(
            visible_resident_count=len(ambient | responders),
            visible_proactive_resident_count=len(proactive | responders),
            visible_resident_response_count=len(responders),
            visible_resident_response_ids=sorted(responders),
            resident_response_activity=(
                "These residents are physically present, watching the shared space and "
                "assisting neighbors following an earlier incident. They are included "
                "in the visible resident and proactive resident counts."
            ),
        )
        return result

    def snapshot(self):
        result = super().snapshot()
        if self._active():
            result.update(
                response_parameters=deepcopy(self.response_parameters),
                pending_resident_response=deepcopy(self.pending_resident_response),
                active_resident_responses=deepcopy(self.active_resident_responses),
            )
        return result
