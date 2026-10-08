"""Compose the frozen hotspot, environment and direct resident interventions.

The inherited arm selector stays ``hotspot`` throughout the world's lifetime:
it dispatches the original patrol placement and police assistance reservation.
ResidentResponseWorld independently supplies actual lagged resident movement
and baseline police reporting. Only the environment overlay and combined dose
accounting are added here; the runner labels the resulting study arm ``joint``.
"""

from .residents import ResidentResponseWorld


class JointPolicyWorld(ResidentResponseWorld):
    """One source population with all three original doses after activation."""

    def __init__(self, ready, seed, parameters=None, response_parameters=None):
        super().__init__(ready, "collective", seed, parameters, response_parameters)
        # A fixed per-instance dispatch selector, never a temporary/global patch.
        # Direct resident response does not branch on this selector; its police
        # reporting hook always retains the original baseline probabilities.
        self.arm = "hotspot"

    def move(self, step):
        super().move(step)
        if step != self.parameters["activation_step"]:
            return
        for kind in ("community_meeting", "project_start"):
            self.logs.append(
                dict(
                    kind=kind,
                    step=step,
                    targets=self.targets,
                    participant_ids=self.participant_ids,
                    willingness=self.parameters["collective_willingness"],
                    coordination=self.parameters["collective_coordination"],
                )
            )
        for location in self.targets:
            self.logs.append(
                dict(
                    kind="environment_activation",
                    step=step,
                    location=location,
                    original_safety_score=self._source_safety[location],
                    effective_safety_score=self.environment(location)["safety_score"],
                )
            )

    def environment(self, location):
        location = str(location)
        result = super().environment(location)
        if self._active(location):
            result["safety_score"] = min(
                1.0, self._source_safety[location] + self.parameters["safety_increment"]
            )
        return result

    def snapshot(self):
        result = super().snapshot()
        if self._active():
            result["policy_dose"].update(
                active_participant_ids=self.participant_ids,
                safety_increments={
                    key: result["effective_safety_scores"][key]
                    - self._source_safety[key]
                    for key in self.targets
                },
            )
        return result
