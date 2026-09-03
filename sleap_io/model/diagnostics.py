"""Quality diagnostics for pose tracking data.

This module answers the question a new user has when handed a predictions file:
*"is this any good, and if not, what went wrong?"* It computes a set of cheap
self-consistency metrics over `Labels.numpy()` and turns them into plain-language
findings with suggested fixes.

Everything here is **ground-truth free**: it reports whether the predictions are
internally consistent (plausible tracks, stable skeletons, confident points), not
whether they are *correct*. Accuracy against labeled data is a separate question
that needs a ground-truth file and lives in the training package.

The metrics:

- **Track occupancy** — the fraction of frames each track appears in. Many
  near-empty tracks means the tracker spawned spurious identities.
- **Node visibility** — how often each body part was detected at all. Names the
  parts the model fails on.
- **Confidence** — the distribution of point scores, and how much of it sits
  below a usable threshold.
- **Fragmentation** — how often tracks stop and restart, a proxy for identity
  switches.
- **Displacement outliers** — frames where an instance teleports, the signature
  of an identity swap. Reported as frame indices so they can be rendered.
- **Rigid segment stability** — the length of each skeleton edge should be
  roughly constant for a rigid body part. Its coefficient of variation is a
  units-free readout of keypoint jitter.

Example:
    >>> import sleap_io as sio
    >>> labels = sio.load_slp("predictions.slp")
    >>> report = labels.diagnose()
    >>> print(report.summary())
    >>> report.findings[0].frames  # frames worth looking at
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import attrs
import numpy as np

if TYPE_CHECKING:
    from sleap_io.model.labels import Labels

#: Tracks occupying a smaller fraction of frames than this are considered
#: spurious (a real animal is present for most of a recording).
SPURIOUS_TRACK_OCCUPANCY = 0.01

#: Nodes detected in a smaller fraction of instances than this are flagged as
#: poorly learned body parts.
LOW_NODE_VISIBILITY = 0.5

#: Point scores below this are treated as unusable predictions.
LOW_CONFIDENCE = 0.3

#: Flag the confidence distribution when more than this fraction of visible
#: points falls below `LOW_CONFIDENCE`.
LOW_CONFIDENCE_FRACTION = 0.05

#: A frame-to-frame centroid displacement larger than this multiple of the
#: median displacement is treated as a teleport (likely identity swap).
TELEPORT_FACTOR = 10.0

#: Skeleton edges whose length varies by more than this coefficient of variation
#: are flagged as unstable. Genuinely articulated joints will exceed this too, so
#: it is a pointer rather than a verdict.
UNSTABLE_SEGMENT_CV = 0.25

#: Maximum number of frame indices attached to a single finding.
MAX_REPORTED_FRAMES = 20


@attrs.define
class Finding:
    """A single diagnostic result with a suggested fix.

    Attributes:
        code: Stable machine-readable identifier, e.g. `"spurious_tracks"`.
        severity: One of `"info"`, `"warning"`, or `"error"`.
        message: What was found, including the numbers that support it.
        suggestion: What the user should do about it.
        frames: Frame indices worth inspecting, capped at
            `MAX_REPORTED_FRAMES`. Empty when the finding is not localized to
            particular frames.
    """

    code: str
    severity: str
    message: str
    suggestion: str
    frames: list[int] = attrs.field(factory=list)

    def __str__(self) -> str:
        """Return a one-line rendering of the finding."""
        line = f"[{self.severity}] {self.message}"
        if self.frames:
            shown = ", ".join(str(f) for f in self.frames[:5])
            more = f", ... (+{len(self.frames) - 5})" if len(self.frames) > 5 else ""
            line += f"\n    frames: {shown}{more}"
        return f"{line}\n    -> {self.suggestion}"


@attrs.define
class Diagnostics:
    """Quality report for a set of pose predictions.

    Returned by `Labels.diagnose()`. Holds the raw metric arrays alongside the
    interpreted `findings`, so callers can either read the summary or do their
    own analysis on the numbers.

    Attributes:
        n_frames: Number of frames analyzed.
        n_tracks: Size of the track axis analyzed. When the labels have no
            tracks this is the maximum number of instances in any frame.
        n_nodes: Number of skeleton nodes.
        node_names: Skeleton node names, ordered to match `node_visibility`.
        tracked: Whether the analyzed instances carried track assignments.
        track_occupancy: Fraction of frames each track is present in, shape
            `(n_tracks,)`.
        node_visibility: Fraction of *present* instances in which each node
            was detected, shape `(n_nodes,)`. Instance slots with no visible
            nodes at all are excluded from the denominator.
        confidence: Summary of predicted point scores with keys `"mean"`,
            `"p05"`, `"p50"`, `"p95"`, and `"fraction_below"`. All values are
            `nan` for labels with no predicted instances, since `Labels.numpy()`
            reports a score of 1.0 for user-annotated points.
        track_breaks: Number of times each track stops after being present,
            shape `(n_tracks,)`.
        displacement: Summary of frame-to-frame centroid motion in pixels with
            keys `"median"`, `"p99"`, and `"max"`.
        outlier_frames: Frame indices where an instance moved more than
            `TELEPORT_FACTOR` times the median displacement.
        segment_cv: Coefficient of variation of each skeleton edge's length,
            shape `(n_edges,)`, ordered to match `segment_names`.
        segment_names: `"src -- dst"` label for each skeleton edge.
        findings: Interpreted results, ordered most severe first.
    """

    n_frames: int
    n_tracks: int
    n_nodes: int
    node_names: list[str]
    tracked: bool
    track_occupancy: np.ndarray
    node_visibility: np.ndarray
    confidence: dict[str, float]
    track_breaks: np.ndarray
    displacement: dict[str, float]
    outlier_frames: list[int]
    segment_cv: np.ndarray
    segment_names: list[str]
    findings: list[Finding] = attrs.field(factory=list)

    @property
    def n_spurious_tracks(self) -> int:
        """Number of tracks occupying less than `SPURIOUS_TRACK_OCCUPANCY`."""
        return int((self.track_occupancy < SPURIOUS_TRACK_OCCUPANCY).sum())

    def to_dict(self) -> dict[str, Any]:
        """Return the report as JSON-serializable nested dictionaries.

        Returns:
            A dictionary with the same structure as the attributes, with arrays
            converted to lists and findings to dictionaries.
        """
        return {
            "n_frames": self.n_frames,
            "n_tracks": self.n_tracks,
            "n_nodes": self.n_nodes,
            "node_names": list(self.node_names),
            "tracked": self.tracked,
            "track_occupancy": self.track_occupancy.tolist(),
            "node_visibility": self.node_visibility.tolist(),
            "confidence": dict(self.confidence),
            "track_breaks": self.track_breaks.tolist(),
            "displacement": dict(self.displacement),
            "outlier_frames": list(self.outlier_frames),
            "segment_cv": self.segment_cv.tolist(),
            "segment_names": list(self.segment_names),
            "findings": [attrs.asdict(f) for f in self.findings],
        }

    def summary(self) -> str:
        """Return a human-readable report.

        Returns:
            A multi-line string with the headline counts followed by each
            finding and its suggested fix.
        """
        lines = [
            f"{self.n_frames} frames, {self.n_tracks} "
            f"{'tracks' if self.tracked else 'instance slots'}, "
            f"{self.n_nodes} nodes",
        ]
        if not np.isnan(self.confidence["mean"]):
            lines.append(
                f"confidence: mean={self.confidence['mean']:.3f} "
                f"p05={self.confidence['p05']:.3f} "
                f"below {LOW_CONFIDENCE}={self.confidence['fraction_below']:.1%}"
            )
        lines.append(
            f"displacement px/frame: median={self.displacement['median']:.2f} "
            f"p99={self.displacement['p99']:.2f} max={self.displacement['max']:.2f}"
        )
        if not self.findings:
            lines.append("\nNo issues found.")
        else:
            lines.append("")
            lines.extend(str(f) for f in self.findings)
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return `summary()`."""
        return self.summary()


def _severity_rank(finding: Finding) -> int:
    """Return a sort key ordering findings from most to least severe."""
    return {"error": 0, "warning": 1, "info": 2}.get(finding.severity, 3)


def _summarize_confidence(scores: np.ndarray) -> dict[str, float]:
    """Summarize the distribution of visible point scores.

    Args:
        scores: Scores of visible predicted points, shape `(n_points,)`. Pass an
            empty array for labels that carry no predictions.

    Returns:
        A dictionary with keys `"mean"`, `"p05"`, `"p50"`, `"p95"`, and
        `"fraction_below"`. All values are `nan` when no scores are available.
    """
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        keys = ("mean", "p05", "p50", "p95", "fraction_below")
        return {k: float("nan") for k in keys}
    p05, p50, p95 = np.percentile(finite, [5, 50, 95])
    return {
        "mean": float(finite.mean()),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "fraction_below": float(np.mean(finite < LOW_CONFIDENCE)),
    }


def _segment_stability(xy: np.ndarray, edge_inds: list[tuple[int, int]]) -> np.ndarray:
    """Compute the coefficient of variation of each skeleton edge's length.

    A rigid body segment should keep a constant length regardless of pose, so
    the spread of its measured length is a units-free readout of keypoint noise.

    Args:
        xy: Point coordinates, shape `(n_frames, n_tracks, n_nodes, 2)`.
        edge_inds: Source and destination node indices for each edge.

    Returns:
        Coefficient of variation per edge, shape `(n_edges,)`. Entries are `nan`
        when an edge is never observed with both endpoints visible.
    """
    if not edge_inds:
        return np.zeros((0,), dtype=float)

    src = np.array([e[0] for e in edge_inds])
    dst = np.array([e[1] for e in edge_inds])
    # (n_frames, n_tracks, n_edges)
    lengths = np.linalg.norm(xy[:, :, src, :] - xy[:, :, dst, :], axis=-1)
    flat = lengths.reshape(-1, len(edge_inds))

    cv = np.full(len(edge_inds), np.nan)
    for i in range(len(edge_inds)):
        vals = flat[:, i]
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            continue
        mean = vals.mean()
        if mean > 0:
            cv[i] = float(vals.std() / mean)
    return cv


def _build_findings(report: Diagnostics) -> list[Finding]:
    """Interpret computed metrics into findings with suggested fixes.

    Args:
        report: A `Diagnostics` with its metric fields populated and `findings`
            still empty.

    Returns:
        Findings ordered most severe first.
    """
    findings: list[Finding] = []

    if report.tracked:
        n_spurious = report.n_spurious_tracks
        if n_spurious:
            n_kept = report.n_tracks - n_spurious
            findings.append(
                Finding(
                    code="spurious_tracks",
                    severity="warning",
                    message=(
                        f"{n_spurious} of {report.n_tracks} tracks occupy less "
                        f"than {SPURIOUS_TRACK_OCCUPANCY:.0%} of frames "
                        f"({n_kept} above that)."
                    ),
                    suggestion=(
                        "The tracker spawned identities it could not sustain. "
                        "Run `Labels.clean()` to drop empty tracks, or re-track "
                        "with a stricter matching threshold."
                    ),
                )
            )

        total_breaks = int(report.track_breaks.sum())
        if total_breaks:
            n_breaking = int((report.track_breaks > 0).sum())
            findings.append(
                Finding(
                    code="track_fragmentation",
                    severity="warning" if total_breaks > report.n_tracks else "info",
                    message=(
                        f"{total_breaks} track interruptions across "
                        f"{n_breaking} tracks."
                    ),
                    suggestion=(
                        "Identities are being lost and re-created mid-recording. "
                        "Increase the tracker's gap-closing window, or proofread "
                        "the breaks in the GUI."
                    ),
                )
            )

    low_vis = np.flatnonzero(report.node_visibility < LOW_NODE_VISIBILITY)
    if low_vis.size:
        worst = low_vis[np.argsort(report.node_visibility[low_vis])][:5]
        named = ", ".join(
            f"{report.node_names[i]} ({report.node_visibility[i]:.0%})" for i in worst
        )
        findings.append(
            Finding(
                code="low_node_visibility",
                severity="warning",
                message=(
                    f"{low_vis.size} of {report.n_nodes} nodes are detected in "
                    f"under {LOW_NODE_VISIBILITY:.0%} of instances: {named}."
                ),
                suggestion=(
                    "These body parts are under-learned. Label more frames where "
                    "they are visible, or remove them from the skeleton if they "
                    "are genuinely not observable."
                ),
            )
        )

    frac_below = report.confidence["fraction_below"]
    if np.isfinite(frac_below) and frac_below > LOW_CONFIDENCE_FRACTION:
        findings.append(
            Finding(
                code="low_confidence",
                severity="warning",
                message=(
                    f"{frac_below:.1%} of detected points score below "
                    f"{LOW_CONFIDENCE} (mean {report.confidence['mean']:.3f})."
                ),
                suggestion=(
                    "A large low-confidence tail usually means the model is "
                    "guessing. Filter these points out before analysis, or train "
                    "longer on more representative frames."
                ),
            )
        )

    if report.outlier_frames:
        findings.append(
            Finding(
                code="displacement_outliers",
                severity="warning",
                message=(
                    f"{len(report.outlier_frames)} frames where an instance "
                    f"moved more than {TELEPORT_FACTOR:.0f}x the median "
                    f"displacement (median "
                    f"{report.displacement['median']:.2f} px/frame, max "
                    f"{report.displacement['max']:.2f})."
                ),
                suggestion=(
                    "Instances that teleport between frames are usually identity "
                    "swaps. Render these frames to confirm, then fix the track "
                    "assignments."
                ),
                frames=report.outlier_frames[:MAX_REPORTED_FRAMES],
            )
        )

    unstable = np.flatnonzero(report.segment_cv > UNSTABLE_SEGMENT_CV)
    if unstable.size:
        worst = unstable[np.argsort(report.segment_cv[unstable])[::-1]][:5]
        named = ", ".join(
            f"{report.segment_names[i]} (cv={report.segment_cv[i]:.2f})" for i in worst
        )
        findings.append(
            Finding(
                code="unstable_segments",
                severity="info",
                message=(
                    f"{unstable.size} skeleton segments vary in length by more "
                    f"than {UNSTABLE_SEGMENT_CV:.0%}: {named}."
                ),
                suggestion=(
                    "Rigid body segments should keep a constant length, so a "
                    "high spread points at jittery keypoints. Articulated joints "
                    "legitimately vary, so check these against the skeleton "
                    "before acting."
                ),
            )
        )

    findings.sort(key=_severity_rank)
    return findings


def _nanmean_points(xy: np.ndarray) -> np.ndarray:
    """Average visible node coordinates per instance, ignoring missing points.

    Args:
        xy: Point coordinates, shape `(n_frames, n_tracks, n_nodes, 2)`.

    Returns:
        Instance centroids of shape `(n_frames, n_tracks, 2)`. Slots with no
        visible nodes are `nan`.
    """
    with np.errstate(invalid="ignore"):
        counts = np.sum(~np.isnan(xy[..., 0]), axis=2)
        totals = np.nansum(xy, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts[..., None] > 0, totals / counts[..., None], np.nan)


def _empty_report(labels: "Labels", tracked: bool) -> Diagnostics:
    """Build a report for labels holding no frames.

    Args:
        labels: The empty `Labels`.
        tracked: Whether track-based analysis was requested.

    Returns:
        A `Diagnostics` with zeroed metrics and a single `"no_data"` finding.
    """
    skeleton = labels.skeleton if labels.skeletons else None
    node_names = list(skeleton.node_names) if skeleton else []
    nan = float("nan")
    return Diagnostics(
        n_frames=0,
        n_tracks=0,
        n_nodes=len(node_names),
        node_names=node_names,
        tracked=tracked,
        track_occupancy=np.zeros(0),
        node_visibility=np.zeros(len(node_names)),
        confidence={
            "mean": nan,
            "p05": nan,
            "p50": nan,
            "p95": nan,
            "fraction_below": nan,
        },
        track_breaks=np.zeros(0, dtype=int),
        displacement={"median": 0.0, "p99": 0.0, "max": 0.0},
        outlier_frames=[],
        segment_cv=np.zeros(0),
        segment_names=[],
        findings=[
            Finding(
                code="no_data",
                severity="error",
                message="These labels contain no labeled frames.",
                suggestion=(
                    "Nothing was predicted or annotated. Check that inference "
                    "actually ran and wrote to this path, and that the video "
                    "it was pointed at could be opened."
                ),
            )
        ],
    )


def diagnose(
    labels: "Labels",
    video: Any = None,
    untracked: bool | None = None,
) -> Diagnostics:
    """Compute a quality report for a set of pose predictions.

    Args:
        labels: The `Labels` to analyze.
        video: Video, filename, or video index to analyze. If `None` (the
            default), uses the first video.
        untracked: If `False`, analyze only instances with a track assignment.
            If `True`, analyze all instances in arbitrary per-frame order. If
            `None` (the default), use tracks when the labels have any and fall
            back to untracked instances otherwise.

    Returns:
        A `Diagnostics` report holding the metric arrays and the interpreted
        findings.

    Notes:
        This is a self-consistency check: it does not compare against ground
        truth, so it can say predictions look implausible but not that they are
        wrong. Track-related findings are skipped when analyzing untracked
        instances, since the track axis is then an arbitrary per-frame ordering.
    """
    if untracked is None:
        untracked = len(labels.tracks) == 0
    tracked = not untracked

    if not labels.videos or not labels.labeled_frames:
        # A file with nothing in it is the most common result of a failed
        # inference run, so report it rather than failing to analyze it.
        return _empty_report(labels, tracked)

    arr = labels.numpy(video=video, untracked=untracked, return_confidence=True)
    xy, scores = arr[..., :2], arr[..., 2]
    n_frames, n_tracks, n_nodes = scores.shape
    visible = ~np.isnan(xy).any(axis=-1)

    present = visible.any(axis=2)  # (n_frames, n_tracks)
    track_occupancy = present.mean(axis=0) if n_frames else np.zeros(n_tracks)
    # Averaged over *present* instance slots only: empty track slots would
    # otherwise dilute every node toward zero on a file with spurious tracks.
    node_visibility = (
        visible[present].mean(axis=0) if present.any() else np.zeros(n_nodes)
    )
    track_breaks = (
        (present[:-1] & ~present[1:]).sum(axis=0)
        if n_frames > 1
        else np.zeros(n_tracks, dtype=int)
    )

    # Centroid motion between consecutive frames, per track.
    centroids = _nanmean_points(xy)  # (n_frames, n_tracks, 2)
    steps = (
        np.linalg.norm(np.diff(centroids, axis=0), axis=-1)
        if n_frames > 1
        else np.zeros((0, n_tracks))
    )
    finite_steps = steps[np.isfinite(steps)]
    if finite_steps.size:
        median_step = float(np.median(finite_steps))
        displacement = {
            "median": median_step,
            "p99": float(np.percentile(finite_steps, 99)),
            "max": float(finite_steps.max()),
        }
        # `steps[i]` is the motion into frame `i + 1`.
        outlier_rows = np.unique(
            np.nonzero(steps > TELEPORT_FACTOR * median_step)[0] + 1
        )
        outlier_frames = [int(i) for i in outlier_rows]
    else:
        displacement = {"median": 0.0, "p99": 0.0, "max": 0.0}
        outlier_frames = []

    skeleton = labels.skeleton if labels.skeletons else None
    node_names = list(skeleton.node_names) if skeleton else []
    edge_inds = list(skeleton.edge_inds) if skeleton else []
    segment_cv = _segment_stability(xy, edge_inds)
    segment_names = [f"{node_names[a]} -- {node_names[b]}" for a, b in edge_inds]

    report = Diagnostics(
        n_frames=int(n_frames),
        n_tracks=int(n_tracks),
        n_nodes=int(n_nodes),
        node_names=node_names,
        tracked=tracked,
        track_occupancy=track_occupancy,
        node_visibility=node_visibility,
        confidence=_summarize_confidence(
            scores[visible] if labels.n_pred_instances else np.zeros(0)
        ),
        track_breaks=track_breaks,
        displacement=displacement,
        outlier_frames=outlier_frames,
        segment_cv=segment_cv,
        segment_names=segment_names,
    )
    report.findings = _build_findings(report)
    return report
