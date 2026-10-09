"""Backup and restore: repositories, snapshots, schedules and the jobs they run.

Reached as ``client.snapshots``. Register a repository (a directory, or any
S3-compatible bucket), then take, list, restore and delete snapshots in it.
Every route here needs the ``admin`` role on an RBAC node.

Snapshot, restore and cleanup are long-running: each returns a
:class:`~gnarl.SnapshotJob` at once, and the work runs in the background, one
job at a time per node. :meth:`Snapshots.wait` polls one to the end:

    job = c.snapshots.create("backups", "nightly-1", index="places")
    done = c.snapshots.wait(job)          # raises JobFailed if it failed
"""

from __future__ import annotations

import asyncio
import builtins
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from . import _models as m
from ._wire import _Call, _esc, _json_body
from .errors import GnarlError, JobFailed

if TYPE_CHECKING:
    from .client import AsyncClient, Client

__all__ = ["Snapshots", "AsyncSnapshots", "RepositorySpec"]

#: What :meth:`Snapshots.register_repository` accepts: a generated spec model,
#: or the same thing as a plain mapping with a ``type`` of ``fs`` or ``s3``.
RepositorySpec = m.FsRepositorySpec | m.S3RepositorySpec | Mapping[str, Any]

#: How long :meth:`Snapshots.wait` polls before giving up, in seconds.
DEFAULT_JOB_TIMEOUT = 3600.0


# ─── Calls, shared by both groups ───────────────────────────────────────────


def _repo(repo: str) -> str:
    if not repo:
        raise ValueError("gnarl: empty repository name")
    return f"/v1/repositories/{_esc(repo)}"


def _snap(repo: str, snapshot: str) -> str:
    if not snapshot:
        raise ValueError("gnarl: empty snapshot name")
    return f"{_repo(repo)}/snapshots/{_esc(snapshot)}"


def _list_repositories() -> _Call[list[m.RegisteredRepository]]:
    return _Call(
        "GET",
        "/v1/repositories",
        lambda raw: m.V1RepositoriesGetResponse.model_validate(raw).repositories or [],
    )


def _register(repo: str, spec: RepositorySpec) -> _Call[m.RegisteredRepository]:
    # Validated through the discriminated union, so a missing bucket or region
    # fails here naming the field — not as a 400 after a network round trip.
    if isinstance(spec, (m.FsRepositorySpec, m.S3RepositorySpec)):
        fields = spec.model_dump(exclude_unset=True)
    else:
        fields = dict(spec)
    model = m.RepositorySpec.model_validate(fields)
    return _Call(
        "PUT", _repo(repo), m.RegisteredRepository.model_validate, body=_json_body(model.root)
    )


def _get_repository(repo: str) -> _Call[m.RegisteredRepository]:
    return _Call("GET", _repo(repo), m.RegisteredRepository.model_validate)


def _unregister(repo: str) -> _Call[m.V1RepositoriesRepoDeleteResponse]:
    return _Call("DELETE", _repo(repo), m.V1RepositoriesRepoDeleteResponse.model_validate)


def _cleanup(repo: str, grace_seconds: int | None) -> _Call[m.SnapshotJob]:
    body = _json_body(
        m.V1RepositoriesRepoCleanupPostRequest.model_validate(
            {} if grace_seconds is None else {"grace_seconds": grace_seconds}
        )
    )
    return _Call("POST", _repo(repo) + "/_cleanup", m.SnapshotJob.model_validate, body=body)


def _get_schedule(repo: str) -> _Call[m.BackupSchedule | None]:
    return _Call(
        "GET",
        _repo(repo) + "/schedule",
        lambda raw: m.V1RepositoriesRepoScheduleGetResponse.model_validate(raw).schedule,
    )


def _set_schedule(
    repo: str, target: str, every_hours: int, enabled: bool | None, prefix: str | None
) -> _Call[m.BackupSchedule]:
    fields: dict[str, Any] = {"target": target, "everyHours": every_hours}
    if enabled is not None:
        fields["enabled"] = enabled
    if prefix is not None:
        fields["prefix"] = prefix
    body = _json_body(m.V1RepositoriesRepoSchedulePutRequest.model_validate(fields))

    def parse(raw: Any) -> m.BackupSchedule:
        schedule = m.V1RepositoriesRepoSchedulePutResponse.model_validate(raw).schedule
        if schedule is None:
            raise GnarlError(
                type="invalid_response",
                reason=f"PUT {_repo(repo)}/schedule: the node returned no schedule",
            )
        return schedule

    return _Call("PUT", _repo(repo) + "/schedule", parse, body=body)


def _clear_schedule(repo: str) -> _Call[bool]:
    return _Call(
        "DELETE",
        _repo(repo) + "/schedule",
        lambda raw: bool(
            m.V1RepositoriesRepoScheduleDeleteResponse.model_validate(raw).removed
        ),
    )


def _list(repo: str) -> _Call[list[str]]:
    return _Call(
        "GET",
        _repo(repo) + "/snapshots",
        lambda raw: (
            m.V1RepositoriesRepoSnapshotsGetResponse.model_validate(raw).snapshots or []
        ),
    )


def _create(
    repo: str,
    snapshot: str,
    index: str | None,
    namespace: str | None,
    allow_shared_pool: bool,
) -> _Call[m.SnapshotJob]:
    if (index is None) == (namespace is None):
        raise ValueError("gnarl: snapshots.create: give exactly one of index= or namespace=")
    fields: dict[str, Any] = {"index": index} if index else {"namespace": namespace}
    if allow_shared_pool:
        fields["allow_shared_pool"] = True
    body = _json_body(m.CreateSnapshotRequest.model_validate(fields))
    return _Call("PUT", _snap(repo, snapshot), m.SnapshotJob.model_validate, body=body)


def _get(repo: str, snapshot: str) -> _Call[m.SnapshotDescriptor]:
    return _Call("GET", _snap(repo, snapshot), m.SnapshotDescriptor.model_validate)


def _delete(
    repo: str, snapshot: str
) -> _Call[m.V1RepositoriesRepoSnapshotsSnapshotDeleteResponse]:
    return _Call(
        "DELETE",
        _snap(repo, snapshot),
        m.V1RepositoriesRepoSnapshotsSnapshotDeleteResponse.model_validate,
    )


def _restore(
    repo: str,
    snapshot: str,
    signer_public_key: str | None,
    allow_unverified_signer: bool,
    allow_overwrite_live_index: bool,
) -> _Call[m.SnapshotJob]:
    fields: dict[str, Any] = {}
    if signer_public_key:
        fields["signer_public_key"] = signer_public_key
    # Each consent flag is sent only when given: an explicit `false` would
    # read, in a request log, as somebody having considered and declined it.
    if allow_unverified_signer:
        fields["allow_unverified_signer"] = True
    if allow_overwrite_live_index:
        fields["allow_overwrite_live_index"] = True
    body = _json_body(m.RestoreRequest.model_validate(fields))
    return _Call(
        "POST", _snap(repo, snapshot) + "/_restore", m.SnapshotJob.model_validate, body=body
    )


def _jobs() -> _Call[list[m.SnapshotJob]]:
    return _Call(
        "GET",
        "/v1/snapshot_jobs",
        lambda raw: m.V1SnapshotJobsGetResponse.model_validate(raw).jobs or [],
    )


def _job(job_id: str) -> _Call[m.SnapshotJob]:
    if not job_id:
        raise ValueError("gnarl: empty job id")
    return _Call("GET", f"/v1/snapshot_jobs/{_esc(job_id)}", m.SnapshotJob.model_validate)


def _job_id(job: str | m.SnapshotJob) -> str:
    job_id = job if isinstance(job, str) else job.id
    if not job_id:
        raise ValueError("gnarl: snapshots.wait: the job has no id")
    return job_id


def _settled(job: m.SnapshotJob, raise_on_failure: bool) -> bool:
    """Whether polling can stop. A job with no state is still running as far
    as anyone can tell — stopping on it would report a result nobody had."""
    if job.state is None or job.state is m.State.running:
        return False
    if job.state is m.State.failed and raise_on_failure:
        raise JobFailed(job)
    return True


def _timed_out(job_id: str, timeout: float) -> GnarlError:
    return GnarlError(
        type="job_timeout",
        reason=(
            f"snapshot job {job_id} was still running after {timeout:.0f}s. It has "
            "not been cancelled; poll it again with snapshots.job()."
        ),
        detail={"job": job_id},
    )


# ─── Sync ───────────────────────────────────────────────────────────────────


class Snapshots:
    """``client.snapshots``. See the module docstring."""

    def __init__(self, client: Client) -> None:
        self._c = client

    # -- repositories --

    def list_repositories(self) -> list[m.RegisteredRepository]:
        """Every repository registered on this node. Credentials are never
        returned; the access key id is, to say which credential is in use."""
        return self._c._call(_list_repositories())

    def register_repository(self, repo: str, spec: RepositorySpec) -> m.RegisteredRepository:
        """Register (or replace) a repository. The node opens and lists it
        first, so a wrong bucket or credential fails HERE — not inside a
        scheduled backup where nobody reads the response."""
        return self._c._call(_register(repo, spec))

    def get_repository(self, repo: str) -> m.RegisteredRepository:
        return self._c._call(_get_repository(repo))

    def unregister_repository(self, repo: str) -> m.V1RepositoriesRepoDeleteResponse:
        """Forget a repository. Its data is left alone."""
        return self._c._call(_unregister(repo))

    def cleanup(self, repo: str, *, grace_seconds: int | None = None) -> m.SnapshotJob:
        """Start reclaiming segments no snapshot references. Objects younger
        than ``grace_seconds`` (node default: a day) are kept, because a
        snapshot in flight writes its segments before its descriptor."""
        return self._c._call(_cleanup(repo, grace_seconds))

    # -- schedules --

    def get_schedule(self, repo: str) -> m.BackupSchedule | None:
        """The repository's automatic backup, or ``None`` if it has none.
        A missing REPOSITORY raises :class:`~gnarl.NotFound` instead."""
        return self._c._call(_get_schedule(repo))

    def set_schedule(
        self,
        repo: str,
        target: str,
        every_hours: int,
        *,
        enabled: bool | None = None,
        prefix: str | None = None,
    ) -> m.BackupSchedule:
        """Back ``target`` up every ``every_hours`` (1 to 168).

        The node divides time into fixed UTC windows, so a machine asleep at
        the due moment still backs up when it wakes, a restart never re-runs a
        window already taken, and editing the schedule keeps that progress.
        """
        return self._c._call(_set_schedule(repo, target, every_hours, enabled, prefix))

    def clear_schedule(self, repo: str) -> bool:
        """Stop automatic backups. Returns whether there was a schedule to
        remove; backups already taken are untouched."""
        return self._c._call(_clear_schedule(repo))

    # -- snapshots --

    def list(self, repo: str) -> list[str]:
        """The snapshot names in a repository."""
        return self._c._call(_list(repo))

    def create(
        self,
        repo: str,
        snapshot: str,
        *,
        index: str | None = None,
        namespace: str | None = None,
        allow_shared_pool: bool = False,
    ) -> m.SnapshotJob:
        """Start a snapshot of ONE index or namespace. Returns the job.

        A promoted namespace is captured exactly; a pooled one is exported as
        documents (``kind: logical`` in the job result), which is exact in
        content but not a point-in-time cut. Snapshotting a ``__pool_N`` index
        by name captures every tenant in it and needs ``allow_shared_pool``.
        """
        return self._c._call(_create(repo, snapshot, index, namespace, allow_shared_pool))

    def get(self, repo: str, snapshot: str) -> m.SnapshotDescriptor:
        """The snapshot's descriptor. Read ``signature_verified``: a
        descriptor that merely parses proves nothing."""
        return self._c._call(_get(repo, snapshot))

    def delete(
        self, repo: str, snapshot: str
    ) -> m.V1RepositoriesRepoSnapshotsSnapshotDeleteResponse:
        """Delete the descriptor. Segments are reclaimed by :meth:`cleanup`."""
        return self._c._call(_delete(repo, snapshot))

    def restore(
        self,
        repo: str,
        snapshot: str,
        *,
        signer_public_key: str | None = None,
        allow_unverified_signer: bool = False,
        allow_overwrite_live_index: bool = False,
    ) -> m.SnapshotJob:
        """Start a restore. Returns the job.

        Refused (:class:`~gnarl.Conflict`) when this node cannot attribute the
        snapshot to itself or a known peer — pass the signer's key, or consent
        with ``allow_unverified_signer`` — and when it would roll a LIVE index
        back, unless ``allow_overwrite_live_index``.
        """
        return self._c._call(
            _restore(
                repo, snapshot, signer_public_key,
                allow_unverified_signer, allow_overwrite_live_index,
            )
        )

    # -- jobs --

    def jobs(self) -> builtins.list[m.SnapshotJob]:
        """Every job, newest first. History survives a restart; a job that was
        running when the node stopped reopens as ``failed``."""
        return self._c._call(_jobs())

    def job(self, job_id: str) -> m.SnapshotJob:
        return self._c._call(_job(job_id))

    def wait(
        self,
        job: str | m.SnapshotJob,
        *,
        timeout: float = DEFAULT_JOB_TIMEOUT,
        interval: float = 1.0,
        raise_on_failure: bool = True,
    ) -> m.SnapshotJob:
        """Poll a job until it is no longer running, and return it.

        Raises :class:`~gnarl.JobFailed` if it failed (unless
        ``raise_on_failure=False``), and a :class:`~gnarl.GnarlError` of type
        ``job_timeout`` if it is still running after ``timeout`` seconds —
        the job itself carries on.
        """
        job_id = _job_id(job)
        deadline = time.monotonic() + timeout
        while True:
            current = self.job(job_id)
            if _settled(current, raise_on_failure):
                return current
            if time.monotonic() >= deadline:
                raise _timed_out(job_id, timeout)
            time.sleep(interval)


# ─── Async ──────────────────────────────────────────────────────────────────


class AsyncSnapshots:
    """:class:`Snapshots`, awaited. Same surface; see it for the documentation."""

    def __init__(self, client: AsyncClient) -> None:
        self._c = client

    async def list_repositories(self) -> list[m.RegisteredRepository]:
        return await self._c._call(_list_repositories())

    async def register_repository(
        self, repo: str, spec: RepositorySpec
    ) -> m.RegisteredRepository:
        return await self._c._call(_register(repo, spec))

    async def get_repository(self, repo: str) -> m.RegisteredRepository:
        return await self._c._call(_get_repository(repo))

    async def unregister_repository(self, repo: str) -> m.V1RepositoriesRepoDeleteResponse:
        return await self._c._call(_unregister(repo))

    async def cleanup(self, repo: str, *, grace_seconds: int | None = None) -> m.SnapshotJob:
        return await self._c._call(_cleanup(repo, grace_seconds))

    async def get_schedule(self, repo: str) -> m.BackupSchedule | None:
        return await self._c._call(_get_schedule(repo))

    async def set_schedule(
        self,
        repo: str,
        target: str,
        every_hours: int,
        *,
        enabled: bool | None = None,
        prefix: str | None = None,
    ) -> m.BackupSchedule:
        return await self._c._call(_set_schedule(repo, target, every_hours, enabled, prefix))

    async def clear_schedule(self, repo: str) -> bool:
        return await self._c._call(_clear_schedule(repo))

    async def list(self, repo: str) -> list[str]:
        return await self._c._call(_list(repo))

    async def create(
        self,
        repo: str,
        snapshot: str,
        *,
        index: str | None = None,
        namespace: str | None = None,
        allow_shared_pool: bool = False,
    ) -> m.SnapshotJob:
        return await self._c._call(_create(repo, snapshot, index, namespace, allow_shared_pool))

    async def get(self, repo: str, snapshot: str) -> m.SnapshotDescriptor:
        return await self._c._call(_get(repo, snapshot))

    async def delete(
        self, repo: str, snapshot: str
    ) -> m.V1RepositoriesRepoSnapshotsSnapshotDeleteResponse:
        return await self._c._call(_delete(repo, snapshot))

    async def restore(
        self,
        repo: str,
        snapshot: str,
        *,
        signer_public_key: str | None = None,
        allow_unverified_signer: bool = False,
        allow_overwrite_live_index: bool = False,
    ) -> m.SnapshotJob:
        return await self._c._call(
            _restore(
                repo, snapshot, signer_public_key,
                allow_unverified_signer, allow_overwrite_live_index,
            )
        )

    async def jobs(self) -> builtins.list[m.SnapshotJob]:
        return await self._c._call(_jobs())

    async def job(self, job_id: str) -> m.SnapshotJob:
        return await self._c._call(_job(job_id))

    async def wait(
        self,
        job: str | m.SnapshotJob,
        *,
        timeout: float = DEFAULT_JOB_TIMEOUT,
        interval: float = 1.0,
        raise_on_failure: bool = True,
    ) -> m.SnapshotJob:
        job_id = _job_id(job)
        deadline = time.monotonic() + timeout
        while True:
            current = await self.job(job_id)
            if _settled(current, raise_on_failure):
                return current
            if time.monotonic() >= deadline:
                raise _timed_out(job_id, timeout)
            await asyncio.sleep(interval)
