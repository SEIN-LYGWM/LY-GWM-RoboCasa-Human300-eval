
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

RUN = Path('/home/bingxing2/home/scx9fvq/m489_robocasa365_gr00t_n1_5/03_contract/s32_pretrain50_lygwm_shadow')
import sys
sys.path.insert(0, str(RUN))
from joint_audit import verify_assets, verify_task_reports, final_evidence
STATE_PATH = RUN / "auto_submit_state.json"
TERMINAL = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
    "BOOT_FAIL", "DEADLINE", "REVOKED",
}

def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def log(message):
    print(stamp() + " " + message, flush=True)

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def save(state):
    state["updated_at"] = stamp()
    atomic(
        STATE_PATH,
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
    )

def command(args):
    # Retry read-only queries; never blindly repeat a submission.
    while True:
        try:
            result = subprocess.run(
                args, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=60,
            )
        except subprocess.TimeoutExpired:
            if args[0] not in ("squeue", "sacct"):
                raise
            log(
                "scheduler_query_timeout=" + args[0]
                + "; retry_after_seconds=30"
            )
            time.sleep(30)
            continue
        if result.returncode and args[0] in ("squeue", "sacct"):
            log(
                "scheduler_query_failed=" + args[0]
                + "; " + result.stderr.strip()
            )
            time.sleep(30)
            continue
        return result

def queue():
    result = command([
        "squeue", "-h", "-r", "-u", "scx9fvq",
        "-o", "%F|%K|%i|%T|%Z",
    ])
    if result.returncode:
        raise RuntimeError("squeue failed: " + result.stderr.strip())

    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) == 5 and fields[4].strip() == str(RUN):
            array, index, job, status, directory = [
                x.strip() for x in fields
            ]
            if not array.isdigit() or not index.isdigit():
                raise RuntimeError(
                    "Unexpected job in Target50 directory: " + line
                )
            rows.append({
                "array": array,
                "index": int(index),
                "state": status,
            })
    return rows

def inventory(manifest, digest):
    results = {}
    for task in manifest["tasks"]:
        index = task["task_index"]
        records = {}
        directory = (
            RUN / "tasks" / ("task_%03d" % index) / "episodes"
        )
        for path in directory.glob("episode_*.json"):
            item = read(path)
            episode = item["episode_index"]
            if not (
                type(episode) is int
                and 0 <= episode < 50
                and episode not in records
                and path.name == "episode_%03d.json" % episode
                and item["protocol_sha256"] == digest
                and item["task_index"] == index
                and item["engineering_valid"] is True
                and item["episode_completed"] is True
                and type(item["task_success"]) is bool
            ):
                raise RuntimeError(
                    "Invalid episode record: " + str(path)
                )
            records[episode] = item
        results[index] = records
    return results

def aggregate(manifest, results):
    groups = {}
    for task in manifest["tasks"]:
        group = groups.setdefault(
            task["task_group"], {"episodes": 0, "successes": 0}
        )
        records = results[task["task_index"]]
        group["episodes"] += len(records)
        group["successes"] += sum(
            int(x["task_success"]) for x in records.values()
        )

    return {
        "completed_tasks": sum(
            len(records) == 50 for records in results.values()
        ),
        "valid_episodes": sum(
            x["episodes"] for x in groups.values()
        ),
        "successes": sum(x["successes"] for x in groups.values()),
        "groups": groups,
    }

def main():
    # Retain the lock descriptor inherited from the launcher.
    os.fstat(int(os.environ["M489_AUTO_LOCK_FD"]))

    state = read(STATE_PATH) if STATE_PATH.exists() else {
        "started_at": stamp(),
        "current_batch": None,
        "submission_intent": None,
        "completed_batches": [],
    }

    try:
        manifest = read(RUN / "target50_manifest.json")
        protocol = read(RUN / "protocol.json")
        build = read(RUN / "build_report.json")
        verify_assets(RUN, protocol)

        digest = hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest()
        worker_sha = hashlib.sha256(
            (RUN / "worker.py").read_bytes()
        ).hexdigest()
        job_sha = hashlib.sha256(
            (RUN / "job.sh").read_bytes()
        ).hexdigest()

        if not (
            build["build_accepted"] is True
            and protocol["manifest"] == manifest
            and digest == build["protocol_sha256"]
            and worker_sha == build["worker_sha256"]
            and len(manifest["tasks"]) == 50
            and [
                t["task_index"] for t in manifest["tasks"]
            ] == list(range(50))
            and all(
                t["episode_count"] == 50
                for t in manifest["tasks"]
            )
        ):
            raise RuntimeError(
                "Manifest, protocol or worker validation failed"
            )

        for name, value in (
            ("protocol_sha256", digest),
            ("worker_sha256", worker_sha),
            ("job_sha256", job_sha),
        ):
            if name in state and state[name] != value:
                raise RuntimeError(
                    "Controller snapshot differs: " + name
                )
            state[name] = value

        state.update(
            status="starting",
            error=None,
            controller_pid=os.getpid(),
            controller_node=os.uname().nodename,
        )
        save(state)
        log("controller_started=True; maximum_batch_size=8")
        query_failures = 0

        while True:
            try:
                rows = queue()
                query_failures = 0
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                query_failures += 1
                if query_failures >= 5:
                    raise
                log("scheduler_query_retry=" + str(exc))
                time.sleep(60)
                continue

            current = state.get("current_batch")

            if rows:
                arrays = {row["array"] for row in rows}
                if len(arrays) != 1 or len(rows) > 8:
                    raise RuntimeError(
                        "Multiple arrays or more than eight active tasks found"
                    )
                array = next(iter(arrays))

                if current is None:
                    indices = sorted({
                        row["index"] for row in rows
                    })
                    if any(i not in range(50) for i in indices):
                        raise RuntimeError(
                            "Unexpected active task index"
                        )

                    intent = state.get("submission_intent")
                    if intent and not set(indices).issubset(
                        intent["task_indices"]
                    ):
                        raise RuntimeError(
                            "Active tasks differ from pending submission intent"
                        )

                    current = {
                        "array_job_id": array,
                        "task_indices": (
                            intent["task_indices"] if intent else indices
                        ),
                        "adopted": True,
                    }
                    state["current_batch"] = current
                    state["submission_intent"] = None
                    atomic(RUN / "array_job_id", array + "\n")
                    log("adopted_existing_array=" + array)

                elif current["array_job_id"] != array:
                    raise RuntimeError(
                        "Another array was submitted outside the controller"
                    )

                current.pop("accounting_wait_started", None)
                state["status"] = "waiting_for_batch"
                save(state)
                log(
                    "array=%s running=%d queued_or_other=%d"
                    % (
                        array,
                        sum(r["state"] == "RUNNING" for r in rows),
                        sum(r["state"] != "RUNNING" for r in rows),
                    )
                )
                time.sleep(60)
                continue

            if current is not None:
                array = current["array_job_id"]
                accounting = command([
                    "sacct", "-n", "-X", "-j", array,
                    "--format=JobID%40,State%30,ExitCode", "-P",
                ])
                if accounting.returncode:
                    raise RuntimeError(
                        "sacct failed: " + accounting.stderr.strip()
                    )

                records = {}
                for line in accounting.stdout.splitlines():
                    fields = [
                        x.strip() for x in line.split("|")
                    ]
                    if len(fields) >= 3 and fields[1]:
                        records[fields[0]] = (
                            fields[1].split()[0].rstrip("+"),
                            fields[2],
                        )

                expected = [
                    array + "_" + str(i)
                    for i in current["task_indices"]
                ]
                if not all(
                    key in records
                    and records[key][0] in TERMINAL
                    for key in expected
                ):
                    start = current.setdefault(
                        "accounting_wait_started", time.time()
                    )
                    if time.time() - start > 600:
                        raise RuntimeError(
                            "Terminal accounting did not become available: "
                            + array
                        )
                    state["status"] = "waiting_for_accounting"
                    save(state)
                    log("waiting_for_accounting=" + array)
                    time.sleep(60)
                    continue

                failures = {
                    key: records[key] for key in expected
                    if records[key] != ("COMPLETED", "0:0")
                }
                if failures:
                    raise RuntimeError(
                        "Batch has failed jobs: " + json.dumps(failures)
                    )

                results = inventory(manifest, digest)
                if any(
                    len(results[i]) != 50
                    for i in current["task_indices"]
                ):
                    raise RuntimeError(
                        "Completed batch is missing valid episode records"
                    )

                verify_task_reports(RUN, results, current['task_indices'], protocol)
                state["completed_batches"].append(current)
                state["current_batch"] = None
                save(state)
                log("batch_completed=" + array)

            elif state.get("submission_intent"):
                raise RuntimeError(
                    "Previous submission outcome is uncertain; "
                    "inspect state before resubmitting"
                )

            results = inventory(manifest, digest)
            totals = aggregate(manifest, results)
            state["totals"] = totals
            pending = [
                i for i in range(50) if len(results[i]) < 50
            ]

            if not pending:
                if totals["valid_episodes"] != 2500:
                    raise RuntimeError(
                        "Unexpected final episode count"
                    )
                final = dict(
                    totals,
                    all_2500_completed=True,
                    split="pretrain",
                    checkpoint_path=protocol["checkpoint_path"],
                    training_config_sha256=protocol["training_config_sha256"],
                    overall_success_rate=totals["successes"] / 2500,
                    protocol_sha256=digest,
                )
                verify_assets(RUN, protocol)
                final.update(final_evidence(RUN, results, protocol))
                for group in final["groups"].values():
                    group["success_rate"] = (
                        group["successes"] / group["episodes"]
                    )
                atomic(
                    RUN / "overall_results.json",
                    json.dumps(
                        final, ensure_ascii=False, indent=2
                    ) + "\n",
                )
                state.update(
                    status="completed",
                    finished_at=stamp(),
                    error=None,
                )
                save(state)
                log(
                    "all_2500_completed=True; successes="
                    + str(totals["successes"])
                )
                return

            for index in pending:
                path = (
                    RUN / "tasks" / ("task_%03d" % index)
                    / "task_summary.json"
                )
                if path.exists() and read(path).get("last_error"):
                    raise RuntimeError(
                        "Unresolved task error: " + str(path)
                    )

            if (
                hashlib.sha256(
                    (RUN / "worker.py").read_bytes()
                ).hexdigest() != worker_sha
                or hashlib.sha256(
                    (RUN / "job.sh").read_bytes()
                ).hexdigest() != job_sha
                or read(RUN / "protocol.json") != protocol
                or read(RUN / "target50_manifest.json") != manifest
            ):
                raise RuntimeError(
                    "Evaluation files changed while controller was running"
                )

            verify_assets(RUN, protocol)
            selected = pending[:1] if totals["valid_episodes"] == 0 else pending[:8]
            name = "m489_ly50_" + uuid.uuid4().hex[:8]
            state["submission_intent"] = {
                "task_indices": selected,
                "job_name": name,
                "created_at": stamp(),
            }
            state["status"] = "submitting"
            save(state)

            array_spec = ",".join(map(str, selected)) + "%8"
            submitted = command([
                "sbatch", "--parsable",
                "--job-name=" + name,
                "--partition=gpu", "--qos=gpugpu",
                "--nodes=1", "--ntasks=1", "--gres=gpu:1",
                "--array=" + array_spec,
                "--exclude=paraai-n32-h-01-agent-4",
                "--time=08:00:00",
                "--export=ALL",
                "--chdir=" + str(RUN),
                "--output=" + str(RUN / "slurm-%A_%a.out"),
                "--error=" + str(RUN / "slurm-%A_%a.out"),
                str(RUN / "job.sh"), str(RUN),
            ])

            atomic(RUN / "submission.out", submitted.stdout)
            atomic(RUN / "submission.err", submitted.stderr)

            if submitted.returncode:
                if any(
                    reason in submitted.stderr for reason in (
                        "AssocMaxSubmitJobLimit",
                        "QOSMaxSubmitJobPerUserLimit",
                        "QOSMaxSubmitJobPerAccountLimit",
                    )
                ):
                    state.update(
                        status="waiting_for_submit_quota",
                        submission_intent=None,
                    )
                    save(state)
                    log("submit_quota_busy; retry_after_seconds=60")
                    time.sleep(60)
                    continue
                raise RuntimeError(
                    "sbatch failed: " + submitted.stderr.strip()
                )

            match = re.fullmatch(
                r"(\d+)(?:;[^\s;]+)?", submitted.stdout.strip()
            )
            if not match:
                raise RuntimeError(
                    "Uncertain submission ID: " + submitted.stdout
                )

            array = match.group(1)
            state.update(
                current_batch={
                    "array_job_id": array,
                    "task_indices": selected,
                    "adopted": False,
                },
                submission_intent=None,
                status="waiting_for_batch",
            )
            save(state)
            atomic(RUN / "array_job_id", array + "\n")
            log(
                "submitted=True array_job_id=%s task_indices=%s"
                % (array, selected)
            )
            time.sleep(60)

    except Exception as exc:
        state.update(
            status="stopped_on_error",
            error=str(exc),
        )
        save(state)
        traceback.print_exc()
        log("controller_stopped_on_error=" + str(exc))
        raise

if __name__ == "__main__":
    main()
