import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path("/home/bingxing2/home/scx9fvq/m489_robocasa365_gr00t_n1_5")
RUN = Path(__file__).resolve().parent
POLICY = ROOT / "02_env/gr00t_n15_py310_t251_cu121"
SIM = ROOT / "02_env/robocasa101_sim_py310_t251_cu121"
THIRD = Path("/home/bingxing2/home/scx9fvq/LY-GWM-RC/third_party")
SOURCE = ROOT / (
    "01_source/Isaac-GR00T-"
    "9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10"
    "_s5_source_recovery_20260909_140603"
)
CHECKPOINT = Path('/home/bingxing2/home/scx9fvq/m489_robocasa365_gr00t_n1_5/04_train_repro/r5_formal_train/formal_trainer/checkpoint-120000')
JOB = os.environ["SLURM_JOB_ID"]
ROLE = sys.argv[1]

# M489_S30_TARGET50_BATCHED_V1
import hashlib
import signal
import subprocess
import time
import traceback

MASTER = Path(__file__).resolve().parent
ROOT = Path("/home/bingxing2/home/scx9fvq/m489_robocasa365_gr00t_n1_5")
POLICY = ROOT / "02_env/gr00t_n15_py310_t251_cu121"
SIM = ROOT / "02_env/robocasa101_sim_py310_t251_cu121"
THIRD = Path("/home/bingxing2/home/scx9fvq/LY-GWM-RC/third_party")
SOURCE = ROOT / (
    "01_source/Isaac-GR00T-"
    "9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10"
    "_s5_source_recovery_20260909_140603"
)
MANIFEST = json.loads(
    (MASTER / "target50_manifest.json").read_text(encoding="utf-8-sig")
)
PROTOCOL = json.loads(
    (MASTER / "protocol.json").read_text(encoding="utf-8-sig")
)
if PROTOCOL["manifest"] != MANIFEST:
    raise RuntimeError("Manifest differs from locked protocol")
PROTOCOL_SHA = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True).encode("utf-8")
).hexdigest()

if MANIFEST["split"] != "pretrain" or PROTOCOL["checkpoint_path"] != str(CHECKPOINT):
    raise RuntimeError("Evaluation split/checkpoint mismatch")
TASK_INDEX = int(os.environ["SLURM_ARRAY_TASK_ID"])
TASK = MANIFEST["tasks"][TASK_INDEX]
if TASK["task_index"] != TASK_INDEX:
    raise RuntimeError("Task index mismatch")

RUN = MASTER / "tasks" / ("task_%03d" % TASK_INDEX)
RUN.mkdir(parents=True, exist_ok=True)
EPISODES = RUN / "episodes"
EPISODES.mkdir(exist_ok=True)

def atomic_json(path, data):
    data = dict(data, split="pretrain", checkpoint_path=str(CHECKPOINT))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def existing_results():
    results = {}
    for path in sorted(EPISODES.glob("episode_*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        index = item["episode_index"]
        if not (
            isinstance(index, int)
            and 0 <= index < TASK["episode_count"]
            and index not in results
            and path.name == "episode_%03d.json" % index
            and item["protocol_sha256"] == PROTOCOL_SHA
            and item["task_index"] == TASK_INDEX
            and item["engineering_valid"] is True
            and item["episode_completed"] is True
            and isinstance(item["task_success"], bool)
        ):
            raise RuntimeError("Invalid existing episode: " + str(path))
        results[index] = item
    return results

def task_summary(results, error=None):
    summary = {
        "task_index": TASK_INDEX,
        "task_name": TASK["task_name"],
        "task_group": TASK["task_group"],
        "protocol_sha256": PROTOCOL_SHA,
        "expected_episodes": TASK["episode_count"],
        "completed_episodes": len(results),
        "successes": sum(int(x["task_success"]) for x in results.values()),
        "engineering_complete": len(results) == TASK["episode_count"],
        "last_job_id": JOB,
        "last_error": error,
    }
    atomic_json(RUN / "task_summary.json", summary)
    return summary

ENDPOINT = RUN / ("endpoint-" + JOB + ".json")

def save(name, value):
    (RUN / (name + "-" + JOB + ".json")).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

def log(message):
    print(message, flush=True)

def service_classes():
    return runpy.run_path(str(SOURCE / "gr00t/eval/service.py"))

def server(report):
    assert Path(sys.prefix).resolve() == POLICY.resolve()
    import torch
    assert torch.__version__ == "2.5.1+cu121"
    assert torch.cuda.is_available(), "Allocated GPU unavailable"
    torch.manual_seed(489 + TASK_INDEX)
    sys.path.insert(0, str(SOURCE))

    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    log("checking=load_policy")
    config = DATA_CONFIG_MAP["panda_omron"]
    with torch.no_grad():
        policy = Gr00tPolicy(
            model_path=str(CHECKPOINT),
            embodiment_tag="new_embodiment",
            modality_config=config.modality_config(),
            modality_transform=config.transform(),
            denoising_steps=4,
            device="cuda",
        )
    report["model_loaded"] = True
    report["gpu_name"] = torch.cuda.get_device_name(0)
    report["successful_requests"] = 0
    report["handler_errors"] = 0
    save("server-progress", report)

    BaseServer = service_classes()["BaseInferenceServer"]
    endpoint_server = BaseServer(host="127.0.0.1", port="*")

    def get_action(observation):
        started = time.monotonic()
        try:
            with torch.inference_mode():
                action = policy.get_action(observation)
            torch.cuda.synchronize()
            report["successful_requests"] += 1
            log("policy_request={} seconds={:.3f}".format(
                report["successful_requests"], time.monotonic() - started
            ))
            return action
        except Exception:
            report["handler_errors"] += 1
            raise

    endpoint_server.register_endpoint("get_action", get_action)
    try:
        import zmq
        address = endpoint_server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        ENDPOINT.write_text(json.dumps({
            "job_id": JOB,
            "host": "127.0.0.1",
            "port": int(address.rsplit(":", 1)[1]),
        }), encoding="utf-8")
        log("endpoint=" + address)
        endpoint_server.run()
        report["accepted"] = (
            report["successful_requests"] > 0 and report["handler_errors"] == 0
        )
        report["cuda_peak_allocated_gib"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        )
    finally:
        endpoint_server.socket.close(0)
        endpoint_server.context.term()


def simulation(report):
    import runpy
    from collections import deque
    import numpy as np
    import torch
    import zmq

    if Path(sys.prefix).resolve() != SIM.resolve():
        raise RuntimeError("Wrong simulation environment")
    if np.__version__ != "2.2.5":
        raise RuntimeError("Simulation NumPy version changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated GPU unavailable")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("Expected one numeric allocated GPU")
    render_device = int(visible)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_device)

    for name in ("robomimic", "robosuite", "robocasa"):
        sys.path.insert(0, str(THIRD / name))

    import gymnasium as gym
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    for group in ("atomic_seen", "composite_seen", "composite_unseen"):
        expected = {
            t["task_name"] for t in MANIFEST["tasks"]
            if t["task_group"] == group
        }
        if expected != set(TASK_SET_REGISTRY[group]):
            raise RuntimeError("Task registry differs from manifest: " + group)
    report["robocasa_version"] = getattr(robocasa, "__version__", "unknown")
    horizon = TASK["max_episode_steps"]
    if get_task_horizon(TASK["task_name"]) != horizon:
        raise RuntimeError("Runtime horizon differs from manifest")

    wrapper_path = SOURCE / "gr00t/eval/wrappers/multistep_wrapper.py"
    Wrapper = runpy.run_path(str(wrapper_path))["MultiStepWrapper"]
    classes = service_classes()
    Client = classes["BaseInferenceClient"]
    Serializer = classes["TorchSerializer"]

    results = existing_results()
    pending = deque(
        i for i in range(TASK["episode_count"]) if i not in results
    )
    live = []
    client = None

    report.update({
        "task_index": TASK_INDEX,
        "task_name": TASK["task_name"],
        "task_group": TASK["task_group"],
        "env_id": TASK["env_id"],
        "split": "pretrain",
        "protocol_sha256": PROTOCOL_SHA,
        "max_episode_steps": horizon,
        "n_action_steps": 16,
        "batch_size_limit": 5,
        "previously_completed_episodes": len(results),
        "newly_completed_episodes": 0,
        "successful_requests": 0,
        "rollout_run": False,
        "extra_action_clipping_applied": False,
        "extra_action_denormalization_applied": False,
    })

    def new_slot(index):
        seed = 489000 + TASK_INDEX * 1000 + index
        print(
            "creating_environment task=%s episode=%d seed=%d horizon=%d"
            % (TASK["task_name"], index, seed, horizon),
            flush=True,
        )
        base = gym.make(
            TASK["env_id"],
            split="pretrain",
            enable_render=True,
            seed=seed,
            render_gpu_device_id=render_device,
        )
        env = None
        try:
            env = Wrapper(
                base,
                video_delta_indices=np.asarray([0], dtype=np.int64),
                state_delta_indices=np.asarray([0], dtype=np.int64),
                n_action_steps=16,
                max_episode_steps=horizon,
            )
            obs, _ = env.reset()
            return {
                "index": index,
                "seed": seed,
                "env": env,
                "obs": obs,
                "success": False,
                "requests": 0,
                "outside_predicted": 0,
                "outside_executed": 0,
                "started": time.monotonic(),
                "closed": False,
            }
        except Exception:
            (env if env is not None else base).close()
            raise

    try:
        endpoint = json.loads(ENDPOINT.read_text(encoding="utf-8"))
        if str(endpoint["job_id"]) != JOB:
            raise RuntimeError("Stale policy endpoint")

        client = Client(
            host=endpoint["host"],
            port=endpoint["port"],
            timeout_ms=180000,
        )
        client.socket.setsockopt(zmq.RCVTIMEO, 180000)
        client.socket.setsockopt(zmq.SNDTIMEO, 180000)
        client.socket.setsockopt(zmq.LINGER, 0)
        task_summary(results)

        while pending or live:
            while pending and len(live) < 5:
                live.append(new_slot(pending.popleft()))

            keys = set(live[0]["obs"])
            if any(set(slot["obs"]) != keys for slot in live):
                raise RuntimeError("Observation keys differ across environments")

            batch = {}
            for key in sorted(keys):
                values = [slot["obs"][key] for slot in live]
                if key.startswith("annotation."):
                    if any(not isinstance(value, str) for value in values):
                        raise RuntimeError("Unexpected language observation type")
                    batch[key] = np.asarray(values)
                else:
                    batch[key] = np.stack(values, axis=0)
                    if key.startswith("state.") and not np.isfinite(batch[key]).all():
                        raise RuntimeError("Non-finite state observation: " + key)
                    if key.startswith("video.") and batch[key].dtype != np.uint8:
                        raise RuntimeError("Unexpected image dtype: " + key)

            actions = client.call_endpoint("get_action", batch)
            spaces = live[0]["env"].action_space.spaces
            if set(actions) != set(spaces):
                raise RuntimeError("Policy action keys do not match action space")

            for key, space in spaces.items():
                value = np.asarray(actions[key])
                expected = (len(live),) + tuple(space.shape)
                if (
                    value.shape != expected
                    or not np.issubdtype(value.dtype, np.floating)
                    or not np.isfinite(value).all()
                ):
                    raise RuntimeError(
                        "Invalid batched action %s: shape=%s expected=%s"
                        % (key, value.shape, expected)
                    )
                actions[key] = value

            report["successful_requests"] += 1
            finished_indices = set()

            for batch_index, slot in enumerate(live):
                env = slot["env"]
                action = {
                    key: value[batch_index] for key, value in actions.items()
                }
                outside = {
                    key: (value < spaces[key].low) | (value > spaces[key].high)
                    for key, value in action.items()
                }

                before = len(env.reward)
                obs, reward, terminated, truncated, info = env.step(action)
                after = len(env.reward)
                executed = after - before

                if not (0 < executed <= 16 and after <= horizon):
                    raise RuntimeError("Unexpected environment step count")

                slot["obs"] = obs
                slot["requests"] += 1
                slot["outside_predicted"] += sum(
                    int(mask.sum()) for mask in outside.values()
                )
                slot["outside_executed"] += sum(
                    int(mask[:executed].sum()) for mask in outside.values()
                )
                if "success" not in info:
                    raise RuntimeError("Official success field missing")
                success_value = np.asarray(info["success"]).reshape(-1)
                if success_value.size == 0:
                    raise RuntimeError("Official success field is empty")
                slot["success"] = slot["success"] or bool(success_value[0])
                report["rollout_run"] = True

                if bool(terminated) or bool(truncated):
                    env.close()
                    slot["closed"] = True
                    index = slot["index"]
                    item = {
                        "protocol_sha256": PROTOCOL_SHA,
                        "task_index": TASK_INDEX,
                        "task_name": TASK["task_name"],
                        "task_group": TASK["task_group"],
                        "episode_index": index,
                        "constructor_seed": slot["seed"],
                        "job_id": JOB,
                        "node": report["node"],
                        "engineering_valid": True,
                        "episode_completed": True,
                        "task_success": bool(slot["success"]),
                        "environment_closed": True,
                        "env_step_count": after,
                        "policy_request_count": slot["requests"],
                        "max_episode_steps": horizon,
                        "n_action_steps": 16,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "outside_predicted_bounds_count": slot["outside_predicted"],
                        "outside_executed_bounds_count": slot["outside_executed"],
                        "elapsed_seconds": round(
                            time.monotonic() - slot["started"], 3
                        ),
                        "extra_action_clipping_applied": False,
                        "extra_action_denormalization_applied": False,
                    }
                    output = EPISODES / ("episode_%03d.json" % index)
                    if output.exists():
                        raise RuntimeError("Episode result already exists")
                    atomic_json(output, item)
                    results[index] = item
                    finished_indices.add(index)
                    report["newly_completed_episodes"] += 1
                    summary = task_summary(results)
                    print(
                        "EPISODE_DONE task=%s episode=%d success=%s "
                        "steps=%d valid=%d/50 successes=%d"
                        % (
                            TASK["task_name"], index, item["task_success"],
                            after, len(results), summary["successes"],
                        ),
                        flush=True,
                    )

            live = [
                slot for slot in live if slot["index"] not in finished_indices
            ]
            if report["successful_requests"] % 10 == 0:
                print(
                    "progress task=%s batch_requests=%d valid=%d/50 live_steps=%s"
                    % (
                        TASK["task_name"],
                        report["successful_requests"],
                        len(results),
                        [len(slot["env"].reward) for slot in live],
                    ),
                    flush=True,
                )

        if len(results) != TASK["episode_count"]:
            raise RuntimeError("Incomplete task")
        report["accepted"] = True
        report["closed_loop_accepted"] = True
        report["task_summary"] = task_summary(results)

    except Exception as exc:
        task_summary(results, str(exc))
        raise
    finally:
        for slot in live:
            if not slot["closed"]:
                try:
                    slot["env"].close()
                except Exception:
                    traceback.print_exc()
        if client is not None:
            try:
                client.socket.setsockopt(zmq.RCVTIMEO, 5000)
                client.socket.setsockopt(zmq.SNDTIMEO, 5000)
                client.socket.send(Serializer.to_bytes({"endpoint": "kill"}))
                Serializer.from_bytes(client.socket.recv())
            except Exception as exc:
                report["server_shutdown_message"] = str(exc)
            finally:
                client.socket.close(0)
                client.context.term()



def orchestrate(report):
    import fcntl

    lock_stream = (RUN / "task.lock").open("a")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_stream.close()
        raise RuntimeError("Another worker is already running this task")

    processes = []
    streams = []
    try:
        results = existing_results()
        if len(results) == TASK["episode_count"]:
            report.update({
                "accepted": True,
                "task_complete": True,
                "already_complete": True,
                "task_summary": task_summary(results),
            })
            return

        task_summary(results)

        def launch(role, python):
            path = RUN / (role + "-" + JOB + ".log")
            stream = path.open("w", encoding="utf-8")
            streams.append((stream, path))
            process = subprocess.Popen(
                [
                    str(python), "-I", "-B", "-u",
                    str(MASTER / "worker.py"), role,
                ],
                cwd=RUN,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            print("started=%s log=%s" % (role, path), flush=True)
            return process

        policy_process = launch("server", POLICY / "bin/python")
        deadline = time.monotonic() + 600
        ready = False
        while time.monotonic() < deadline:
            if policy_process.poll() is not None:
                raise RuntimeError("Policy server exited before ready")
            if ENDPOINT.exists():
                try:
                    endpoint = json.loads(
                        ENDPOINT.read_text(encoding="utf-8")
                    )
                    ready = str(endpoint["job_id"]) == JOB
                except (ValueError, KeyError, OSError):
                    ready = False
                if ready:
                    break
            time.sleep(1)
        if not ready:
            raise RuntimeError("Policy server readiness timed out")

        print("policy_ready=True", flush=True)
        simulation_process = launch("simulation", SIM / "bin/python")
        simulation_rc = simulation_process.wait(timeout=25200)
        report["simulation_rc"] = simulation_rc
        if simulation_rc != 0:
            raise RuntimeError("Simulation failed; inspect simulation log")

        server_rc = policy_process.wait(timeout=30)
        report["server_rc"] = server_rc
        if server_rc != 0:
            raise RuntimeError("Policy server exited with an error")

        sim_report = json.loads(
            (RUN / ("simulation-report-" + JOB + ".json")).read_text(
                encoding="utf-8"
            )
        )
        server_report = json.loads(
            (RUN / ("server-report-" + JOB + ".json")).read_text(
                encoding="utf-8"
            )
        )
        if not sim_report["accepted"] or not server_report["accepted"]:
            raise RuntimeError("A component report was not accepted")
        if sim_report["successful_requests"] != server_report["successful_requests"]:
            raise RuntimeError("Policy request counts do not match")

        results = existing_results()
        if len(results) != TASK["episode_count"]:
            raise RuntimeError("Expected 50 valid episode records")

        report.update({
            "accepted": True,
            "task_complete": True,
            "closed_loop_accepted": True,
            "model_loaded": True,
            "rollout_run": True,
            "task_index": TASK_INDEX,
            "task_name": TASK["task_name"],
            "task_summary": task_summary(results),
        })

    except Exception as exc:
        task_summary(existing_results(), str(exc))
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                except ProcessLookupError:
                    pass
        for stream, path in streams:
            stream.close()
            print("##### COMPONENT_LOG_TAIL: %s #####" % path.name, flush=True)
            try:
                with path.open("rb") as reader:
                    reader.seek(0, 2)
                    size = reader.tell()
                    reader.seek(max(0, size - 16000))
                    print(reader.read().decode("utf-8", errors="replace"),
                          flush=True)
            except OSError as exc:
                print(str(exc), flush=True)
        lock_stream.close()


report = {
    "job_id": JOB,
    "node": socket.gethostname(),
    "role": ROLE,
    "accepted": False,
    "closed_loop_accepted": False,
    "model_loaded": False,
    "rollout_run": False,
    "training_run": False,
    "packages_installed": False,
    "formal_target50_evaluation": True,
    "split": "pretrain",
    "checkpoint_path": str(CHECKPOINT),
    "protocol_sha256": PROTOCOL_SHA,
}
try:
    {"server": server, "simulation": simulation, "orchestrate": orchestrate}[ROLE](report)
except Exception as exc:
    report["accepted"] = False
    report["error_type"] = type(exc).__name__
    report["error"] = str(exc)
    traceback.print_exc()

name = "report" if ROLE == "orchestrate" else ROLE + "-report"
save(name, report)
log("##### COMPACT_REPORT_BEGIN: M489_S30_" + ROLE.upper() + " #####")
log(json.dumps(report, ensure_ascii=False, indent=2))
log("##### COMPACT_REPORT_END: M489_S30_" + ROLE.upper() + " #####")
raise SystemExit(0 if report["accepted"] else 1)
