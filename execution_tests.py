import subprocess
import sys
from pathlib import Path

PYTHON_EXEC = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "main_mpo.py"

runs = [
    {
        "name": "03_inverted_without_task_hint_no_penalty_tanh_default_rand",
        "use_action_penalty": False,
        "use_tanh_on_mean": True,
        "task_mode": "inverted_without_task_hint",
        "rand_mode": "default",
    },
    {
        "name": "05_default_no_penalty_tanh_rfi",
        "use_action_penalty": False,
        "use_tanh_on_mean": True,
        "task_mode": "default",
        "rand_mode": "RFI",
    },
    {
        "name": "06_default_no_penalty_tanh_rao",
        "use_action_penalty": False,
        "use_tanh_on_mean": True,
        "task_mode": "default",
        "rand_mode": "RAO",
    },
]

def to_cli_args(run_cfg: dict) -> list[str]:
    args = []

    for key, value in run_cfg.items():
        if key == "name":
            continue

        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            else:
                args.append(f"--no-{key}")
        else:
            args.extend([f"--{key}", str(value)])

    return args

def main():
    if not TRAIN_SCRIPT.exists():
        print(f"Fehler: Trainingsskript nicht gefunden: {TRAIN_SCRIPT}")
        sys.exit(1)

    results = []

    for i, run in enumerate(runs, start=1):
        run_name = run["name"]

        cmd = [
            PYTHON_EXEC,
            str(TRAIN_SCRIPT),
            "--exp_name", run_name,
            "--log_dir", f"mpo_logs/{run_name}",
            *to_cli_args(run),
        ]

        print("\n" + "=" * 100)
        print(f"Starte Run {i}/{len(runs)}: {run_name}")
        print("Command:")
        print(" ".join(cmd))
        print("=" * 100 + "\n")

        try:
            completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            results.append((run_name, "SUCCESS", completed.returncode))
            print(f"\nRun erfolgreich: {run_name}\n")
        except subprocess.CalledProcessError as e:
            results.append((run_name, "FAILED", e.returncode))
            print(f"\nRun fehlgeschlagen: {run_name} (return code {e.returncode})")
            print("Der nächste Run wird trotzdem gestartet.\n")
        except Exception as e:
            results.append((run_name, "FAILED", str(e)))
            print(f"\nUnerwarteter Fehler bei Run {run_name}: {e}")
            print("Der nächste Run wird trotzdem gestartet.\n")

    print("\n" + "#" * 100)
    print("Sweep abgeschlossen. Zusammenfassung:\n")
    for run_name, status, info in results:
        print(f"{status:8} | {run_name} | {info}")
    print("#" * 100 + "\n")

if __name__ == "__main__":
    main()