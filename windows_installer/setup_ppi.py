"""
PPI Analyser - Setup GUI
Backend fixe : WSL2 + Podman dans WSL (pas de Podman Desktop, pas de VM Hyper-V)
Requires: Python 3.8+ (stdlib only, tkinter inclus)
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import subprocess
import urllib.request
import os
import re
import sys
import time
import webbrowser


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, **kwargs
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def wsl(cmd):
    """Exécute une commande bash dans WSL."""
    escaped = cmd.replace('"', '\\"')
    return run(f'wsl bash -lc "{escaped}"')


def wsl_path(win_path):
    """Convertit un chemin Windows en chemin WSL (/mnt/c/...)."""
    drive, rest = os.path.splitdrive(win_path)
    return f"/mnt/{drive.rstrip(':').lower()}{rest.replace(chr(92), '/')}"


def compose_dir():
    return os.path.dirname(os.path.abspath(__file__))


def compose_file():
    return os.path.join(compose_dir(), "docker-compose.yml")


def ensure_compose_file():
    if not os.path.isfile(compose_file()):
        raise StepError(
            f"docker-compose.yml introuvable dans {compose_dir()}. "
            "Placez-le dans le même dossier que ce script."
        )


def compose_run(args):
    wsl_dir = wsl_path(compose_dir())
    return wsl(f"cd '{wsl_dir}' && podman compose {args}")


def wsl_available():
    code, _, _ = run("wsl --version")
    return code == 0


def wsl_podman_available():
    code, _, _ = wsl("podman --version")
    return code == 0


def compose_available():
    code, _, _ = compose_run("version")
    return code == 0


def list_compose_images():
    code, out, _ = compose_run("config --images")
    if code == 0 and out.strip():
        return [l.strip() for l in out.splitlines() if l.strip()]
    return []


def get_host_port(container="ppi_analyser"):
    code, out, _ = wsl(f"podman port {container}")
    if code != 0 or not out.strip():
        return None
    m = re.search(r":(\d+)\s*$", out.splitlines()[0].strip())
    return m.group(1) if m else None


def container_running():
    code, out, _ = wsl('podman ps --filter "name=ppi_analyser" --format {{.Names}}')
    return "ppi_analyser" in out


def app_ready(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────

class SetupApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PPI Analyser — Installation")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")
        self.app_url = None
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self.geometry("860x500")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor="#313244",
                        background="#89b4fa", thickness=8)

        root_frame = tk.Frame(self, bg="#1e1e2e")
        root_frame.pack(fill="both", expand=True, padx=16, pady=16)
        root_frame.columnconfigure(0, weight=0, minsize=260)
        root_frame.columnconfigure(1, weight=1)
        root_frame.rowconfigure(0, weight=1)

        # ── Colonne gauche ────────────────────────────────────
        left = tk.Frame(root_frame, bg="#181825", padx=16, pady=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        tk.Label(left, text="PPI Analyser", font=("Segoe UI", 15, "bold"),
                 fg="#cdd6f4", bg="#181825").pack(anchor="w")
        tk.Label(left, text="Installation automatique",
                 font=("Segoe UI", 8), fg="#585b70", bg="#181825").pack(anchor="w", pady=(0, 4))
        tk.Label(left, text="Backend : Podman via WSL2",
                 font=("Segoe UI", 8, "italic"), fg="#89b4fa", bg="#181825").pack(anchor="w", pady=(0, 14))

        self.step_labels = {}
        self.step_icons  = {}
        steps = [
            ("python",   "Python"),
            ("wsl",      "WSL2"),
            ("podman",   "Podman (dans WSL)"),
            ("compose",  "podman-compose"),
            ("pull",     "Image Docker"),
            ("run",      "Conteneur"),
            ("ready",    "Application prête"),
        ]
        for key, label in steps:
            row = tk.Frame(left, bg="#181825")
            row.pack(fill="x", pady=2)
            icon = tk.Label(row, text="○", font=("Segoe UI", 11),
                            fg="#585b70", bg="#181825", width=2)
            icon.pack(side="left")
            lbl = tk.Label(row, text=label, font=("Segoe UI", 9),
                           fg="#a6adc8", bg="#181825", anchor="w")
            lbl.pack(side="left")
            self.step_icons[key]  = icon
            self.step_labels[key] = lbl

        self.progress = ttk.Progressbar(left, length=220, mode="determinate",
                                        maximum=len(steps))
        self.progress.pack(fill="x", pady=(14, 16))

        self.btn = tk.Button(
            left, text="▶  Démarrer",
            font=("Segoe UI", 10, "bold"),
            bg="#89b4fa", fg="#1e1e2e", relief="flat",
            activebackground="#74c7ec", activeforeground="#1e1e2e",
            cursor="hand2", pady=8, command=self._start
        )
        self.btn.pack(fill="x")

        # ── Colonne droite : log ──────────────────────────────
        right = tk.Frame(root_frame, bg="#181825")
        right.grid(row=0, column=1, sticky="nsew")

        tk.Label(right, text="Journal d'installation",
                 font=("Segoe UI", 8, "bold"), fg="#585b70", bg="#181825",
                 anchor="w").pack(fill="x", padx=10, pady=(10, 4))

        self.log = scrolledtext.ScrolledText(
            right, state="disabled",
            bg="#181825", fg="#cdd6f4", font=("Consolas", 9),
            insertbackground="white", relief="flat", bd=0
        )
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.eval('tk::PlaceWindow . center')

    # ── Logging ──────────────────────────────

    def _log(self, msg, color="#cdd6f4"):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n", color)
        self.log.tag_configure(color, foreground=color)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_step(self, key, status):
        icons  = {"pending":"○","running":"◉","ok":"✔","skip":"—","error":"✘"}
        colors = {"pending":"#585b70","running":"#fab387","ok":"#a6e3a1",
                  "skip":"#585b70","error":"#f38ba8"}
        self.step_icons[key].config(text=icons[status], fg=colors[status])
        self.step_labels[key].config(fg=colors[status] if status != "pending" else "#a6adc8")

    def _advance(self):
        self.progress["value"] += 1

    # ── Main thread ───────────────────────────

    def _start(self):
        self.btn.config(state="disabled", text="Installation en cours…")
        threading.Thread(target=self._run_setup, daemon=True).start()

    def _run_setup(self):
        steps = [
            ("python",  self._step_python),
            ("wsl",     self._step_wsl),
            ("podman",  self._step_podman),
            ("compose", self._step_compose),
            ("pull",    self._step_pull),
            ("run",     self._step_run),
            ("ready",   self._step_ready),
        ]
        for key, fn in steps:
            self._set_step(key, "running")
            try:
                fn()
                self._set_step(key, "ok")
            except StepSkipped as e:
                self._set_step(key, "skip")
                self._log(f"  ↷ {e}", "#585b70")
            except StepError as e:
                self._set_step(key, "error")
                self._log(f"  ✘ {e}", "#f38ba8")
                self._finish(success=False)
                return
            self._advance()
        self._finish(success=True)

    # ── Steps ────────────────────────────────

    def _step_python(self):
        self._log("Vérification de Python…")
        code, out, _ = run("python --version")
        if code == 0:
            raise StepSkipped(f"Python déjà installé ({out})")
        raise StepError("Python introuvable — relance via LANCER_PPI.bat")

    def _step_wsl(self):
        self._log("Vérification de WSL2…")
        if wsl_available():
            code, out, _ = run("wsl --version")
            # Extrait juste la 1re ligne (version WSL)
            version_line = out.splitlines()[0] if out else "WSL disponible"
            raise StepSkipped(version_line)

        self._log("Installation de WSL2…", "#fab387")
        self._log("  (nécessite un redémarrage Windows)", "#fab387")
        code, _, err = run("wsl --install --no-distribution")
        if code != 0:
            raise StepError(
                f"Échec installation WSL2 : {err}\n"
                "Activez manuellement : Panneau de configuration → "
                "Fonctionnalités Windows → Sous-système Windows pour Linux"
            )
        raise StepError(
            "WSL2 installé — REDÉMARRE Windows puis relance ce script."
        )

    def _step_podman(self):
        self._log("Vérification de Podman dans WSL…")
        if wsl_podman_available():
            _, out, _ = wsl("podman --version")
            raise StepSkipped(f"Podman déjà présent ({out})")

        self._log("Installation de Podman dans WSL (apt)…", "#fab387")
        steps_apt = [
            "sudo apt-get update -qq",
            "sudo apt-get install -y podman",
        ]
        for cmd in steps_apt:
            self._log(f"  $ {cmd}", "#585b70")
            code, _, err = wsl(cmd)
            if code != 0:
                raise StepError(f"Échec : {err}")

        if not wsl_podman_available():
            raise StepError("Podman installé mais introuvable — redémarre et relance.")
        _, out, _ = wsl("podman --version")
        self._log(f"  Podman installé ({out})", "#a6e3a1")

    def _step_compose(self):
        self._log("Vérification de podman-compose dans WSL…")
        code, out, _ = wsl("podman compose version")
        if code == 0:
            raise StepSkipped(f"podman-compose déjà présent ({out.splitlines()[0]})")

        self._log("Installation de podman-compose…", "#fab387")
        # Essai 1 : apt
        code, _, _ = wsl("sudo apt-get install -y podman-compose")
        if code != 0:
            # Essai 2 : pip
            code, _, err = wsl("pip3 install --user podman-compose")
            if code != 0:
                raise StepError(f"Échec installation podman-compose : {err}")
        code2, out2, _ = wsl("podman compose version")
        if code2 != 0:
            raise StepError("podman-compose installé mais introuvable — redémarre et relance.")
        self._log(f"  podman-compose prêt ({out2.splitlines()[0]})", "#a6e3a1")

    def _step_pull(self):
        ensure_compose_file()
        self._log("Vérification de docker-compose.yml…")
        images = list_compose_images()
        if images:
            self._log(f"  Image(s) : {', '.join(images)}", "#585b70")
        self._log("Téléchargement / mise à jour de l'image…", "#fab387")
        code, _, err = compose_run("pull")
        if code != 0:
            raise StepError(f"Échec pull image : {err}")

    def _step_run(self):
        self._log("Lancement du conteneur…")
        if container_running():
            self._log("  Conteneur déjà actif, redémarrage…", "#fab387")
        code, _, err = compose_run("up -d --force-recreate")
        if code != 0:
            raise StepError(f"Échec lancement conteneur : {err}")
        host_port = get_host_port()
        if not host_port:
            raise StepError("Conteneur lancé mais port introuvable (podman port).")
        self.app_url = f"http://localhost:{host_port}"
        self._log(f"  Application sur {self.app_url}", "#89b4fa")

    def _step_ready(self):
        self._log(f"Attente de l'application…")
        for i in range(20):
            if app_ready(self.app_url):
                self._log("Application prête !", "#a6e3a1")
                return
            time.sleep(3)
            self._log(f"  … {(i+1)*3}s", "#585b70")
        self._log("Démarre encore (modèles NLP). Ouvre le navigateur dans 1-2 min.", "#fab387")

    # ── Finish ───────────────────────────────

    def _finish(self, success):
        if success:
            self._log("\n✔ Installation terminée !", "#a6e3a1")
            self.btn.config(state="normal", text="🌐  Ouvrir PPI Analyser",
                            bg="#a6e3a1", command=lambda: webbrowser.open(self.app_url))
        else:
            self._log("\n✘ Installation interrompue. Voir les erreurs ci-dessus.", "#f38ba8")
            self.btn.config(state="normal", text="↺  Réessayer",
                            bg="#f38ba8", command=self._restart)

    def _restart(self):
        self.btn.config(state="disabled", text="Installation en cours…", bg="#89b4fa")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.progress["value"] = 0
        for key in self.step_icons:
            self._set_step(key, "pending")
        threading.Thread(target=self._run_setup, daemon=True).start()

    def _on_close(self):
        if container_running():
            self._log("\nArrêt du conteneur…", "#fab387")
            self.btn.config(state="disabled", text="Arrêt en cours…", bg="#f38ba8")
            threading.Thread(target=self._shutdown, daemon=False).start()
        else:
            self.destroy()
            sys.exit(0)

    def _shutdown(self):
        compose_run("down")
        self.after(0, self._quit_now)

    def _quit_now(self):
        self.destroy()
        sys.exit(0)


class StepSkipped(Exception):
    pass

class StepError(Exception):
    pass


if __name__ == "__main__":
    app = SetupApp()
    app.mainloop()
