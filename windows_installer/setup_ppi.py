"""
PPI Analyser - Setup GUI
Supporte deux backends :
  • native  — Podman Desktop + podman machine (virtualisation dispo)
  • wsl     — WSL2 + Podman dans WSL (pas de virtualisation / Hyper-V absent)
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
import tempfile

# ─────────────────────────────────────────────────────────────────────────────
#  BACKEND  ("native" | "wsl")
#  Déterminé une seule fois dans _step_detect_backend(), puis figé.
# ─────────────────────────────────────────────────────────────────────────────
BACKEND = "native"   # valeur par défaut avant détection


def _wsl(cmd):
    """Préfixe une commande pour l'exécuter dans WSL."""
    # On passe via cmd /c pour gérer les quotes Windows
    return f'wsl bash -lc "{cmd}"'


def run(cmd, **kwargs):
    """Lance une commande et retourne (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, **kwargs
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# ─────────────────────────────────────────────
#  Détection backend
# ─────────────────────────────────────────────

def _virt_available():
    """Renvoie True si Hyper-V / virtualisation est accessible pour podman machine."""
    # podman machine init fait appel à WSL2 ou HyperV ; on teste en vérifiant
    # que wsl --status ne remonte pas d'erreur de virtualisation ET que
    # "Virtual Machine Platform" est activé.
    code, out, _ = run("wsl --status")
    if code != 0:
        return False
    # Si WSL indique lui-même que la virtualisation est absente
    if "virtualization" in out.lower() and "not" in out.lower():
        return False
    # Vérification rapide via systeminfo (lent mais fiable)
    code2, out2, _ = run(
        'powershell -NoProfile -Command '
        '"(Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform).State"'
    )
    return code2 == 0 and "enabled" in out2.lower()


def _wsl_available():
    code, _, _ = run("wsl --version")
    return code == 0


def _wsl_podman_available():
    code, _, _ = run(_wsl("podman --version"))
    return code == 0


# ─────────────────────────────────────────────
#  Helpers (adaptés au backend)
# ─────────────────────────────────────────────

def podman_available():
    if BACKEND == "wsl":
        return _wsl_podman_available()
    code, _, _ = run("podman --version")
    return code == 0


def compose_dir():
    return os.path.dirname(os.path.abspath(__file__))


def compose_file():
    return os.path.join(compose_dir(), "docker-compose.yml")


def ensure_compose_file():
    if not os.path.isfile(compose_file()):
        raise StepError(
            "docker-compose.yml introuvable à côté de ce script "
            f"({compose_dir()}). Placez-le dans le même dossier."
        )


def _wsl_path(win_path):
    """Convertit un chemin Windows en chemin WSL (/mnt/c/...)."""
    drive, rest = os.path.splitdrive(win_path)
    letter = drive.rstrip(":").lower()
    rest_unix = rest.replace("\\", "/")
    return f"/mnt/{letter}{rest_unix}"


def compose_run(args, **kwargs):
    if BACKEND == "wsl":
        wsl_dir = _wsl_path(compose_dir())
        cmd = _wsl(f"cd '{wsl_dir}' && podman compose {args}")
    else:
        cmd = f"podman compose {args}"
    return run(cmd, cwd=compose_dir() if BACKEND == "native" else None, **kwargs)


def compose_available():
    code, _, _ = compose_run("version")
    return code == 0


def list_compose_images():
    code, out, _ = compose_run("config --images")
    if code == 0 and out.strip():
        return [line.strip() for line in out.splitlines() if line.strip()]
    return []


def get_host_port(container_name="ppi_analyser"):
    if BACKEND == "wsl":
        code, out, _ = run(_wsl(f"podman port {container_name}"))
    else:
        code, out, _ = run(f"podman port {container_name}")
    if code != 0 or not out.strip():
        return None
    m = re.search(r":(\d+)\s*$", out.splitlines()[0].strip())
    return m.group(1) if m else None


def vm_exists():
    if BACKEND == "wsl":
        return True   # pas de VM dans le mode WSL
    code, out, _ = run("podman machine list --format {{.Name}}")
    return "podman-machine-default" in out


def vm_running():
    if BACKEND == "wsl":
        return True   # pas de VM dans le mode WSL
    code, out, _ = run("podman machine list --format {{.LastUp}}")
    return "Currently running" in out


def container_running():
    if BACKEND == "wsl":
        code, out, _ = run(_wsl('podman ps --filter "name=ppi_analyser" --format {{.Names}}'))
    else:
        code, out, _ = run('podman ps --filter "name=ppi_analyser" --format {{.Names}}')
    return "ppi_analyser" in out


def app_ready(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────

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
        self.geometry("860x540")

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

        tk.Label(
            left, text="PPI Analyser", font=("Segoe UI", 15, "bold"),
            fg="#cdd6f4", bg="#181825"
        ).pack(anchor="w")
        tk.Label(
            left, text="Installation automatique",
            font=("Segoe UI", 8), fg="#585b70", bg="#181825"
        ).pack(anchor="w", pady=(0, 14))

        # Badge backend (mis à jour après détection)
        self.backend_badge = tk.Label(
            left, text="Backend : détection…",
            font=("Segoe UI", 8, "italic"), fg="#fab387", bg="#181825"
        )
        self.backend_badge.pack(anchor="w", pady=(0, 8))

        self.step_labels = {}
        self.step_icons  = {}
        steps = [
            ("python",   "Python"),
            ("backend",  "Détection backend"),
            ("podman",   "Podman"),
            ("wsl",      "WSL2 (si nécessaire)"),
            ("vm_init",  "VM  (init)"),
            ("vm_start", "VM  (démarrage)"),
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
            cursor="hand2", pady=8,
            command=self._start
        )
        self.btn.pack(fill="x")

        # ── Colonne droite : log ──────────────────────────────
        right = tk.Frame(root_frame, bg="#181825")
        right.grid(row=0, column=1, sticky="nsew")

        tk.Label(
            right, text="Journal d'installation",
            font=("Segoe UI", 8, "bold"), fg="#585b70", bg="#181825",
            anchor="w"
        ).pack(fill="x", padx=10, pady=(10, 4))

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
        icon, color = icons[status], colors[status]
        self.step_icons[key].config(text=icon, fg=color)
        self.step_labels[key].config(fg=color if status != "pending" else "#a6adc8")

    def _advance(self):
        self.progress["value"] += 1

    # ── Main thread ───────────────────────────

    def _start(self):
        self.btn.config(state="disabled", text="Installation en cours…")
        threading.Thread(target=self._run_setup, daemon=True).start()

    def _run_setup(self):
        steps = [
            ("python",   self._step_python),
            ("backend",  self._step_detect_backend),
            ("podman",   self._step_podman),
            ("wsl",      self._step_wsl),
            ("vm_init",  self._step_vm_init),
            ("vm_start", self._step_vm_start),
            ("pull",     self._step_pull),
            ("run",      self._step_run),
            ("ready",    self._step_ready),
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

    def _step_detect_backend(self):
        global BACKEND
        self._log("Détection du backend conteneur…")

        if _virt_available():
            BACKEND = "native"
            self._log("  ✔ Virtualisation disponible → backend Podman natif", "#a6e3a1")
        elif _wsl_available():
            BACKEND = "wsl"
            self._log("  ⚠ Virtualisation absente, WSL2 détecté → backend Podman-in-WSL", "#fab387")
        else:
            raise StepError(
                "Ni la virtualisation ni WSL2 ne sont disponibles.\n"
                "Activez 'Virtual Machine Platform' ou WSL2 dans les fonctionnalités Windows, "
                "puis redémarrez."
            )

        self.after(0, lambda: self.backend_badge.config(
            text=f"Backend : {'Podman natif' if BACKEND == 'native' else 'Podman via WSL2'}",
            fg="#a6e3a1" if BACKEND == "native" else "#fab387"
        ))

    def _step_wsl(self):
        """Installe / configure Podman dans WSL si backend=wsl."""
        if BACKEND != "wsl":
            raise StepSkipped("Non nécessaire (backend natif)")

        self._log("Vérification de Podman dans WSL…")
        if _wsl_podman_available():
            code, out, _ = run(_wsl("podman --version"))
            raise StepSkipped(f"Podman déjà présent dans WSL ({out})")

        self._log("Installation de Podman dans WSL (apt)…", "#fab387")
        cmds = [
            "sudo apt-get update -qq",
            "sudo apt-get install -y podman",
        ]
        for cmd in cmds:
            code, _, err = run(_wsl(cmd))
            if code != 0:
                raise StepError(f"Échec dans WSL : {err}")

        # podman-compose
        self._log("Installation de podman-compose dans WSL…", "#fab387")
        code, _, err = run(_wsl("pip3 install --user podman-compose -q || "
                                "sudo apt-get install -y podman-compose"))
        if code != 0:
            raise StepError(f"Échec podman-compose : {err}")

        if not _wsl_podman_available():
            raise StepError("Podman installé dans WSL mais introuvable — redémarre et relance.")
        self._log("Podman installé dans WSL.", "#a6e3a1")

    def _step_podman(self):
        """Installe Podman Desktop (backend natif uniquement)."""
        if BACKEND == "wsl":
            raise StepSkipped("Non nécessaire (backend WSL)")

        self._log("Vérification de Podman…")
        if podman_available():
            code, out, _ = run("podman --version")
            raise StepSkipped(f"Podman déjà installé ({out})")

        self._log("Téléchargement de Podman (~200 MB)…", "#fab387")
        url = "https://github.com/containers/podman/releases/latest/download/podman-setup.exe"
        dest = os.path.join(tempfile.gettempdir(), "podman-setup.exe")

        def _progress(count, block, total):
            if total > 0:
                pct = min(int(count * block * 100 / total), 100)
                self._log(f"  … {pct}%", "#585b70") if pct % 20 == 0 else None

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        self._log("Installation de Podman…", "#fab387")
        code, _, err = run(f'"{dest}" /quiet')
        if code != 0:
            raise StepError(f"Échec installation Podman : {err}")
        os.environ["PATH"] += (
            r";C:\Program Files\RedHat\Podman"
            r";C:\Program Files (x86)\RedHat\Podman"
        )
        if not podman_available():
            raise StepError("Podman installé mais introuvable dans PATH — redémarre et relance.")
        self._log("Podman installé.", "#a6e3a1")

    def _step_vm_init(self):
        if BACKEND == "wsl":
            raise StepSkipped("Pas de VM Podman en mode WSL")
        self._log("Vérification de la VM Podman…")
        if vm_exists():
            raise StepSkipped("VM déjà existante")
        self._log("Initialisation de la VM (~700 MB, patientez…)", "#fab387")
        code, _, err = run("podman machine init")
        if code != 0:
            raise StepError(f"Échec init VM : {err}")

    def _step_vm_start(self):
        if BACKEND == "wsl":
            raise StepSkipped("Pas de VM Podman en mode WSL")
        self._log("Démarrage de la VM…")
        if vm_running():
            raise StepSkipped("VM déjà en cours d'exécution")
        code, _, err = run("podman machine start")
        if code == 0:
            time.sleep(3)
            return
        # Erreur connue Windows : "key already exists" (32773)
        # La VM est dans un état corrompu → reset et retry
        if "key already exists" in err or "32773" in err:
            self._log("  ⚠ VM corrompue (key already exists), réinitialisation…", "#fab387")
            run("podman machine stop")
            run("podman machine rm -f")
            time.sleep(2)
            self._log("  Recréation de la VM…", "#fab387")
            code2, _, err2 = run("podman machine init")
            if code2 != 0:
                raise StepError(f"Échec re-init VM : {err2}")
            code3, _, err3 = run("podman machine start")
            if code3 != 0:
                raise StepError(f"Échec démarrage VM après reset : {err3}")
            time.sleep(3)
            self._log("  VM redémarrée après reset.", "#a6e3a1")
        else:
            raise StepError(f"Échec démarrage VM : {err}")

    def _step_pull(self):
        ensure_compose_file()
        self._log("Vérification de docker-compose.yml…")
        if not compose_available():
            raise StepError(
                "podman compose indisponible. "
                + ("Dans WSL : sudo apt install podman-compose"
                   if BACKEND == "wsl"
                   else "pip install podman-compose ou utilisez une version récente de Podman")
                + ", puis relancez."
            )
        images = list_compose_images()
        if images:
            self._log(f"  Image(s) définie(s) : {', '.join(images)}", "#585b70")
        self._log("Téléchargement / mise à jour de l'image…", "#fab387")
        code, _, err = compose_run("pull")
        if code != 0:
            raise StepError(f"Échec pull image : {err}")

    def _step_run(self):
        self._log("Lancement du conteneur via docker-compose…")
        if container_running():
            self._log("  Conteneur déjà actif, redémarrage…", "#fab387")
        code, _, err = compose_run("up -d --force-recreate")
        if code != 0:
            raise StepError(f"Échec lancement conteneur : {err}")
        host_port = get_host_port()
        if not host_port:
            raise StepError("Conteneur lancé mais port publié introuvable (podman port).")
        self.app_url = f"http://localhost:{host_port}"
        self._log(f"  Application servie sur {self.app_url}", "#89b4fa")

    def _step_ready(self):
        self._log(f"Attente de l'application sur {self.app_url}…")
        for i in range(20):
            if app_ready(self.app_url):
                self._log("Application prête !", "#a6e3a1")
                return
            time.sleep(3)
            self._log(f"  … {(i+1)*3}s", "#585b70")
        self._log("L'application démarre encore (modèles NLP). Ouvre le navigateur dans 1-2 min.", "#fab387")

    # ── Finish ───────────────────────────────

    def _finish(self, success):
        if success:
            self._log("\n✔ Installation terminée !", "#a6e3a1")
            self.btn.config(
                state="normal", text="🌐  Ouvrir PPI Analyser",
                bg="#a6e3a1", command=lambda: webbrowser.open(self.app_url)
            )
        else:
            self._log("\n✘ Installation interrompue. Voir les erreurs ci-dessus.", "#f38ba8")
            self.btn.config(
                state="normal", text="↺  Réessayer",
                bg="#f38ba8", command=self._restart
            )

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
