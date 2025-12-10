# Configure and understand tmux
_Exported on 12/10/2025 at 02:58:55 GMT from Cursor (2.1.50)_

---

**User**

i have a chat from another window i exported, if you could read it first then come back to this part and my problem should make sense. i'm not sure howt to use tmux, these commands don't seem to work probably beacaus i'm making a mistake somewhere. colud you set this multiplex editor system up and explain what you're doing so i understand. here's the chatlog from the prenious window: "D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\misc\cursor_optimize_multi_pc_ai_setup_with.md"

---

**Cursor**

Here’s a concise setup you can follow to get tmux running for your worktrees without extra IDE windows. I’ll assume you’re SSH’ing into the “server” (paths like `/mnt/d/Users/dilli/...` suggest WSL or a Linux shell on that box).

## One-time check/install
- Verify tmux is present: `tmux -V` (install via your package manager if missing: `sudo apt-get install tmux` on WSL/Ubuntu).

## Start/attach the shared session
- From any SSH session on the client → server: `tmux new -A -s plasma`
  - `-A` attaches if it exists, creates if not.

## Create windows for each worktree (once per session)
Inside tmux:
- New window: `Ctrl-b c`
- Change to the worktree dir and rename:
  - `cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender`
  - `Ctrl-b ,` → type `blender`
- Repeat for others:
  - `cd .../PlasmaDX-PINN-v4`, rename `pinn`
  - `cd .../PlasmaDX-MultiAgent`, rename `multi`
  - `cd .../PlasmaDX-Test`, rename `test`
- You now have 4 windows, each anchored to a worktree.

## Everyday flow
1) SSH in from the client: `tmux new -A -s plasma`
2) Switch windows: `Ctrl-b w` (list) or `Ctrl-b n/p` (next/prev)
3) Optional splits in a window (same worktree):
   - Vertical: `Ctrl-b %`
   - Horizontal: `Ctrl-b "`
   - Move focus: `Ctrl-b` + arrow keys
4) Detach (leave everything running): `Ctrl-b d`
5) Reattach anytime: `tmux attach -t plasma`
6) Close a pane/window when done: `exit` in that pane/window (or `Ctrl-b &` to kill window)

## Light quality-of-life config (optional)
Create `~/.tmux.conf` on the server:
```
set -g mouse on
set -g base-index 1
set -g renumber-windows on
setw -g automatic-rename on
```
Apply in-session: `tmux source-file ~/.tmux.conf`

## Tips to avoid confusion
- Keep one branch per worktree; don’t check out the same branch twice.
- Use one Cursor/IDE window pointed at your main worktree (e.g., Blender); run all other CLI/AI processes inside tmux windows for other worktrees so you don’t need more IDE instances.
- If a pane starts elsewhere, run `pwd` then `cd` to the right worktree so commands/logs stay isolated.

If you want, tell me which shell you’re using to reach the server (WSL/Ubuntu, Git Bash, etc.) and I’ll tailor the exact install/alias steps. Also, if you’d like, I can give you a single command that spins up all four windows with their directories pre-set.

```plaintext
set -g mouse on
set -g base-index 1
set -g renumber-windows on
setw -g automatic-rename on
```

---

**User**

i'm getting some things wrong i'm sure of it. i shut everything down apart from this window to start again. i'm using WSL/Ubuntu. ignore *-Test, that was just for setting this up. i'd like to rename it and i do need to create a few more, but for now we have Blender, PINN-v4 and MultiAgent to work with. please be specific with instruction, like specify whether i should be working on the server (this machine) or the client.

---

**Cursor**

Here’s a clean, specific plan for WSL/Ubuntu (server) vs client. I couldn’t inspect your repo directly because no folder is open here, so I’ll guide you step-by-step.

## What to run on the **server** (WSL/Ubuntu box)
1) Make sure tmux is installed: `sudo apt-get update && sudo apt-get install -y tmux`
2) Go to your main repo (adjust path if different):
   - `cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean`
   - Check worktrees: `git worktree list`
3) Remove the temporary Test worktree (if you’re done with it):
   - `git worktree remove /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Test`
   - `git worktree prune`
   - If you want it back with a better name, re-add (example):  
     `git worktree add /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Scratch origin/main -b scratch/demo`
     (swap path/branch as you like)
4) Spin up a tmux session with windows per worktree (one-time per login):
   ```
   session=plasma
   tmux has-session -t $session 2>/dev/null || \
     tmux new-session -d -s $session -n blender -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender
   tmux new-window -t $session -n pinn -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4
   tmux new-window -t $session -n multi -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent
   tmux attach -t $session
   ```
   - Navigation: `Ctrl-b w` (list), `Ctrl-b n/p` (next/prev), `Ctrl-b d` (detach), `Ctrl-b %` / `Ctrl-b "` (splits), `exit` to close pane/window.
   - Optional `~/.tmux.conf`:
     ```
     set -g mouse on
     set -g base-index 1
     set -g renumber-windows on
     setw -g automatic-rename on
     ```
     Reload: `tmux source-file ~/.tmux.conf`
5) Daily use: SSH in → `tmux attach -t plasma` → pick the window → work. One IDE window can stay on Blender; run commands for PINN/multi in their tmux windows without extra IDE windows.

## What to do on the **client**
- Just SSH into the server and `tmux attach -t plasma`. No need for multiple Cursor windows on the client.
- If you must open an IDE, point it to one worktree (e.g., Blender) and leave the rest to tmux.

## Please confirm
- Run `git worktree list` on the server and share the output.
- Tell me the new name/path you want instead of `PlasmaDX-Test` and whether to keep its branch; I’ll give exact add/remove commands.
- If any path above differs, paste your actual paths and I’ll update the tmux one-liner.

```plaintext
   session=plasma
   tmux has-session -t $session 2>/dev/null || \
     tmux new-session -d -s $session -n blender -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender
   tmux new-window -t $session -n pinn -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4
   tmux new-window -t $session -n multi -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent
   tmux attach -t $session
```

```plaintext
     set -g mouse on
     set -g base-index 1
     set -g renumber-windows on
     setw -g automatic-rename on
```

---

**User**

i was going fine up until this point, i'm just unsure of certain things. so tmux doesn't eliminate the need for multiple windows on the server? i can't have one main window and change between worktrees like tabs? could you elaborate on spin up a tmux session with windows per worktree. i'm not confident with what i should be doing here, could you be as explantory and detailed. would it also be possible to perform the server side steps then i can inspect what you did. i'll share a screenshot, anothr thing i'm not sure about is when i started over i just closed it all down serverside and started a blank new window. you can see that tmux has atleast been invoked with the worktree names present

---

**Cursor**

Tmux can replace multiple terminal windows: you keep one terminal attached to a tmux session, and inside tmux you switch “tabs” (windows) for each worktree. You don’t need extra Cursor/IDE windows on the server; just one IDE (pointed at a main worktree) plus tmux for all other shells.

### Do this on the server (WSL/Ubuntu)
Run these commands in your existing terminal:

1) Standardize the tmux session with exactly three windows (blender/pinn/multi):
```
session=plasma
tmux kill-session -t $session 2>/dev/null
tmux new-session  -d -s $session -n blender -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender
tmux new-window   -t $session:2 -n pinn  -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4
tmux new-window   -t $session:3 -n multi -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent
tmux select-window -t $session:1
tmux attach -t $session
```
- Switch tmux windows: `Ctrl-b w` (list) or `Ctrl-b 1/2/3` (direct).  
- Splits (same worktree window): `Ctrl-b %` (vertical), `Ctrl-b "` (horizontal).  
- Detach (leave running): `Ctrl-b d`; reattach later: `tmux attach -t plasma`.

2) Optional cleanup of the old Test worktree (if you’re done with it):
```
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
git worktree remove /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Test
git worktree prune
```
If you want a replacement (e.g., Scratch), add it with:
```
git worktree add /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Scratch origin/main -b scratch/demo
```

3) Check current worktrees (please paste this back):
```
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
git worktree list
```

### How to use day-to-day
- Open one Cursor/IDE window pointed at your main worktree (e.g., Blender).  
- For other branches, just change tmux windows inside the same terminal; no extra IDE instances needed.  
- From the client, SSH into the server and run `tmux attach -t plasma` to resume the same windows.

If anything in the paths differs, paste your `git worktree list` output and I’ll adjust the tmux setup command for you.

```plaintext
session=plasma
tmux kill-session -t $session 2>/dev/null
tmux new-session  -d -s $session -n blender -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender
tmux new-window   -t $session:2 -n pinn  -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4
tmux new-window   -t $session:3 -n multi -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent
tmux select-window -t $session:1
tmux attach -t $session
```

```plaintext
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
git worktree remove /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Test
git worktree prune
```

```plaintext
git worktree add /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Scratch origin/main -b scratch/demo
```

```plaintext
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean
git worktree list
```

---

**User**

ok cool, that's very helpful thank you. i do have an issue, because my workflow involves using several terminals with (multiple) claude code windows, or codex, gemini cli . is there a way to do this but also detatch a terminal for each instance to run in?

---

**Cursor**

You can keep one tmux session and park multiple AI terminals (Claude, Codex, Gemini) in their own windows/panes, or even run separate tmux sessions per AI. Each keeps running when you detach.

### Option A: one session, many windows
From the server (WSL):
```
session=plasma
tmux new -A -s $session

# example AI runners bound to worktree dirs
tmux new-window -t $session -n claude-blender  -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender
tmux new-window -t $session -n claude-pinn     -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4
tmux new-window -t $session -n gemini-multi    -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent
```
In each new window, start the CLI (Claude, Gemini, etc.). Detach with `Ctrl-b d`; everything keeps running. Reattach any time: `tmux attach -t plasma`.

### Option B: one session per AI (clean isolation)
```
tmux new-session -d -s ai-claude -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender
tmux new-session -d -s ai-gemini -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent
tmux new-session -d -s ai-codex  -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4
```
Attach to any: `tmux attach -t ai-claude` (or ai-gemini/ai-codex). Detach: `Ctrl-b d`. List sessions: `tmux ls`.

### Quick usage tips
- One IDE window is enough; terminals live in tmux windows/sessions.
- Name windows clearly (e.g., `claude-blender`, `gemini-multi`).
- Use splits inside a window if you want log + CLI together (`Ctrl-b %` / `Ctrl-b "`).
- To restart a CLI, just return to its window/session and rerun.

Tell me which model/CLI combinations you want, and I’ll give you exact commands (paths + names) to spin them up.

```plaintext
session=plasma
tmux new -A -s $session

# example AI runners bound to worktree dirs
tmux new-window -t $session -n claude-blender  -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender
tmux new-window -t $session -n claude-pinn     -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4
tmux new-window -t $session -n gemini-multi    -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent
```

```plaintext
tmux new-session -d -s ai-claude -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender
tmux new-session -d -s ai-gemini -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent
tmux new-session -d -s ai-codex  -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4
```

---

**User**

for now i'm doing this for claude and i'll do the rest after i know it meets my needs. for the first 3 worktrees:  PINN-v4 > claude/opus, Blender > claude/sonnet > Multi Agent > claude/opus. will they connect to their client instances on their own? i have them all running right now

---

**Cursor**

You can wire this up cleanly per worktree with dedicated tmux windows. If your Claude CLI is already configured (API key + org), it will “just connect” when you run it in each window—no extra client step needed.

### One tmux session (`plasma`), three Claude windows
From the server (WSL), inside or before attaching to tmux:
```
session=plasma
tmux new -A -s $session

tmux new-window -t $session -n pinn-claude   -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4
tmux new-window -t $session -n blender-claude -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender
tmux new-window -t $session -n multi-claude   -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent
tmux select-window -t $session:1
tmux attach -t $session
```

Then, in each tmux window, start the model you want:
- `pinn-claude` window: run your Claude CLI with opus (e.g., `claude ... --model opus` or whatever your command is).
- `blender-claude` window: run the same CLI but target sonnet (`--model sonnet`).
- `multi-claude` window: run opus again.

Detach with `Ctrl-b d`; reattach anytime with `tmux attach -t plasma`. Each window keeps running independently.

### If you prefer fully separate sessions (optional)
```
tmux new-session -d -s claude-pinn   -c /mnt/d/.../PlasmaDX-PINN-v4
tmux new-session -d -s claude-blender -c /mnt/d/.../PlasmaDX-Blender
tmux new-session -d -s claude-multi   -c /mnt/d/.../PlasmaDX-MultiAgent
```
Attach: `tmux attach -t claude-blender` (etc). Detach: `Ctrl-b d`. List: `tmux ls`.

### On “will they connect to their client instances?”
- If your Claude CLI is already configured and running, launching it in each window with the right `--model` flag is enough; it will reuse your credentials/config and connect on its own.
- No need for extra Cursor/IDE windows; keep one IDE pointing at whichever worktree you’re editing, and use tmux windows for the CLIs.

If you want, tell me the exact Claude CLI command you use (with placeholder for secrets), and I’ll drop in the ready-to-run lines per window.

```plaintext
session=plasma
tmux new -A -s $session

tmux new-window -t $session -n pinn-claude   -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4
tmux new-window -t $session -n blender-claude -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender
tmux new-window -t $session -n multi-claude   -c /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent
tmux select-window -t $session:1
tmux attach -t $session
```

```plaintext
tmux new-session -d -s claude-pinn   -c /mnt/d/.../PlasmaDX-PINN-v4
tmux new-session -d -s claude-blender -c /mnt/d/.../PlasmaDX-Blender
tmux new-session -d -s claude-multi   -c /mnt/d/.../PlasmaDX-MultiAgent
```

---

**User**

forgive me if this has already been addressed, but is it possible to have one IDE window which autoswitches to the relevant CLI instance when selected? or is there no way around muliple windows if want this. because if this isn't possible is there much point me using a second PC to do this? will i be saving performance or memory in this case? in other words i'm trying to get as close as possible to running several claude terminals in one IDE window as i can. if this isn't possible i've been given some bad advice lol

---

**Cursor**

Short answer: you don’t need multiple IDE windows. Keep one IDE, use a single terminal in it that attaches to tmux. Tmux is your “tab switcher” for all the Claude CLIs. There’s no built-in auto-follow-the-active-file terminal in Cursor/VS Code, but you can switch tmux windows instantly (Ctrl-b 1/2/3 or Ctrl-b w) and each window already lives in the right worktree/CLI.

What this looks like
- One IDE window open to any worktree you’re editing (e.g., Blender).
- One terminal panel in the IDE running `tmux attach -t plasma`.
- Inside tmux you have windows: `pinn-claude`, `blender-claude`, `multi-claude`, etc.
- Switch “tabs” with tmux keys (Ctrl-b 1/2/3, or Ctrl-b w to pick by name). No extra IDE windows needed.

Why this still helps on two PCs
- Heavy CLI/model chatter stays on the server; your client just renders one IDE + one terminal.
- Tmux sessions persist when you detach/close the IDE terminal; everything keeps running.
- Multiple GUI IDE windows on the server would cost more RAM/CPU; tmux is very light.

If you really want auto-switching tied to the active file, that isn’t a standard Cursor/VS Code feature. The closest smooth flow is:
- Keep one IDE terminal attached to tmux.
- Name tmux windows by worktree/CLI (`blender-claude`, `pinn-claude`, etc.).
- Jump with Ctrl-b window numbers/names as you change context.

If you want, I can drop a tiny helper script/keybinding so you can type `tb` → picks a window by name, or `t1/t2/t3` aliases for your three Claude panes.

---

**User**

could you write some scripts that can set all of this up for me, the original approach and the second with your modified idea. i'm just still struggling to get this set up properly. could you also create a .md document that explains everything about setting it up in as much detail as you can, because i'm having some difficulties getting this working

---

**Cursor**

```shellscript
#!/usr/bin/env bash
# Set up a single tmux session with windows for each worktree and Claude CLIs.
# Assumes WSL/Ubuntu paths; update ROOT if your paths differ.
set -euo pipefail

ROOT="/mnt/d/Users/dilli/AndroidStudioProjects"
SESSION="plasma"

# Ordered pairs: window-name then directory.
WINDOWS=(
  blender        "$ROOT/PlasmaDX-Blender"
  pinn           "$ROOT/PlasmaDX-PINN-v4"
  multi          "$ROOT/PlasmaDX-MultiAgent"
  claude-blender "$ROOT/PlasmaDX-Blender"
  claude-pinn    "$ROOT/PlasmaDX-PINN-v4"
  claude-multi   "$ROOT/PlasmaDX-MultiAgent"
)

main_name="${WINDOWS[0]}"
main_dir="${WINDOWS[1]}"

# Start the session (detached) with the first window.
tmux has-session -t "$SESSION" 2>/dev/null || \
  tmux new-session -d -s "$SESSION" -n "$main_name" -c "$main_dir"

# Create the rest of the windows.
for ((i=2; i<${#WINDOWS[@]}; i+=2)); do
  name="${WINDOWS[i]}"
  dir="${WINDOWS[i+1]}"
  tmux new-window -t "$SESSION" -n "$name" -c "$dir"
done

# Jump to the first window and attach.
tmux select-window -t "$SESSION:1"
tmux attach -t "$SESSION"

```

```shellscript
#!/usr/bin/env bash
# Start separate tmux sessions per Claude CLI to keep them isolated.
# Assumes WSL/Ubuntu paths; update ROOT if needed.
set -euo pipefail

ROOT="/mnt/d/Users/dilli/AndroidStudioProjects"

declare -A SESSIONS=(
  [claude-pinn]="$ROOT/PlasmaDX-PINN-v4"
  [claude-blender]="$ROOT/PlasmaDX-Blender"
  [claude-multi]="$ROOT/PlasmaDX-MultiAgent"
)

for name in "${!SESSIONS[@]}"; do
  dir="${SESSIONS[$name]}"
  tmux has-session -t "$name" 2>/dev/null || tmux new-session -d -s "$name" -c "$dir"
done

echo "Available sessions:"
tmux ls
echo "Attach with: tmux attach -t <session-name> (e.g., tmux attach -t claude-blender)"

```

```markdown
# tmux + worktrees + Claude CLI (WSL)

This doc shows two patterns to run multiple Claude CLIs across your worktrees with minimal IDE windows:
- **Single session / many windows** (recommended): one `tmux` session named `plasma` with windows for each worktree plus dedicated Claude windows.
- **Per-AI sessions** (optional): separate `tmux` sessions for each Claude instance.

Everything here assumes WSL/Ubuntu paths. Adjust `ROOT` if your paths differ.

---

## Prereqs
- `sudo apt-get update && sudo apt-get install -y tmux`
- Worktrees on the server, e.g.:
  - `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender`
  - `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4`
  - `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent`
- Claude CLI already configured (API key/org). Each tmux window just runs the CLI; it will connect on its own.

## Quick tmux keys
- List/switch windows: `Ctrl-b w` (choose) or `Ctrl-b 1/2/3...`
- Splits (same worktree window): `Ctrl-b %` (vertical), `Ctrl-b "` (horizontal)
- Detach: `Ctrl-b d` (everything keeps running)
- Reattach: `tmux attach -t plasma` (or session name)

---

## Pattern A: One session, many windows (recommended)
Script: `scripts/tmux_single_session.sh`

What it does:
- Creates/attaches to session `plasma`.
- Windows (rename as needed):
  - `blender` → `/PlasmaDX-Blender`
  - `pinn` → `/PlasmaDX-PINN-v4`
  - `multi` → `/PlasmaDX-MultiAgent`
  - `claude-blender` → same dir, run Claude Sonnet here.
  - `claude-pinn` → same dir, run Claude Opus here.
  - `claude-multi` → same dir, run Claude Opus here.
- Attaches to the session.

Run it:
```bash
chmod +x scripts/tmux_single_session.sh
./scripts/tmux_single_session.sh
```

After attaching, start the CLIs in their windows:
- In `claude-blender`: `claude ... --model sonnet` (example)
- In `claude-pinn`: `claude ... --model opus`
- In `claude-multi`: `claude ... --model opus`

Day-to-day loop:
1) SSH into the server (from client) → `tmux attach -t plasma`
2) `Ctrl-b w` to jump between worktrees/CLIs
3) `Ctrl-b d` to leave everything running

---

## Pattern B: Separate sessions per Claude (optional isolation)
Script: `scripts/tmux_per_ai_sessions.sh`

What it does:
- Creates detached sessions: `claude-pinn`, `claude-blender`, `claude-multi`, each in its own worktree.
- Prints the session list.

Run it:
```bash
chmod +x scripts/tmux_per_ai_sessions.sh
./scripts/tmux_per_ai_sessions.sh
tmux attach -t claude-blender   # or claude-pinn / claude-multi
```

Detach with `Ctrl-b d`; repeat `tmux attach -t <name>` as needed. `tmux ls` to list them.

---

## Optional helpers
- Put tmux QOL in `~/.tmux.conf`:
  ```bash
  set -g mouse on
  set -g base-index 1
  set -g renumber-windows on
  setw -g automatic-rename on
  ```
  Reload in-session: `tmux source-file ~/.tmux.conf`

- Simple shell aliases (add to `~/.bashrc` or `~/.zshrc`):
  ```bash
  alias ta='tmux attach -t plasma'
  alias tls='tmux ls'
  alias t1="tmux switch-client -t plasma \; select-window -t 1"
  alias t2="tmux switch-client -t plasma \; select-window -t 2"
  alias t3="tmux switch-client -t plasma \; select-window -t 3"
  ```

---

## Notes on IDE windows
- You can keep **one** Cursor/IDE window. Use its terminal to attach to tmux; switch tmux windows instead of opening more IDE windows.
- The CLI windows keep running after you close/detach the IDE terminal; reattach later from any client.

---

## Troubleshooting
- “Session already exists”: `tmux kill-session -t plasma` (or the name) if you truly want a fresh start.
- “Path doesn’t exist”: update `ROOT` in the scripts to your actual mount/drive.
- “CLI won’t connect”: re-check your Claude CLI auth/env vars; tmux itself just keeps the process alive.

```

I added two runnable scripts plus a detailed doc:

- `scripts/tmux_single_session.sh`: one tmux session (`plasma`) with windows for your three worktrees and three Claude CLIs (sonnet for Blender, opus for PINN/multi). Runs detached, builds the windows, then attaches.
- `scripts/tmux_per_ai_sessions.sh`: separate tmux sessions (`claude-blender`, `claude-pinn`, `claude-multi`) for isolated CLIs; lists them so you can attach individually.
- `docs/tmux_worktrees_setup.md`: step-by-step setup, daily usage, key bindings, optional aliases/config, and troubleshooting.

Key usage (server/WSL):
1) `chmod +x scripts/tmux_single_session.sh`  
2) `./scripts/tmux_single_session.sh`  
   - In windows `claude-blender`/`claude-pinn`/`claude-multi`, launch your Claude CLI with the right `--model` (sonnet or opus).  
   - Switch windows: `Ctrl-b w` or `Ctrl-b 1/2/3...`; detach `Ctrl-b d`; reattach `tmux attach -t plasma`.
3) If you prefer isolated sessions, run `chmod +x scripts/tmux_per_ai_sessions.sh && ./scripts/tmux_per_ai_sessions.sh`, then `tmux attach -t claude-blender` (etc).

Notes:
- Update `ROOT` in the scripts if your paths differ from `/mnt/d/Users/dilli/AndroidStudioProjects`.
- You only need one IDE window; use its terminal to `tmux attach -t plasma` and switch tmux windows instead of opening more IDE instances.
- CLIs will connect as usual if your Claude CLI is already configured.

If you want, tell me your exact Claude CLI commands (with secrets redacted) and I’ll bake them into the windows so they auto-start when the session comes up. Also let me know the new name/path you want instead of the old Test worktree and I’ll script that add/remove.

