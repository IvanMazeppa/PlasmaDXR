\# tmux + worktrees + Claude CLI (WSL)



This doc shows two patterns to run multiple Claude CLIs across your worktrees with minimal IDE windows:

\- \*\*Single session / many windows\*\* (recommended): one `tmux` session named `plasma` with windows for each worktree plus dedicated Claude windows.

\- \*\*Per-AI sessions\*\* (optional): separate `tmux` sessions for each Claude instance.



Everything here assumes WSL/Ubuntu paths. Adjust `ROOT` if your paths differ.



---



\## Prereqs

\- ` `

\- Worktrees on the server, e.g.:

&nbsp; - `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Blender`

&nbsp; - `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-PINN-v4`

&nbsp; - `/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-MultiAgent`

\- Claude CLI already configured (API key/org). Each tmux window just runs the CLI; it will connect on its own.



\## Quick tmux keys

\- List/switch windows: `Ctrl-b w` (choose) or `Ctrl-b 1/2/3...`

\- Splits (same worktree window): `Ctrl-b %` (vertical), `Ctrl-b "` (horizontal)

\- Detach: `Ctrl-b d` (everything keeps running)

\- Reattach: `tmux attach -t plasma` (or session name)



---



\## Pattern A: One session, many windows (recommended)

Script: `scripts/tmux\_single\_session.sh`



What it does:

\- Creates/attaches to session `plasma`.

\- Windows (rename as needed):

&nbsp; - `blender` → `/PlasmaDX-Blender`

&nbsp; - `pinn` → `/PlasmaDX-PINN-v4`

&nbsp; - `multi` → `/PlasmaDX-MultiAgent`

&nbsp; - `claude-blender` → same dir, run Claude Sonnet here.

&nbsp; - `claude-pinn` → same dir, run Claude Opus here.

&nbsp; - `claude-multi` → same dir, run Claude Opus here.

\- Attaches to the session.



Run it:

```bash

chmod +x scripts/tmux\_single\_session.sh

./scripts/tmux\_single\_session.sh

```



After attaching, start the CLIs in their windows:

\- In `claude-blender`: `claude ... --model sonnet` (example)

\- In `claude-pinn`: `claude ... --model opus`

\- In `claude-multi`: `claude ... --model opus`



Day-to-day loop:

1\) SSH into the server (from client) → `tmux attach -t plasma`

2\) `Ctrl-b w` to jump between worktrees/CLIs

3\) `Ctrl-b d` to leave everything running



---



\## Pattern B: Separate sessions per Claude (optional isolation)

Script: `scripts/tmux\_per\_ai\_sessions.sh`



What it does:

\- Creates detached sessions: `claude-pinn`, `claude-blender`, `claude-multi`, each in its own worktree.

\- Prints the session list.



Run it:

```bash

chmod +x scripts/tmux\_per\_ai\_sessions.sh

./scripts/tmux\_per\_ai\_sessions.sh

tmux attach -t claude-blender   # or claude-pinn / claude-multi

```



Detach with `Ctrl-b d`; repeat `tmux attach -t <name>` as needed. `tmux ls` to list them.



---



\## Optional helpers

\- Put tmux QOL in `~/.tmux.conf`:

&nbsp; ```bash

&nbsp; set -g mouse on

&nbsp; set -g base-index 1

&nbsp; set -g renumber-windows on

&nbsp; setw -g automatic-rename on

&nbsp; ```

&nbsp; Reload in-session: `tmux source-file ~/.tmux.conf`



\- Simple shell aliases (add to `~/.bashrc` or `~/.zshrc`):

&nbsp; ```bash

&nbsp; alias ta='tmux attach -t plasma'

&nbsp; alias tls='tmux ls'

&nbsp; alias t1="tmux switch-client -t plasma \\; select-window -t 1"

&nbsp; alias t2="tmux switch-client -t plasma \\; select-window -t 2"

&nbsp; alias t3="tmux switch-client -t plasma \\; select-window -t 3"

&nbsp; ```



---



\## Notes on IDE windows

\- You can keep \*\*one\*\* Cursor/IDE window. Use its terminal to attach to tmux; switch tmux windows instead of opening more IDE windows.

\- The CLI windows keep running after you close/detach the IDE terminal; reattach later from any client.



---



\## Troubleshooting

\- “Session already exists”: `tmux kill-session -t plasma` (or the name) if you truly want a fresh start.

\- “Path doesn’t exist”: update `ROOT` in the scripts to your actual mount/drive.

\- “CLI won’t connect”: re-check your Claude CLI auth/env vars; tmux itself just keeps the process alive.





