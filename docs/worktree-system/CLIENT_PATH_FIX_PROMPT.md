# Client Machine PATH Fix

**Issue:** The `tstart` script can't run from outside Cursor because `~/.bashrc` aliases aren't loaded in all shell contexts.

**Fix:** Add the scripts folder to PATH in `~/.bashrc` so `tmux_per_ai_sessions.sh` can be called directly from any terminal.

Run this command:

```bash
echo 'export PATH="$PATH:/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/scripts"' >> ~/.bashrc && source ~/.bashrc
```

**Verify it works:**
```bash
# From any terminal (not just Cursor)
tmux_per_ai_sessions.sh
```

**Alternative:** If you prefer to keep using the alias, ensure your terminal sources `~/.bashrc` on startup. For non-login shells, add to `~/.bash_profile`:

```bash
echo 'source ~/.bashrc' >> ~/.bash_profile
```

---

**Full alias block (if missing from client's ~/.bashrc):**

```bash
cat >> ~/.bashrc << 'EOF'
# PlasmaDX tmux worktree aliases
alias tb='tmux attach -t claude-blender'
alias tp='tmux attach -t claude-pinn'
alias tm='tmux attach -t claude-multi'
alias tls='tmux ls'
alias tstart='/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/scripts/tmux_per_ai_sessions.sh'
alias thelp='echo "tb=blender, tp=pinn, tm=multi, tls=list, tstart=create sessions"'
export PATH="$PATH:/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/scripts"
EOF
source ~/.bashrc
```
