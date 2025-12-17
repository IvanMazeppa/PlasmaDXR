kkkkkkkkkkkkkkkkkkkkkkkk# Client Script Consolidation Prompt

Copy this into a Claude session on the client machine:

---

I have these scripts on my client PC that I'd like to consolidate:
- `open_worktrees` (or `open_worktrees.sh`)
- `connect_worktrees` (or `connect_worktrees.sh`)
- `setup_ssh_key` (or similar)

**Tasks:**

1. **Find the scripts:**
   ```bash
   find ~ -name "*worktree*" -o -name "*connect*ssh*" -o -name "setup_ssh*" 2>/dev/null | grep -E "\.(sh|py)$"
   ```

2. **Check if they use absolute paths** - If they have hardcoded paths like `/home/user/...` they'll work from anywhere. If they use relative paths like `./` or `../` they need to be updated.

3. **Copy them to the server's worktree-system folder:**
   ```bash
   scp <script> server-ip:/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/docs/worktree-system/
   ```

4. **Update ~/.bashrc on the client** to add the server's scripts folder to PATH:
   ```bash
   echo 'export PATH="$PATH:/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/scripts"' >> ~/.bashrc
   source ~/.bashrc
   ```

5. **Fix the status bar in Cursor** - add to settings.json:
   ```json
   {
     "workbench.statusBar.visible": true
   }
   ```
   Open with: Ctrl+Shift+P → "Preferences: Open User Settings (JSON)"

---

**Goal:** All worktree/tmux/SSH scripts consolidated in one place on the server, accessible from both machines.
