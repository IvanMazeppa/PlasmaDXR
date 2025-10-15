import argparse
import datetime
import json
import os
import pathlib
import sys
from typing import List, Dict, Any

try:
	from anthropic import Anthropic
except Exception as e:
	Anthropic = None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
VERSIONS_DIR = REPO_ROOT / "Versions"
DOCS_GLOB = [
	"README.md",
	"MASTER_ROADMAP_V2.md",
	"NEXT_SESSION_START_HERE.md",
	"SESSION_SUMMARY*.md",
	"CLAUDE.md",
	"DX12_MCP_README.md",
	"RESTIR_*.md",
	"SHADOW_*.md",
	"PIX_*.md",
]


def load_recent_docs(max_chars: int = 20000) -> str:
	contents: List[str] = []
	for pattern in DOCS_GLOB:
		for p in REPO_ROOT.glob(pattern):
			try:
				text = p.read_text(encoding="utf-8", errors="ignore")
				contents.append(f"# FILE: {p.relative_to(REPO_ROOT)}\n\n" + text)
			except Exception:
				continue
	joined = "\n\n\n".join(sorted(contents, key=len, reverse=True))
	return joined[:max_chars]


def build_system_prompt() -> str:
	guardrails = (
		"MCP Background Agent Guardrails:\n"
		"- Write-only via versioned patches under Versions/ (YYYYMMDD-HHMM_description.patch).\n"
		"- No deletions/renames without explicit APPROVE_DELETE token in patch note.\n"
		"- Prefer minimal, non-destructive edits; explain rationale succinctly.\n"
	)
	context = load_recent_docs()
	return (
		"You are a repository assistant operating under strict guardrails.\n"
		+ guardrails
		+ "When the user requests changes, output a concise diff plan and propose patch content.\n"
		+ "Focus on DX12/DXR code quality, readability, and performance.\n\n"
		+ "Context (recent docs):\n" + context
	)


def write_patch(description_slug: str, body: str) -> pathlib.Path:
	now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
	filename = f"{now}_{description_slug}.patch"
	VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
	path = VERSIONS_DIR / filename
	path.write_text(body, encoding="utf-8")
	return path


def summarize_messages(messages: List[Dict[str, Any]]) -> str:
	# Minimal rolling summary to keep context small
	summary_lines: List[str] = []
	for m in messages[-10:]:
		role = m.get("role", "user")
		content = m.get("content", "")
		if isinstance(content, list):
			content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
		summary_lines.append(f"[{role}] {content[:300].strip()}")
	return "\n".join(summary_lines)


def call_anthropic(model: str, system_prompt: str, user_content: str, max_tokens: int = 1200) -> str:
	if Anthropic is None:
		raise RuntimeError("anthropic package not installed. pip install anthropic")
	api_key = os.environ.get("ANTHROPIC_API_KEY")
	if not api_key:
		raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
	client = Anthropic(api_key=api_key)
	resp = client.messages.create(
		model=model,
		max_tokens=max_tokens,
		system=system_prompt,
		messages=[{"role": "user", "content": user_content}],
	)
	# Extract plain text
	parts = []
	for block in getattr(resp, "content", []) or []:
		text = getattr(block, "text", None)
		if text:
			parts.append(text)
	return "\n".join(parts).strip()


def main() -> int:
	parser = argparse.ArgumentParser(description="Minimal Anthropic-backed repo agent")
	parser.add_argument("prompt", nargs="*", help="User request to the agent")
	parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Anthropic model name")
	parser.add_argument("--slug", default="agent_note", help="Patch filename slug")
	parser.add_argument("--no-write", action="store_true", help="Do not write a patch file, just print output")
	args = parser.parse_args()

	user_prompt = " ".join(args.prompt).strip()
	if not user_prompt:
		print("Provide a prompt, e.g.: python agent.py 'propose blit pipeline edits'", file=sys.stderr)
		return 2

	system_prompt = build_system_prompt()
	conversation_header = (
		"Task: Respond with a short plan and, if appropriate, a patch note body that can be saved under Versions/.\n"
		"If proposing code edits, include a summary diff plan and file paths, not full code.\n"
	)

	llm_input = conversation_header + "\n\n" + user_prompt
	try:
		output = call_anthropic(args.model, system_prompt, llm_input)
	except Exception as e:
		print(f"[ERROR] {e}", file=sys.stderr)
		return 1

	if args.no_write:
		print(output)
		return 0

	patch_path = write_patch(args.slug, output + "\n")
	print(f"Wrote: {patch_path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())


