import glob
import os
import shutil
import subprocess
import sys
import time
import shlex
from typing import Dict, List, Optional

from experience.llm_client.agent_task import AgentTask
from experience.fs_util.pack_dir import pack_dir


def _find_ducc_bin() -> str:
    """Find ducc binary in order of priority:
    1. DUCC_BIN environment variable
    2. ducc in PATH
    3. glob pattern matching ~/.comate/extensions/baidu.baidu-cc-*/...
    """
    # 1. Check environment variable
    if env_bin := os.environ.get("DUCC_BIN"):
        return env_bin

    # 2. Check PATH
    if path_bin := shutil.which("ducc"):
        return path_bin

    # 3. Glob pattern for comate extension
    pattern = os.path.expanduser(
        "~/.comate/extensions/baidu.baidu-cc-*/resources/native-binary/bin/ducc"
    )
    matches = sorted(glob.glob(pattern), reverse=True)  # Sort descending to get latest version
    if matches:
        return matches[0]

    # Fallback: raise error with helpful message
    raise FileNotFoundError(
        "Cannot find ducc binary. Please either:\n"
        "  1. Set DUCC_BIN environment variable\n"
        "  2. Add ducc to your PATH\n"
        "  3. Install baidu-cc extension in ~/.comate/extensions/"
    )


DUCC_BIN = _find_ducc_bin()

# Default configuration for tmux interactive mode
TMUX_SESSION_PREFIX = "ducc_interactive"
TMUX_CHECK_INTERVAL = 3  # Check interval in seconds
TMUX_IDLE_THRESHOLD = 5  # Number of consecutive unchanged checks to consider idle
TMUX_INITIAL_WAIT = 5    # Initial wait time for ducc to start (seconds)

# Prompt patterns that require auto-confirmation
AUTO_CONFIRM_PATTERNS = {
    "Do you want to proceed": "Enter",
    "Yes, I trust this folder": "Enter",
    "allow all edits during this session": "Down Enter",
    "Press Enter to continue": "Enter",
    "Yes, I accept": "Down Enter",  # Bypass Permissions warning - select "Yes, I accept"
    "No, exit": "Down Enter",  # Same warning - default is "No, exit", need to go down to "Yes"
}


def _flatten_nested(nested) -> list:
    """Flatten a nested list structure into a flat list."""
    if not isinstance(nested, list):
        return [nested]
    result = []
    for item in nested:
        result.extend(_flatten_nested(item))
    return result


def _grep_by_file_content_hint(root_dir: str, todo_file_content_hint: str) -> List[str]:
    """Find all files under root_dir whose content contains the hint string."""
    todo_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if todo_file_content_hint in content:
                    todo_files.append(file_path)
            except (UnicodeDecodeError, OSError):
                continue
    return todo_files


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    import re
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)


def _tmux_available() -> bool:
    """Check if tmux command is available on the system."""
    try:
        subprocess.run(["tmux", "-V"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _tmux_session_exists(session_name: str) -> bool:
    """Check if a tmux session exists."""
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _tmux_create_session(session_name: str, working_dir: str) -> None:
    """Create a new tmux session."""
    try:
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, "-c", working_dir],
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "tmux is not installed or not found in PATH. "
            "Please install tmux to use interactive mode."
        ) from e


def _tmux_kill_session(session_name: str) -> None:
    """Kill a tmux session."""
    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True,
    )


def _tmux_send_keys(session_name: str, *keys: str) -> None:
    """Send keys to a tmux session."""
    subprocess.run(
        ["tmux", "send-keys", "-t", session_name] + list(keys),
        check=True,
    )


def _tmux_send_text(session_name: str, text: str) -> None:
    """Send long text to a tmux session using load-buffer and paste.

    This is more reliable than send-keys for long or multi-line text.
    """
    import tempfile
    # Write text to a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(text)
        temp_path = f.name

    try:
        # Load the text into tmux buffer
        subprocess.run(
            ["tmux", "load-buffer", temp_path],
            check=True,
        )
        # Paste the buffer into the session
        subprocess.run(
            ["tmux", "paste-buffer", "-t", session_name],
            check=True,
        )
    finally:
        # Clean up temp file
        os.remove(temp_path)


def _tmux_capture_pane(session_name: str) -> str:
    """Capture the content of a tmux pane."""
    result = subprocess.run(
        ["tmux", "capture-pane", "-p", "-e", "-S", "-", "-t", session_name],
        capture_output=True,
        text=True,
    )
    return result.stdout


def _check_and_auto_confirm(session_name: str, patterns: Dict[str, str]) -> bool:
    """Check for auto-confirm patterns and send keys if matched.

    Returns True if a pattern was matched and keys were sent.
    """
    content = _tmux_capture_pane(session_name)
    clean_content = _strip_ansi(content)

    for pattern, keys in patterns.items():
        if pattern in clean_content:
            print(f"[tmux] Detected '{pattern}' -> sending {keys}", file=sys.stderr)
            for key in keys.split():
                _tmux_send_keys(session_name, key)
            return True
    return False


def _wait_for_ducc_idle(
    session_name: str,
    check_interval: float = TMUX_CHECK_INTERVAL,
    idle_threshold: int = TMUX_IDLE_THRESHOLD,
    auto_confirm: bool = True,
    timeout: float = 600,  # 10 minutes default timeout
) -> str:
    """Wait for ducc to become idle (screen content stops changing).

    Args:
        session_name: tmux session name
        check_interval: seconds between checks
        idle_threshold: consecutive unchanged checks to consider idle
        auto_confirm: whether to auto-confirm prompts
        timeout: maximum wait time in seconds

    Returns:
        The final screen content
    """
    last_content = ""
    idle_count = 0
    start_time = time.time()
    command_completed = False

    while True:
        if time.time() - start_time > timeout:
            print(f"[tmux] Timeout ({timeout}s), forcing exit", file=sys.stderr)
            break

        if not _tmux_session_exists(session_name):
            print(f"[tmux] Session '{session_name}' has been closed", file=sys.stderr)
            break

        content = _tmux_capture_pane(session_name)
        clean_content = _strip_ansi(content)

        # Auto-confirm if enabled
        if auto_confirm:
            if _check_and_auto_confirm(session_name, AUTO_CONFIRM_PATTERNS):
                time.sleep(1)  # Wait a bit after auto-confirm
                idle_count = 0
                last_content = ""  # Reset to re-check
                continue

        # Check if command has completed (shell prompt appears after ducc output)
        # Look for patterns indicating ducc has finished
        lines = clean_content.strip().split('\n')
        last_lines = '\n'.join(lines[-5:]) if len(lines) >= 5 else clean_content

        # ducc completed indicators:
        # 1. DUCC_COMPLETED marker (from our wrapper script)
        # 2. Shell prompt at the end (ends with % or $ after ducc ran)
        # 3. "Done." message from ducc
        # 4. ducc idle state: empty prompt line with ❯ at the end (after processing)
        if not command_completed:
            # Check for ducc idle state: line ends with ❯
            # This indicates ducc has finished processing and is waiting for new input
            ducc_idle = False
            if '❯' in last_lines:
                # ducc shows ❯ when idle, check if it's at the end of a line by itself
                for line in lines[-3:]:
                    stripped = line.strip()
                    # Empty prompt line or line ending with just ❯
                    if stripped == '❯' or (stripped.endswith('❯') and len(stripped) < 5):
                        ducc_idle = True
                        break

            if ('DUCC_COMPLETED' in clean_content or
                'Done.' in clean_content or
                ducc_idle or
                (clean_content.count('%') >= 2 and last_lines.rstrip().endswith('%'))):
                command_completed = True
                print(f"[tmux] ducc command execution completed", file=sys.stderr)

        if command_completed:
            if content == last_content:
                idle_count += 1
                print(f"[tmux] Screen unchanged ({idle_count}/{idle_threshold})", file=sys.stderr)
                if idle_count >= idle_threshold:
                    break
            else:
                idle_count = 0
                last_content = content
        else:
            print(f"[tmux] Waiting for ducc to complete...", file=sys.stderr)
            last_content = content

        time.sleep(check_interval)

    return content if 'content' in dir() else ""


class DuccTaskHandler:
    """Handler for running tasks via ducc CLI.

    Supports two modes:
    - interactive=False (default): Run ducc via subprocess, non-interactive
    - interactive=True: Run ducc in tmux session for visual observation
    """

    def __call__(
        self,
        all_tasks,
        llm_env: Optional[Dict[str, str]] = None,
        interactive: bool = False,
        auto_confirm: bool = True,
        tmux_session: Optional[str] = None,
    ) -> None:
        """Execute tasks via ducc.

        Args:
            all_tasks: List of AgentTask objects
            llm_env: Optional environment variables
            interactive: If True, run in tmux for visual interaction
            auto_confirm: If True (and interactive), auto-confirm prompts
            tmux_session: Custom tmux session name (interactive mode only)
        """
        if interactive:
            self._run_interactive(all_tasks, llm_env, auto_confirm, tmux_session)
        else:
            self._run_batch(all_tasks, llm_env)

    def _run_batch(self, all_tasks, llm_env: Optional[Dict[str, str]] = None) -> None:
        """Original batch mode: run ducc via subprocess."""
        flat_tasks = _flatten_nested(all_tasks)

        for task_idx, task in enumerate(flat_tasks):
            workspace_dir = task.workspace_dir
            prompt = task.prompt
            output_relative_dir_or_list = task.output_relative_dir
            todo_file_content_hint = task.todo_file_content_hint

            packed_workspace = pack_dir(workspace_dir)

            if isinstance(output_relative_dir_or_list, str):
                output_relative_dirs = [output_relative_dir_or_list]
            else:
                output_relative_dirs = output_relative_dir_or_list

            for output_relative_dir in output_relative_dirs:
                output_root = os.path.join(workspace_dir, output_relative_dir)
                todo_file_paths = _grep_by_file_content_hint(output_root, todo_file_content_hint)

                for todo_file_path in todo_file_paths:
                    ducc_prompt = (
                        f"{prompt}\n\n"
                        f"The Directories are packed as following like repomix:\n\n"
                        f"{packed_workspace}\n\n"
                        f"The whole task are split into several pieces, One LLMs request for one piece, "
                        f"In This request, Your output will replace the {todo_file_content_hint} placeholder in file {todo_file_path}. \n"
                        f"Output raw text only. Do NOT wrap in markdown code fences (``` or ```lang).\n"
                        f"Do not generate unrelated content."
                    )

                    print(f"\n[ducc] Task {task_idx + 1}/{len(flat_tasks)}: {todo_file_path}", file=sys.stderr)

                    cmd = [
                        DUCC_BIN,
                        "-p", ducc_prompt,
                        "--allowedTools", "Read,Edit,Write",
                        "--permission-mode", "bypassPermissions",
                    ]

                    result = subprocess.run(
                        cmd,
                        cwd=workspace_dir,
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode != 0:
                        print(f"[ducc] stderr: {result.stderr}", file=sys.stderr)

                    output_content = result.stdout.strip() if result.stdout else ""
                    if output_content:
                        with open(todo_file_path, "w", encoding="utf-8") as f:
                            f.write(output_content)

    def _run_interactive(
        self,
        all_tasks,
        llm_env: Optional[Dict[str, str]] = None,
        auto_confirm: bool = True,
        tmux_session: Optional[str] = None,
    ) -> None:
        """Interactive mode: run ducc in tmux session for visual observation.

        This mode launches ducc without -p flag first, then sends the prompt
        via tmux send-keys, allowing full TUI visualization.
        """
        flat_tasks = _flatten_nested(all_tasks)

        for task_idx, task in enumerate(flat_tasks):
            workspace_dir = task.workspace_dir
            prompt = task.prompt
            output_relative_dir_or_list = task.output_relative_dir
            todo_file_content_hint = task.todo_file_content_hint

            packed_workspace = pack_dir(workspace_dir)

            if isinstance(output_relative_dir_or_list, str):
                output_relative_dirs = [output_relative_dir_or_list]
            else:
                output_relative_dirs = output_relative_dir_or_list

            for output_relative_dir in output_relative_dirs:
                output_root = os.path.join(workspace_dir, output_relative_dir)
                todo_file_paths = _grep_by_file_content_hint(output_root, todo_file_content_hint)

                for file_idx, todo_file_path in enumerate(todo_file_paths):
                    ducc_prompt = (
                        f"{prompt}\n\n"
                        f"The Directories are packed as following like repomix:\n\n"
                        f"{packed_workspace}\n\n"
                        f"The whole task are split into several pieces, One LLMs request for one piece, "
                        f"In This request, Your output will replace the {todo_file_content_hint} placeholder in file {todo_file_path}. \n"
                        f"Output raw text only. Do NOT wrap in markdown code fences (``` or ```lang).\n"
                        f"Do not generate unrelated content."
                    )

                    # Create unique session name
                    session_name = tmux_session or f"{TMUX_SESSION_PREFIX}_{task_idx}_{file_idx}"

                    print(f"\n[ducc-interactive] Task {task_idx + 1}/{len(flat_tasks)}: {todo_file_path}", file=sys.stderr)
                    print(f"[ducc-interactive] tmux session: {session_name}", file=sys.stderr)
                    print(f"[ducc-interactive] Use 'tmux attach -t {session_name}' to watch real-time output", file=sys.stderr)

                    # Kill existing session if any
                    _tmux_kill_session(session_name)

                    # Create new tmux session in workspace directory
                    _tmux_create_session(session_name, workspace_dir)

                    # Write prompt to a temp file (for reference, and to send via tmux)
                    prompt_file = os.path.join(workspace_dir, ".ducc_prompt.txt")
                    with open(prompt_file, "w", encoding="utf-8") as f:
                        f.write(ducc_prompt)

                    # Step 1: Start ducc with reduced interactions
                    # IS_SANDBOX=1 skips some prompts, --permission-mode bypassPermissions skips permission checks
                    ducc_cmd = f'IS_SANDBOX=1 {DUCC_BIN} --permission-mode bypassPermissions --allowedTools "Read,Edit,Write"'
                    _tmux_send_keys(session_name, ducc_cmd, "Enter")

                    # Step 2: Wait for ducc to start and show trust prompt
                    print(f"[ducc-interactive] Waiting for ducc to start...", file=sys.stderr)
                    time.sleep(3)

                    # Step 3: Auto-confirm trust folder if enabled
                    if auto_confirm:
                        # Wait for and handle the trust folder prompt
                        for _ in range(10):  # Try for up to 10 iterations
                            if _check_and_auto_confirm(session_name, AUTO_CONFIRM_PATTERNS):
                                time.sleep(1)
                                break
                            time.sleep(1)

                    # Step 4: Wait for ducc to be ready for input (show the input prompt)
                    print(f"[ducc-interactive] Waiting for ducc to be ready...", file=sys.stderr)
                    time.sleep(2)

                    # Step 5: Send the prompt text to ducc
                    # Use tmux load-buffer and paste for long text
                    _tmux_send_text(session_name, ducc_prompt)
                    time.sleep(0.5)
                    _tmux_send_keys(session_name, "Enter")

                    # Step 6: Wait for ducc to complete the task
                    print(f"[ducc-interactive] Waiting for ducc to process task...", file=sys.stderr)
                    _wait_for_ducc_idle(
                        session_name,
                        auto_confirm=auto_confirm,
                    )

                    print(f"[ducc-interactive] Task completed, session '{session_name}' preserved for observation", file=sys.stderr)

                    # Clean up prompt file
                    try:
                        os.remove(prompt_file)
                    except OSError:
                        pass

                    # Keep session alive for user observation
                    # User can manually clean up with: tmux kill-session -t <session_name>


if __name__ == "__main__":
    import tempfile

    print("Running DuccTaskHandler tests...\n")

    # Test 1: _flatten_nested
    nested = [[1, 2], [3, [4, 5]]]
    assert _flatten_nested(nested) == [1, 2, 3, 4, 5]
    print("  ok: _flatten_nested works")

    # Test 2: Handler instantiation
    handler = DuccTaskHandler()
    assert callable(handler)
    print("  ok: DuccTaskHandler is callable")

    # Test 3: Integration test with ducc
    print("\n  Test 3: Integration test (ducc call)")
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "const_input"))
        with open(os.path.join(tmpdir, "const_input", "data.txt"), "w") as f:
            f.write("Hello world in English")
        os.makedirs(os.path.join(tmpdir, "mutable_output"))
        with open(os.path.join(tmpdir, "mutable_output", "data.txt"), "w") as f:
            f.write("TODO")
        task = AgentTask(
            workspace_dir=tmpdir,
            output_relative_dir="mutable_output",
            prompt="Translate the English text in const_input to French.",
        )
        handler([task])

        with open(os.path.join(tmpdir, "mutable_output", "data.txt"), "r") as f:
            output = f.read()
        assert "TODO" not in output, f"Output still contains TODO: {output}"
        print(f"  ok: ducc replaced TODO with: {repr(output)}")

    print("\nAll tests passed.")
