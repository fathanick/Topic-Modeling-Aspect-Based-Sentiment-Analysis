import json
import subprocess
import sys


def ol_js(expr: str) -> str:
    """Run JS in the logged-in Overleaf tab via AppleScript, no window raise."""
    script = f'''
tell application "Google Chrome"
  repeat with w from 1 to (count of windows)
    repeat with t from 1 to (count of tabs of window w)
      if URL of tab t of window w contains "overleaf.com/project" then
        return execute (tab t of window w) javascript {json.dumps(expr)}
      end if
    end repeat
  end repeat
end tell
'''
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return r.stdout.strip() or r.stderr.strip()


if __name__ == "__main__":
    print(ol_js(sys.argv[1]))
