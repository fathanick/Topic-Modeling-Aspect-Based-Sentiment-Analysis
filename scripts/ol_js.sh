#!/bin/bash
# Evaluate JS in the Overleaf tab without raising the window.
# usage: ol_js.sh '<javascript expression>'
osascript <<EOF
tell application "Google Chrome"
  repeat with w from 1 to (count of windows)
    repeat with t from 1 to (count of tabs of window w)
      if URL of tab t of window w contains "overleaf.com/project" then
        return execute (tab t of window w) javascript "$1"
      end if
    end repeat
  end repeat
end tell
EOF
