"""Upload files into the logged-in Overleaf project via its internal API.

Chrome's AppleScript `execute javascript` returns the value of the expression
and does not await promises, so each upload assigns its result to
window.__olres and a second call polls that variable.

Session cookie and CSRF token come from the page itself; no credentials here.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ol import ol_js  # noqa: E402

PROJECT = "6aefb23bdc9b295019a65fb1"
BASE = Path(__file__).resolve().parents[1] / "overleaf_jitt"


def upload(local, remote, folder_id="", binary=False):
    if binary:
        b64 = local.read_bytes().decode("latin-1")
        make_blob = (
            "const bin = atob(%s);"
            "const arr = new Uint8Array(bin.length);"
            "for (let i=0;i<bin.length;i++) arr[i]=bin.charCodeAt(i);"
            "const blob = new Blob([arr], {type:'application/pdf'});" % json.dumps(_b64(local))
        )
    else:
        make_blob = (
            "const blob = new Blob([%s], {type:'text/plain'});"
            % json.dumps(local.read_text(encoding="utf-8"))
        )

    expr = """(() => {
      window.__olres = 'pending';
      try {
      const csrf = (document.querySelector('meta[name=ol-csrfToken]')||{}).content;
      if (!csrf) { window.__olres = 'NOCSRF'; return 'nostart'; }
      %s
      const fd = new FormData();
      fd.append('name', %s);
      fd.append('type', 'file');
      %s
      fd.append('qqfile', blob, %s);
      fetch('/project/%s/upload?folder_id=%s', {
        method: 'POST',
        headers: {'X-CSRF-Token': csrf},
        credentials: 'same-origin',
        body: fd
      }).then(r => r.text().then(t => { window.__olres = r.status + ' ' + t.slice(0,200); }))
        .catch(e => { window.__olres = 'ERR ' + e; });
      } catch (e) { window.__olres = 'THROW ' + e; }
      return 'started';
    })()""" % (
        make_blob,
        json.dumps(remote),
        ("fd.append('folder_id', %s);" % json.dumps(folder_id)) if folder_id else "",
        json.dumps(remote),
        PROJECT,
        folder_id or "",
    )
    ol_js(expr)
    for _ in range(40):
        time.sleep(0.25)
        r = ol_js("window.__olres")
        if r and r != "pending":
            return r
    return "TIMEOUT"


def _b64(path):
    import base64
    return base64.b64encode(path.read_bytes()).decode()


if __name__ == "__main__":
    f = BASE / "main.tex"
    print(upload(f, "main.tex"))
