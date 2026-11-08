"""Upload the full LaTeX package into the Overleaf project, into the right folders.

Root folder id was discovered by probing /project/<id>/upload?folder_id=...
The project is the stock sn-jnl template, so bst/ and figures/ must exist as
folders first; existing template files are replaced where the name matches.
"""
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ol import ol_js  # noqa: E402

PID = "6aefb23bdc9b295019a65fb1"
ROOT = "6aefb23cdc9b295019a66012"
BASE = Path(__file__).resolve().parents[1] / "overleaf_jitt"


def _wait(key="window.__r", tries=60):
    for _ in range(tries):
        time.sleep(0.25)
        r = ol_js(key)
        if r and r != "pending":
            return r
    return "TIMEOUT"


def mkdir(name, parent):
    expr = """(()=>{window.__r='pending';
      const csrf=(document.querySelector('meta[name=ol-csrfToken]')||{}).content;
      fetch('/project/%s/folder',{method:'POST',
        headers:{'X-CSRF-Token':csrf,'Content-Type':'application/json'},
        body:JSON.stringify({name:%s,parent_folder_id:%s})})
        .then(r=>r.text().then(t=>window.__r=r.status+' '+t.slice(0,150)))
        .catch(e=>window.__r='ERR '+e);
      return 'go'})()""" % (PID, json.dumps(name), json.dumps(parent))
    ol_js(expr)
    return _wait()


def upload(local, remote, folder, binary=False):
    if binary:
        b64 = base64.b64encode(Path(local).read_bytes()).decode()
        mkblob = ("const bin=atob(%s);const arr=new Uint8Array(bin.length);"
                  "for(let i=0;i<bin.length;i++)arr[i]=bin.charCodeAt(i);"
                  "const blob=new Blob([arr],{type:'application/pdf'});" % json.dumps(b64))
    else:
        text = Path(local).read_text(encoding="utf-8", errors="replace")
        mkblob = ("const blob=new Blob([%s],{type:'text/plain'});"
                  % json.dumps(text))
    expr = """(()=>{window.__r='pending';try{
      const csrf=(document.querySelector('meta[name=ol-csrfToken]')||{}).content;
      %s
      const fd=new FormData();
      fd.append('name',%s);fd.append('type','file');
      fd.append('qqfile',blob,%s);
      fetch('/project/%s/upload?folder_id=%s',{method:'POST',
        headers:{'X-CSRF-Token':csrf},body:fd})
        .then(r=>r.text().then(t=>window.__r=r.status+' '+t.slice(0,120)))
        .catch(e=>window.__r='ERR '+e);
      }catch(e){window.__r='THROW '+e}return 'go'})()""" % (
        mkblob, json.dumps(remote), json.dumps(remote), PID, folder)
    ol_js(expr)
    return _wait()


if __name__ == "__main__":
    print("mkdir bst     :", mkdir("bst", ROOT))
    print("mkdir figures :", mkdir("figures", ROOT))
