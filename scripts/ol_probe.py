import json, sys, time
sys.path.insert(0, 'scripts')
from ol import ol_js
from pathlib import Path

PID = "6aefb23bdc9b295019a65fb1"
CANDS = ["6aefb23cdc9b295019a66012", "6aefb23bdc9b295019a65fbe",
         "6aefb23bdc9b295019a65fd8", "6aefb23bdc9b295019a65fee",
         "6aefb23cdc9b295019a66010", "6aefb23bdc9b295019a65fbf"]

for fid in CANDS:
    expr = """(()=>{window.__r='pending';try{
      const csrf=(document.querySelector('meta[name=ol-csrfToken]')||{}).content;
      const fd=new FormData();
      fd.append('name','probe.tex');fd.append('type','file');
      fd.append('qqfile',new Blob(['x'],{type:'text/plain'}),'probe.tex');
      fetch('/project/%s/upload?folder_id=%s',{method:'POST',headers:{'X-CSRF-Token':csrf},body:fd})
        .then(r=>r.text().then(t=>window.__r=r.status+' '+t.slice(0,80)))
        .catch(e=>window.__r='ERR '+e);
      }catch(e){window.__r='THROW '+e}return 'go'})()""" % (PID, fid)
    ol_js(expr)
    for _ in range(40):
        time.sleep(0.25)
        r = ol_js("window.__r")
        if r and r != "pending":
            print(fid, "->", r)
            break
