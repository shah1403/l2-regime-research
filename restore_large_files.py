from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
for record in json.loads((root/'large_files.json').read_text()):
    out=root/record['path'];temporary=out.with_name(out.name+'.assembling')
    with temporary.open('wb') as f:
        for part in record['parts']:f.write((root/part).read_bytes())
    assert hashlib.file_digest(temporary.open('rb'),'sha256').hexdigest()==record['sha256']
    temporary.replace(out)
    print('Restored',record['path'])
