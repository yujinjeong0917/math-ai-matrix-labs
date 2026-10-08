"""4장 최소 구현: 내용으로 이름을 정하는(content-addressed) 이미지 빌더를 파이썬 표준 라이브러리로 흉내 낸다.

Docker 데몬 없이 원리만 보인다. 실제 Docker/BuildKit의 바이트·다이제스트와는 다르다.
- 레이어 = 그 단계에서 바뀐 파일들을 정해진 규칙으로 묶은 tar 바이트. 레이어 다이제스트 = sha256(tar 바이트).
- 이미지 다이제스트 = sha256(설정 JSON). 설정에는 레이어 다이제스트 목록, ENV, USER, ENTRYPOINT, 생성 시각이 들어간다.
- 캐시 키 = (앞 단계까지의 다이제스트, 명령 문자열, COPY라면 복사한 파일 내용의 체크섬).
  Docker 문서대로 COPY는 파일 내용을 보고(수정 시각은 보지 않음), RUN은 명령 문자열만 본다.
- 레지스트리의 태그는 바뀔 수 있고(mutable), 다이제스트로 가리킨 이미지는 내용이 같을 때만 같다.
"""

import hashlib
import io
import json
import re
import shlex
import tarfile


def sha256(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


# ---------- Dockerfile 읽기 ----------

def parse_dockerfile(text):
    """(명령, 인자 문자열) 목록. 줄 끝 역슬래시 이어 쓰기와 주석을 처리한다."""
    lines, buf = [], ""
    for raw in text.splitlines():
        s = raw.strip()
        if not buf and (not s or s.startswith("#")):
            continue
        if s.endswith("\\"):
            buf += s[:-1].strip() + " "
            continue
        buf += s
        op, _, rest = buf.partition(" ")
        lines.append((op.upper(), rest.strip()))
        buf = ""
    return lines


def lint_dockerfile(text):
    """이 장이 지키기로 한 규칙을 검사하고, 어긴 항목 목록을 돌려준다(빈 목록이면 통과)."""
    ins = parse_dockerfile(text)
    ops = [op for op, _ in ins]
    problems = []
    froms = [a for op, a in ins if op == "FROM"]
    if not froms or "@sha256:" not in froms[0]:
        problems.append("베이스 이미지를 다이제스트(@sha256:...)로 고정하지 않았다")
    copies = [(i, shlex.split(a)[:-1]) for i, (op, a) in enumerate(ins) if op == "COPY"]
    lock_i = next((i for i, srcs in copies if "requirements.lock" in srcs or "." in srcs), None)
    src_i = next((i for i, srcs in copies if any(s != "requirements.lock" for s in srcs)), None)
    run_install = next((i for i, (op, a) in enumerate(ins) if op == "RUN" and "requirements.lock" in a), None)
    if lock_i is None or run_install is None:
        problems.append("lock 파일로 설치하지 않는다")
    elif src_i is not None and not (lock_i < run_install < src_i):
        problems.append("의존성 설치가 소스 복사보다 뒤에 있다(소스를 고칠 때마다 다시 설치)")
    if run_install is not None and "--require-hashes" not in ins[run_install][1]:
        problems.append("설치할 때 파일 해시를 확인하지 않는다(--require-hashes)")
    users = [a for op, a in ins if op == "USER"]
    if not users or users[-1].split(":")[0] in ("root", "0"):
        problems.append("마지막 USER가 root다")
    ep = [a for op, a in ins if op == "ENTRYPOINT"]
    if not ep or not ep[-1].startswith("["):
        problems.append("ENTRYPOINT가 exec 형식(JSON 배열)이 아니다")
    if "ADD" in ops:
        problems.append("ADD 대신 COPY를 쓴다")
    return problems


# ---------- 레이어와 레지스트리 ----------

def tar_layer(files, epoch=None):
    """files: {경로: (바이트, 수정시각)}. 경로 순서·소유자·권한을 고정한다. epoch가 있으면 모든 수정시각을 그 값으로."""
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for path in sorted(files):
            data, mtime = files[path]
            ti = tarfile.TarInfo(path)
            ti.size, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0o644, 0, 0, "", ""
            ti.mtime = int(epoch if epoch is not None else mtime)
            tf.addfile(ti, io.BytesIO(data))
    return bio.getvalue()


class Registry:
    def __init__(self):
        self.tags, self.images = {}, {}

    def push(self, name_tag, image):
        self.images[image["digest"]] = image
        self.tags[name_tag] = image["digest"]
        return image["digest"]

    def resolve(self, ref):
        """'name:tag' 이면 지금 태그가 가리키는 다이제스트, 'name:tag@sha256:...' 이면 그 다이제스트(없으면 오류)."""
        if "@" in ref:
            digest = ref.split("@", 1)[1]
            if digest not in self.images:
                raise KeyError(f"레지스트리에 없는 다이제스트: {digest[:19]}")
            return digest
        return self.tags[ref]


def make_base(name, files, created):
    """베이스 이미지 하나(레이어 1개)."""
    layer = tar_layer(files)
    config = {"layers": [sha256(layer)], "env": {}, "user": "root", "entrypoint": None, "created": created, "name": name}
    return {"digest": sha256(canonical(config)), "config": config, "fs": dict(files), "layer_sizes": [len(layer)]}


# ---------- 빌드 ----------

def build(dockerfile, context, registry, runners, cache=None, now=0, epoch=None):
    """dockerfile 문자열을 한 줄씩 실행해 이미지를 만든다.

    context: {경로: (바이트, 수정시각)}  빌드할 때 보내는 파일들
    runners: {RUN 명령 문자열: fn(fs, context) -> {경로: 바이트}}  RUN이 실제로 하는 일
    cache: 딕셔너리(빌드 사이에 재사용). None이면 캐시 없이 빌드.
    now: 빌드 시각(정수), epoch: SOURCE_DATE_EPOCH처럼 고정할 시각
    반환: {"digest", "config", "fs", "log": [(명령, "cache"|"run"|"copy"|"pull", 레이어 바이트)]}
    """
    use_cache = cache is not None
    cache = cache if use_cache else {}
    ins = parse_dockerfile(dockerfile)
    op, ref = ins[0]
    assert op == "FROM", "첫 줄은 FROM"
    base = registry.images[registry.resolve(ref)]
    fs, layers, sizes = dict(base["fs"]), list(base["config"]["layers"]), list(base["layer_sizes"])
    env, user, entry = dict(base["config"]["env"]), base["config"]["user"], base["config"]["entrypoint"]
    chain = base["digest"]
    log = [(f"FROM {ref}", "pull", 0)]
    for op, arg in ins[1:]:
        text = f"{op} {arg}"
        if op in ("ENV", "USER", "ENTRYPOINT", "WORKDIR", "CMD", "LABEL", "ARG"):
            if op == "ENV":
                k, _, v = arg.partition("=")
                env[k.strip()] = v.strip()
            elif op == "USER":
                user = arg
            elif op == "ENTRYPOINT":
                entry = json.loads(arg) if arg.startswith("[") else arg
            chain = sha256(canonical([chain, text]))
            log.append((text, "meta", 0))
            continue
        if op == "COPY":
            srcs, dest = shlex.split(arg)[:-1], shlex.split(arg)[-1]
            picked = {p: context[p] for p in sorted(context) if any(p == s or s == "." or p.startswith(s.rstrip("/") + "/") for s in srcs)}
            checksum = sha256(canonical({p: hashlib.sha256(d).hexdigest() for p, (d, _) in picked.items()}))  # 수정시각은 보지 않는다
            key = sha256(canonical([chain, text, checksum]))
            new = {dest.rstrip("/") + "/" + p: (d, m) for p, (d, m) in picked.items()}
        elif op == "RUN":
            key = sha256(canonical([chain, text]))  # 명령 문자열만 본다
            new = None
        else:
            raise ValueError(f"지원하지 않는 명령: {op}")
        if use_cache and key in cache:
            layer_digest, size, files = cache[key]
            how = "cache"
        else:
            if new is None:
                written = runners[arg](fs, context)
                new = {p: (d, now) for p, d in written.items()}
                how = "run"
            else:
                how = "copy"
            layer = tar_layer(new, epoch=epoch)
            layer_digest, size, files = sha256(layer), len(layer), new
            cache[key] = (layer_digest, size, files)
        fs.update(files)
        layers.append(layer_digest)
        sizes.append(size)
        chain = key
        log.append((text, how, 0 if how == "cache" else size))
    config = {"layers": layers, "env": env, "user": user, "entrypoint": entry,
              "created": epoch if epoch is not None else now, "base": base["digest"]}
    return {"digest": sha256(canonical(config)), "config": config, "fs": fs, "log": log, "layer_sizes": sizes}


def installed_versions(fs):
    """이미지 파일 시스템에서 site-packages/<이름>-<버전>.dist-info 를 찾아 {이름: 버전}."""
    out = {}
    for p in fs:
        m = re.search(r"site-packages/([a-z0-9\-]+)-([^/]+)\.dist-info/", p)
        if m:
            out[m.group(1)] = m.group(2)
    return out
