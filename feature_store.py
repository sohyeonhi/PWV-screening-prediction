# src/feature_store.py
import os, json, hashlib, datetime, re
from typing import Tuple, List, Dict, Optional

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(SRC_DIR, "..", "artifacts", "features")

DEFAULT_NAME = "selected_features.json"
DEFAULT_PREFIX = "selected_features"

def _ensure_dir() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

def _fingerprint(features: List[str]) -> str:
    feats = sorted(list(dict.fromkeys(features)))
    return hashlib.md5(",".join(feats).encode()).hexdigest()[:8]

def feature_artifact_path(name: str) -> str:
    _ensure_dir()
    return os.path.join(ARTIFACT_DIR, name)

def save_feature_list(features: List[str], meta: Optional[Dict] = None, name: str = DEFAULT_NAME) -> str:
    feats = sorted(list(dict.fromkeys(features)))
    payload = {
        "features": feats,
        "meta": {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "count": len(feats),
            "fingerprint": _fingerprint(feats),
            **(meta or {})
        },
    }
    path = feature_artifact_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def save_feature_list_versioned(features: List[str], meta: Optional[Dict] = None,
                                prefix: str = DEFAULT_PREFIX, also_update_alias: bool = True) -> Tuple[str, Optional[str]]:
    """버전 파일을 저장하고, 옵션에 따라 alias(selected_features.json)도 함께 갱신."""
    feats = sorted(list(dict.fromkeys(features)))
    fp = _fingerprint(feats)
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    ver_name = f"{prefix}_{date_str}_{fp}.json"
    ver_path = save_feature_list(feats, meta=meta, name=ver_name)

    alias_path = None
    if also_update_alias:
        # 최신 alias도 갱신
        alias_path = save_feature_list(feats, meta=meta, name=DEFAULT_NAME)
    return ver_path, alias_path

def _find_latest_with_prefix(prefix: str = DEFAULT_PREFIX) -> Optional[str]:
    _ensure_dir()
    pat = re.compile(rf"^{re.escape(prefix)}_\d{{8}}_[0-9a-f]{{8}}\.json$")
    candidates = [f for f in os.listdir(ARTIFACT_DIR) if pat.match(f)]
    if not candidates:
        return None
    # 파일 수정시각 기준 최신
    candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(ARTIFACT_DIR, fn)), reverse=True)
    return candidates[0]

def load_feature_list(name: Optional[str] = None) -> Tuple[List[str], Dict]:
    """
    name이 None이면:
      1) alias(selected_features.json)가 있으면 그것을 로드
      2) 없으면 prefix 최신 버전 파일을 자동 탐색하여 로드
    특정 버전을 지정하고 싶으면 name에 파일명을 넘겨라.
    """
    # 1) alias 우선
    alias_path = feature_artifact_path(DEFAULT_NAME)
    target = None
    if name:
        target = feature_artifact_path(name)
    elif os.path.exists(alias_path):
        target = alias_path
    else:
        latest = _find_latest_with_prefix()
        if latest:
            target = feature_artifact_path(latest)

    if not target or not os.path.exists(target):
        raise FileNotFoundError(
            f"선정 피처 파일이 없습니다. 먼저 `python -m src.feature_step`를 실행해 저장하세요.\n"
            f"찾은 경로: {target or '(없음)'}"
        )

    with open(target, encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("features", []), obj.get("meta", {})
