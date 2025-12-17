from pathlib import Path
import os

def load_names(chk_dir):
    # chk_dir = "../checkfiles/NC15_checkfile/"
    list_dir = os.listdir(chk_dir)
    pick = {}
    for f in list_dir:
        if "pychk" in f:
            p = Path(f)
            stem = p.stem
            is_rks = stem.endswith("-RKS")
            prefix = stem[:-4] if is_rks else stem
            key = (p.parent, prefix, p.suffix)  # 同目录+同前缀+同扩展名算一组

            if is_rks:
                pick[key] = f              # 有 -RKS 就覆盖为 -RKS
            else:
                pick.setdefault(key, f)    # 没有 -RKS 才保留原始版本

    kept_files = list(pick.values())
    return [chk_dir+i for i in kept_files]
