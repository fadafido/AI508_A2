"""Makes build/authors_code/, a patched copy of the authors' code (external/ stays untouched).

Each patch is the smallest change that gets the published code to run, and each one is described in
docs/issues.md. A diff of all the changes is written to patches/authors_code.diff.
"""
import difflib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "external" / "garralda-performance-model"
DST = ROOT / "build" / "authors_code" / "garralda_performance_model"

PATCHES = [
    # (file, old, new, issue id)
    ("algorithm.py", "import math\nimport array\n", "import math\nimport array\nfrom typing import Tuple  # [patch I-03] missing import, NameError on import\n", "I-03"),
    ("validation.py", "from performance.domain.model.garralda_perf_model import GarraldaPerformanceModel",
     "from algorithm import GarraldaPerformanceModel  # [patch I-04] module path did not exist in the repo", "I-04"),
    ("baselines/prats_perf_model.py", "\n       for train_index, test_index in logocv.split(X=self.w, groups=self.w_groups):",
     "\n        for train_index, test_index in logocv.split(X=self.w, groups=self.w_groups):  # [patch I-05] IndentationError", "I-05"),
    # I-06: the published Prats LOGOCV body can't run (wrong keyword arguments, and it predicts the whole test
    # group as one 3-D input). Replaced with what it was meant to do: one model per held-out group, trained on
    # [descriptor, input size, configuration] like the LOOCV branch, then one prediction per test sample.
    ("baselines/prats_perf_model.py",
     """                trained_model = self.perf_model.get_random_forest_regressor_model(
                    X=w_train,
                    y=t_train
                )

                # For each sample in test_data (workload group) predict the execution time
                for i in range(len(w_test)):
                    prediction = trained_model.predict([
                        np.concatenate(
                            (
                                w_test,
                                ids_test,
                                cs_test
                            ),
                            axis=1
                        )
                    ])
""",
     """                # [patch I-06] published call used wrong keyword arguments and predicted the whole group at once
                trained_model = self.perf_model.get_random_forest_regressor_model(
                    workload_descriptors=w_train,
                    input_data_sizes=ids_train,
                    configuration_settings=cs_train,
                    execution_times=t_train,
                )

                # For each sample in test_data (workload group) predict the execution time
                for i in range(len(w_test)):
                    prediction = trained_model.predict(
                        np.concatenate(
                            (
                                w_test[i:i + 1],
                                ids_test[i:i + 1],
                                cs_test[i:i + 1]
                            ),
                            axis=1
                        )
                    )
""", "I-06"),
    # I-09: scikit-learn >= 1.3 rejects max_features='auto'. On DS and DS' those grid points just fail and are
    # skipped, but with only 5 neighbours (Dou on DS'') every CV score is NaN, the refit falls on the first grid
    # point ('auto') and the whole run crashes. Dropping 'auto' leaves the other results unchanged.
    ("baselines/dou_perf_model.py", "            'rf__max_features': ['auto', 'sqrt']\n",
     "            'rf__max_features': ['sqrt']  # [patch I-09] 'auto' no longer exists in scikit-learn 1.3+\n", "I-09"),
    ("baselines/prats_perf_model.py", "            'rf__max_features': ['auto', 'sqrt']\n",
     "            'rf__max_features': ['sqrt']  # [patch I-09] 'auto' no longer exists in scikit-learn 1.3+\n", "I-09"),
]


def main():
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns(".git", "resources", "*.png"))
    (DST / "baselines" / "__init__.py").touch()
    diffs = []
    for rel, old, new, issue in PATCHES:
        p = DST / rel
        text = p.read_text()
        if text.count(old) != 1:
            sys.exit(f"patch {issue} on {rel}: expected exactly one match, found {text.count(old)}")
        new_text = text.replace(old, new)
        diffs += difflib.unified_diff(text.splitlines(True), new_text.splitlines(True),
                                      f"a/{rel}", f"b/{rel}", n=2)
        p.write_text(new_text)
        print(f"applied {issue} to {rel}")
    (ROOT / "patches" / "authors_code.diff").write_text("".join(diffs))


if __name__ == "__main__":
    main()
