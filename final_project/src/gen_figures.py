import pandas as pd
from pathlib import Path

p = Path("../reports")
accuracy_report_paths = p.glob("**/*nn*")
nn_dirs = [path for path in accuracy_report_paths if path.is_dir()]
dfs = []
for nn_dir in nn_dirs:
    with open(nn_dir / Path("desc.txt"), "r") as f:
        get_arg = lambda s: s.strip().split("=")
        opts = {k: v for k, v in map(get_arg, f.readlines())}
        df = pd.read_csv(nn_dir / Path("accuracy.csv"), index_col=0)
        opts["df"] = df
        ax = df.plot(title=f"{opts['loss_func']}, shape={opts['shapes']}, lr={opts['lr']}")
        ax.figure.savefig(nn_dir / Path("fig.pdf"))
