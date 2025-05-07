# %%
import json
from pathlib import Path

import streamlit as st

from rdagent.log.ui.conf import UI_SETTING
from rdagent.utils.repo.diff import generate_diff_from_dict

aide_path = UI_SETTING.aide_path
if not Path(aide_path).exists():
    st.error(f"Path {aide_path} does not exist, set it by `UI_AIDE_PATH`")
    st.stop()

jps = [str(i) for i in Path(aide_path).rglob("**/filtered_journal.json")]
jps = sorted(jps)
# st.write(jps)
left, right = st.columns([1, 4])
with left:
    default = 0
    ppp = f"{aide_path}/{st.query_params.get('jnp')}/logs/filtered_journal.json"
    if ppp in jps:
        default = jps.index(ppp)

    jnp = st.radio("Select Journal", options=jps, index=default, format_func=lambda x: str(x).split("/")[-3])
    jnp = Path(jnp)


with jnp.open("r") as f:
    d = json.load(f)
# with jnp_.open("r") as f:
#     d1 = json.load(f)

nm = {nd["id"]: nd for nd in d["nodes"]}
# %%
with right:
    st.header("AIDE trace", divider="rainbow")
    st.subheader(jnp)
    for c, p in d["node2parent"].items():
        f = nm[p]
        t = nm[c]
        df_lines = generate_diff_from_dict({"aide.py": f["code"]}, {"aide.py": t["code"]})

        with st.expander(f"Node {p} -> {c}"):
            st.markdown(f"## Parent ({f['metric']['value']}) Analysis")
            st.code(f["analysis"], wrap_lines=True)
            st.markdown(f"## Child ({t['metric']['value']}) Plan")
            st.code(t["plan"], wrap_lines=True)
            st.markdown("## Diff")
            st.code("".join(df_lines), language="diff", wrap_lines=True)
        # print("".join(df_lines))

# %%
