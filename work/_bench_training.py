import time, numpy as np, torch
from src.ich_dynamic.train_evaluate import load_dataset, model_sequence_inputs, model_static_frame
from src.ich_dynamic.models import SequencePreprocessor, train_temporal_model

ds = load_dataset("mimic4")
seq, msk = model_sequence_inputs(ds, "observed")
static = model_static_frame(ds, "all").to_numpy(dtype=np.float32)
n = len(ds.labels); idx = np.arange(n)
tr = idx[: int(n*0.8)]; va = idx[int(n*0.8):]
pre = SequencePreprocessor.fit(seq[tr], msk[tr], static[tr])
s_tr, m_tr = pre.transform_sequence(seq[tr], msk[tr])
s_va, m_va = pre.transform_sequence(seq[va], msk[va])
st_tr = pre.transform_static(static[tr]); st_va = pre.transform_static(static[va])
for name, graph, reg in [("gru", False, False), ("dggru", True, False), ("dggru_tr", True, True)]:
    t0 = time.time()
    p, traj, hist = train_temporal_model(
        s_tr, m_tr, st_tr, ds.labels[tr], s_va, m_va, st_va, ds.labels[va],
        use_graph=graph, use_regularization=reg, architecture="dggru" if graph else "gru",
        epochs=2, batch_size=128, learning_rate=1e-3, weight_decay=1e-4,
        hidden=64, dropout=0.2, seed=1, patience=6,
    )
    print(name, "2 epochs:", round(time.time()-t0,1), "s", "hist", hist)
