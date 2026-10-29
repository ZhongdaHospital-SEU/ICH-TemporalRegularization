import json
import time
from collections import Counter
import numpy as np
import src.ich_dynamic.train_evaluate as te

config = json.loads(open('configs/study.json', encoding='utf-8').read())
dataset = te.load_dataset('mimic4')
splits, _ = te.build_repeated_splits(
    dataset.labels,
    dataset.cohort['subject_key'].astype(str).to_numpy(),
    folds=5,
    repeats=1,
    seed=int(config['random_seed']),
)
train_index, test_index = splits[0]
counts = Counter()
epoch_calls = Counter()

def fake_probs(n):
    base = np.tile(np.array([[0.60, 0.25, 0.15]], dtype=np.float64), (n, 1))
    offsets = (np.arange(n) % 3)
    base[np.arange(n), offsets] += 0.03
    base[np.arange(n), (offsets + 1) % 3] -= 0.03
    return base

def fake_train_temporal_model(train_seq, train_mask, train_static, train_y,
                              valid_seq, valid_mask, valid_static, valid_y,
                              *, epochs=30, **kwargs):
    counts['train_temporal_model'] += 1
    epoch_calls['train_temporal_model'] += int(epochs)
    p = fake_probs(len(valid_y))
    traj = np.zeros((len(valid_y), int(valid_seq.shape[1])), dtype=np.float32)
    hist = [{'epoch': 1, 'train_loss': 1.0, 'valid_loss': 1.0}]
    return p, traj, hist

def fake_fit_full_temporal_model(*args, epochs=30, **kwargs):
    counts['fit_full_temporal_model'] += 1
    epoch_calls['fit_full_temporal_model'] += int(epochs)
    return object()

def fake_predict_temporal_model(model, sequences, masks, static, **kwargs):
    return fake_probs(len(sequences)), np.zeros((len(sequences), 1), dtype=np.float32)

def fake_fit_xgboost(train_x, train_y, test_x, seed, sample_weight=None, parameters=None):
    counts['fit_xgboost'] += 1
    return fake_probs(len(test_x))

te.train_temporal_model = fake_train_temporal_model
te.fit_full_temporal_model = fake_fit_full_temporal_model
te.predict_temporal_model = fake_predict_temporal_model
te.fit_xgboost = fake_fit_xgboost

orig_fit_temporal = te.fit_temporal
def counted_fit_temporal(*args, **kwargs):
    name = kwargs.get('model_name', 'unknown')
    counts[f'fit_temporal:{name}'] += 1
    return orig_fit_temporal(*args, **kwargs)
te.fit_temporal = counted_fit_temporal

orig_member = te._fit_member_predictions
def counted_member(*args, **kwargs):
    name = kwargs.get('name', 'unknown')
    counts[f'member:{name}'] += 1
    return orig_member(*args, **kwargs)
te._fit_member_predictions = counted_member

start = time.time()
result = te._train_fold(
    dataset,
    0,
    train_index,
    test_index,
    25,
    int(config['random_seed']),
    te.ALL_MODEL_ORDER,
    float(config['lambda_monotone']),
    float(config['lambda_smooth']),
    tuple(float(x) for x in config['lambda_grid']),
    float(config['lambda_smooth_ratio']),
    te.MODEL_SEEDS,
    int(config['nested_inner_folds']),
    float(config['stacking_c']),
    tuple(config['stacking_members']),
    tuple(config['hyperparameters']['xgboost']['candidates']),
    tuple(config['hyperparameters']['sequence_models']['candidates']),
    np.ones(len(dataset.labels), dtype=np.float64),
)
print('elapsed_counter_only_seconds', round(time.time() - start, 2))
print('models', len(te.ALL_MODEL_ORDER))
for key in sorted(counts):
    print(f'{key}: {counts[key]}')
print('train_temporal_epoch_units', epoch_calls['train_temporal_model'])
print('fit_full_epoch_units', epoch_calls['fit_full_temporal_model'])
print('total_fit_calls', counts['train_temporal_model'] + counts['fit_full_temporal_model'] + counts['fit_xgboost'])
