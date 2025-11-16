import json
import argparse
from transformers import pipeline

def load_tests(path='tests/test_cases.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_classifier(model_name=None):
    # Try preferred models in order: model_name, phoBERT, multilingual distilbert, default
    candidates = []
    if model_name:
        candidates.append(model_name)
    candidates += ['vinai/phobert-base-v2', 'distilbert-base-multilingual-cased']
    last_exc = None
    for m in candidates:
        try:
            print(f'Trying model: {m}')
            clf = pipeline('sentiment-analysis', model=m)
            print(f'Loaded model: {m}')
            return clf
        except Exception as e:
            print(f'Failed to load {m}: {e}')
            last_exc = e
    print('Falling back to default pipeline model')
    return pipeline('sentiment-analysis')

def normalize_label(label, classifier=None):
    if label is None:
        return 'NEUTRAL'
    lab = label.upper()
    if 'POS' in lab:
        return 'POSITIVE'
    if 'NEG' in lab:
        return 'NEGATIVE'
    if 'NEU' in lab:
        return 'NEUTRAL'
    # handle LABEL_0 style using model config if available
    if classifier is not None:
        try:
            cfg = classifier.model.config
            if hasattr(cfg, 'id2label'):
                # try to map LABEL_X to a human label
                if lab.startswith('LABEL_'):
                    idx = int(lab.split('_')[-1])
                    mapped = cfg.id2label.get(idx)
                    if mapped:
                        return normalize_label(mapped, None)
        except Exception:
            pass
    return label

def run_tests(model_name=None, tests_path='tests/test_cases.json'):
    tests = load_tests(tests_path)
    clf = get_classifier(model_name)
    total = len(tests)
    correct = 0
    results = []
    for i, case in enumerate(tests, start=1):
        text = case.get('text')
        expected = case.get('sentiment')
        try:
            res = clf(text)
            if isinstance(res, list) and len(res) > 0:
                label = res[0].get('label')
                score = res[0].get('score', 0.0)
            else:
                label = None
                score = 0.0
        except Exception as e:
            label = None
            score = 0.0
            print(f'Error classifying case {i}: {e}')

        pred = normalize_label(label, clf)
        ok = (pred == expected)
        if ok:
            correct += 1
        results.append({'text': text, 'expected': expected, 'predicted': pred, 'score': float(score)})
        print(f'{i:02d}. "{text}" -> expected: {expected} ; predicted: {pred} (score={score:.3f})')

    acc = correct / total * 100 if total > 0 else 0.0
    print('\nSummary:')
    print(f'Correct: {correct}/{total}  Accuracy: {acc:.1f}%')
    return {'total': total, 'correct': correct, 'accuracy': acc, 'results': results}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Optional model name to use')
    parser.add_argument('--tests', type=str, default='tests/test_cases.json', help='Path to test cases JSON')
    args = parser.parse_args()
    run_tests(model_name=args.model, tests_path=args.tests)
