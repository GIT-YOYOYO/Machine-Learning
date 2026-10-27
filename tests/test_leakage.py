from src.evaluation.leakage import compare_feature_selection_leakage


def test_feature_selection_leakage_inflates_score():
    leak, correct = compare_feature_selection_leakage(
        n_samples=60, n_features=200, n_selected=10, random_state=0
    )

    # 纯噪声数据上，正确写法接近随机猜测；泄露写法会明显虚高
    assert 0.4 <= correct <= 0.7
    assert leak > correct + 0.1