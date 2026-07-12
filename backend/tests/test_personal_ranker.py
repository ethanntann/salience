from salience_api.ranking.personal_ranker import train_personal_ranker


def test_ranker_returns_zero_without_enough_feedback():
    model = train_personal_ranker([], [])

    assert model.predict([[0.8, 0.8, 0.1, 1.0, 0.8]]) == [0.0]


def test_ranker_learns_positive_preference():
    features = [
        [0.9, 0.8, 0.1, 1.0, 0.9],
        [0.1, 0.0, 0.9, 0.2, 0.1],
        [0.8, 0.7, 0.2, 1.0, 0.8],
        [0.2, 0.1, 0.8, 0.2, 0.2],
    ]
    labels = [1.0, -1.0, 0.8, -0.8]

    model = train_personal_ranker(features, labels)

    assert model.predict([[0.85, 0.8, 0.1, 1.0, 0.85]])[0] > 0.5
    assert model.predict([[0.1, 0.0, 0.95, 0.2, 0.1]])[0] < 0.5


def test_ranker_responds_to_single_positive_signal():
    model = train_personal_ranker([[0.9, 0.8, 0.1, 1.0, 0.9]], [1.0])

    assert model.predict([[0.88, 0.78, 0.12, 1.0, 0.88]])[0] > 0.9


def test_ranker_returns_zero_for_exact_match_to_single_negative_signal():
    features = [0.9, 0.8, 0.1, 1.0, 0.9]
    model = train_personal_ranker([features], [-1.0])

    assert model.predict([features]) == [0.0]


def test_ranker_uses_individual_teacher_features():
    features = [
        [0.4, 0.4, 0.2, 0.8, 0.4, 0.0, 0.0, 1.0, 0.0],
        [0.4, 0.4, 0.2, 0.8, 0.4, 0.0, 0.0, 0.0, 1.0],
    ]
    labels = [1.0, -1.0]

    model = train_personal_ranker(features, labels)

    assert model.predict([[0.4, 0.4, 0.2, 0.8, 0.4, 0.0, 0.0, 1.0, 0.0]])[0] > 0.5
    assert model.predict([[0.4, 0.4, 0.2, 0.8, 0.4, 0.0, 0.0, 0.0, 1.0]])[0] < 0.5
