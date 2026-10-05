from energiewende import evaluation


def test_numbers_in_reads_german_and_english_style():
    numbers = evaluation.numbers_in("Im Mittel 168,29 EUR/MWh, insgesamt 1.168.081,41 MWh, Minimum -5.2")

    assert 168.29 in numbers
    assert 1168081.41 in numbers
    assert -5.2 in numbers


def test_number_found_allows_thousands_and_millions():
    assert evaluation.number_found(1168081.41, "Es wurden 1.168.081 MWh erzeugt.")
    assert evaluation.number_found(1168081.41, "Es wurden rund 1,17 Mio. MWh erzeugt.")
    assert evaluation.number_found(1168081.41, "Das sind 1.168 GWh.")


def test_number_found_rejects_wrong_numbers():
    assert not evaluation.number_found(168.29, "Der Preis lag bei 155,78 EUR/MWh.")


def test_lookup_walks_a_path():
    assert evaluation.lookup({"summary": {"max": {"value": 7}}}, "summary.max.value") == 7


def test_metrics_skip_missing_values():
    rows = [
        {"tools_ok": True, "hit": True, "number_ok": None, "text_ok": True, "correct": True, "latency_ms": 100, "tokens": 1000},
        {"tools_ok": False, "hit": None, "number_ok": False, "text_ok": None, "correct": False, "latency_ms": 300, "tokens": 3000},
    ]

    result = evaluation.metrics(rows)

    assert result == {
        "tool_accuracy": 0.5,
        "hit_rate": 1.0,
        "number_accuracy": 0.0,
        "text_accuracy": 1.0,
        "answer_accuracy": 0.5,
        "latency_ms": 200.0,
        "tokens": 2000.0,
    }
