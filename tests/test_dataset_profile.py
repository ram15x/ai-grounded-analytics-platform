import pandas as pd
import pytest

from grounded_analytics.datasets import profile as P


def col(values, name="c"):
    return P.profile_column(pd.Series(values, dtype=object), name, name)


def test_safe_names_are_ascii_snake_unique_and_start_with_a_letter():
    assert P.safe_names(["Order ID", "order id", " 2024 Sales ", "", "Ünït Price"]) == [
        "order_id", "order_id_2", "c_2024_sales", "column_4", "n_t_price",
    ]  # fmt: skip


def test_types():
    assert col(["1", "2", "3"]).type == "integer"
    assert col(["1.5", "2", "3,000.25"]).type == "decimal"
    assert col(["yes", "no", "yes"]).type == "boolean"
    assert col(["2024-01-05", "2024-02-06"]).type == "date"
    assert col(["2024-01-05 10:30", "2024-02-06 11:00"]).type == "datetime"
    assert col(["a", "b", "c"]).type == "text"
    assert col([None, "", "  "]).type == "empty"


def test_numbers_in_text_columns_are_flagged_not_coerced():
    c = col(["1", "2", "3", "x"])
    assert c.type == "text" and any(i.startswith("mixed_numbers_and_text") for i in c.issues)


def test_day_first_is_detected_and_used():
    c = col(["03/01/2024", "15/01/2024"])
    assert c.type == "date" and (c.min, c.max) == ("2024-01-03", "2024-01-15")


def test_ambiguous_and_conflicting_date_orders_are_reported():
    assert any("ambiguous" in i for i in col(["03/01/2024", "04/02/2024"]).issues)
    assert any("conflict" in i for i in col(["15/01/2024", "01/15/2024"]).issues)


def test_numeric_stats_and_counts():
    c = col(["-2", "0", "4"])
    assert (c.negatives, c.zeros, c.min, c.max) == (1, 1, "-2", "4")


def test_hints_are_hints_and_personal_data_is_flagged():
    c = col(["a@x.com", "b@y.org", "c@z.io"], "contact")
    assert (
        any(h.startswith("possible_personal_data") for h in c.hints) and "category" not in c.hints
    )
    assert "identifier" in col([str(i) for i in range(50)], "order_id").hints


def test_frame_profile_counts_duplicates_and_renames():
    df = pd.DataFrame({"Order ID": ["1", "1", "2"], "Amt": ["5", "5", "6"]})
    p = P.profile_frame(df, "f.csv")
    assert p.duplicate_rows == 1 and p.rows == 3
    assert any("column names were made safe" in i for i in p.issues)


def test_read_table_csv_xlsx_and_refusals(tmp_path):
    (tmp_path / "a.csv").write_text("A,B\n1,x\n2,y\n", encoding="utf-8")
    assert list(P.read_table(tmp_path / "a.csv").columns) == ["A", "B"]
    pd.DataFrame({"A": [1, 2]}).to_excel(tmp_path / "b.xlsx", index=False)
    assert P.profile_file(tmp_path / "b.xlsx").columns[0].type == "integer"
    with pytest.raises(P.ProfileError):
        P.read_table(tmp_path / "missing.csv")
    (tmp_path / "c.pdf").write_bytes(b"%PDF")
    with pytest.raises(P.ProfileError, match="unsupported"):
        P.read_table(tmp_path / "c.pdf")
    with pytest.raises(P.ProfileError):
        P.profile_frame(pd.DataFrame())


def test_latin1_csv_is_read(tmp_path):
    (tmp_path / "d.csv").write_bytes("name\ncafé\n".encode("cp1252"))
    assert P.read_table(tmp_path / "d.csv")["name"][0] == "café"


def test_category_hint_needs_few_distinct_values_and_ids_are_not_categories():
    many = col([str(i % 400) for i in range(4000)], "quantity")
    assert "category" not in many.hints
    assert "category" in col(["N", "S", "E", "N", "S", "E"], "region").hints
    assert "identifier" in col([str(i) for i in range(60)], "invoiceno").hints
