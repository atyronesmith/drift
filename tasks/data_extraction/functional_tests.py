"""
Functional equivalence tests for the data_extraction task.

Tests the baseline extractor's JSON output against the known correct
values from the commercial invoice. All 12 line items, financial totals,
and structural completeness are validated.
"""

import json


def extract(code: str):
    """Parse the JSON output from the LLM."""
    # Strip markdown fences if present
    text = code.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # skip ```json or ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


# --- Known correct values ---

EXPECTED_LINE_ITEMS = [
    {"line_number": 1, "part_number": "HMS-4420", "quantity": 12, "unit_price": 184.50, "total": 2214.00},
    {"line_number": 2, "part_number": "BRG-6205-2RS", "quantity": 24, "unit_price": 38.75, "total": 930.00},
    {"line_number": 3, "part_number": "GKT-7712", "quantity": 8, "unit_price": 67.20, "total": 537.60},
    {"line_number": 4, "part_number": "FTG-NPT-075", "quantity": 50, "unit_price": 12.30, "total": 615.00},
    {"line_number": 5, "part_number": "LUB-SYNTH-5", "quantity": 6, "unit_price": 94.00, "total": 564.00},
    {"line_number": 6, "part_number": "CTL-PLC-800", "quantity": 2, "unit_price": 2475.00, "total": 4950.00},
    {"line_number": 7, "part_number": "SNR-PT100-6", "quantity": 10, "unit_price": 145.00, "total": 1450.00},
    {"line_number": 8, "part_number": "CBL-SHL-18-4", "quantity": 500, "unit_price": 2.85, "total": 1425.00},
    {"line_number": 9, "part_number": "VLV-BFY-4", "quantity": 4, "unit_price": 312.00, "total": 1248.00},
    {"line_number": 10, "part_number": "PNL-HMI-10", "quantity": 2, "unit_price": 1890.00, "total": 3780.00},
    {"line_number": 11, "part_number": "HMS-4420-KIT", "quantity": 6, "unit_price": 42.00, "total": 252.00},
    {"line_number": 12, "part_number": "DIN-RAIL-1M", "quantity": 20, "unit_price": 8.50, "total": 170.00},
]


# --- Tests ---

def test_line_item_count(data):
    items = data.get("line_items", [])
    assert len(items) == 12, f"Expected 12 line items, got {len(items)}"


def test_part_numbers(data):
    items = data.get("line_items", [])
    extracted = [item["part_number"] for item in items]
    expected = [li["part_number"] for li in EXPECTED_LINE_ITEMS]
    assert extracted == expected, f"Part numbers mismatch: {extracted}"


def test_quantities(data):
    items = data.get("line_items", [])
    for item, expected in zip(items, EXPECTED_LINE_ITEMS):
        assert item["quantity"] == expected["quantity"], (
            f"Line {expected['line_number']} qty: expected {expected['quantity']}, got {item['quantity']}"
        )


def test_unit_prices(data):
    items = data.get("line_items", [])
    for item, expected in zip(items, EXPECTED_LINE_ITEMS):
        assert abs(item["unit_price"] - expected["unit_price"]) < 0.01, (
            f"Line {expected['line_number']} price: expected {expected['unit_price']}, got {item['unit_price']}"
        )


def test_line_totals(data):
    items = data.get("line_items", [])
    for item, expected in zip(items, EXPECTED_LINE_ITEMS):
        assert abs(item["total"] - expected["total"]) < 0.01, (
            f"Line {expected['line_number']} total: expected {expected['total']}, got {item['total']}"
        )


def test_line_math(data):
    """Verify qty * unit_price == total for each line."""
    items = data.get("line_items", [])
    for item in items:
        computed = item["quantity"] * item["unit_price"]
        assert abs(computed - item["total"]) < 0.01, (
            f"Line {item.get('line_number')}: {item['quantity']} * {item['unit_price']} = {computed} != {item['total']}"
        )


def test_subtotal(data):
    assert abs(data["subtotal"] - 18135.60) < 0.01, f"Subtotal: expected 18135.60, got {data['subtotal']}"


def test_subtotal_matches_line_sum(data):
    """Verify subtotal == sum of line item totals."""
    items = data.get("line_items", [])
    computed = sum(item["total"] for item in items)
    assert abs(computed - data["subtotal"]) < 0.01, (
        f"Subtotal mismatch: sum of lines = {computed}, subtotal = {data['subtotal']}"
    )


def test_invoice_total(data):
    assert abs(data["invoice_total"] - 18988.70) < 0.01, (
        f"Invoice total: expected 18988.70, got {data['invoice_total']}"
    )


def test_amount_due(data):
    assert abs(data["amount_due"] - 18238.70) < 0.01, (
        f"Amount due: expected 18238.70, got {data['amount_due']}"
    )


def test_seller_buyer(data):
    assert data["seller"]["name"] == "Precision Industrial Supply Co.", f"Seller name wrong: {data['seller']['name']}"
    assert data["buyer"]["name"] == "Cascade Manufacturing Ltd.", f"Buyer name wrong: {data['buyer']['name']}"
    assert data["seller"]["ein"] == "74-3218956", f"Seller EIN wrong"
    assert data["buyer"]["ein"] == "93-1785420", f"Buyer EIN wrong"


def test_invoice_metadata(data):
    meta = data.get("invoice_metadata", {})
    assert meta.get("number") == "PIS-2026-03841", f"Invoice number wrong: {meta.get('number')}"
    assert meta.get("po_number") == "CM-PO-77214", f"PO number wrong: {meta.get('po_number')}"
    assert meta.get("terms") == "Net 30", f"Terms wrong: {meta.get('terms')}"
    assert meta.get("currency") == "USD", f"Currency wrong: {meta.get('currency')}"


tests = {
    "line_item_count": test_line_item_count,
    "part_numbers": test_part_numbers,
    "quantities": test_quantities,
    "unit_prices": test_unit_prices,
    "line_totals": test_line_totals,
    "line_math": test_line_math,
    "subtotal": test_subtotal,
    "subtotal_matches_line_sum": test_subtotal_matches_line_sum,
    "invoice_total": test_invoice_total,
    "amount_due": test_amount_due,
    "seller_buyer": test_seller_buyer,
    "invoice_metadata": test_invoice_metadata,
}
