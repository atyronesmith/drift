Extract all structured data from this invoice into JSON. Include: seller, buyer (with name, address, phone, fax, ein, contact_name, contact_email), invoice metadata (number, date, due_date, terms, po_number, ship_date, ship_via, fob, currency), line_items (array with line_number, part_number, description, quantity, unit, unit_price, total), subtotal, discounts (array with description and amount), taxes (array with description, rate, amount), fees (array with description and amount), credits (array with description and amount), invoice_total, amount_due, payment_info, and notes.

{input}
