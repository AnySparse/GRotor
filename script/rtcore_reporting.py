"""Validate contiguous RT-Core wall intervals and stage dispatch."""
import math
import re


def parse_spatial_settings(text):
    rows, armed = [], False
    for line in text.splitlines():
        if line.startswith(("RTCORE_STAGE ", "RTCORE_POLICY ")):armed = True
        elif armed and line.startswith("RTCORE_SPACE "):
            fields = dict(token.split("=", 1) for token in line.split()[1:])
            rows.append(dict(spatial_model=fields.get("model", "layered"),
                             point_model=fields.get("point_model", "cross"),
                             z_gap=float(fields["z_gap"]), margin=float(fields["margin"])))
            armed = False
    if len(rows) != 2:raise RuntimeError("Missing spatial settings for the two routing stages")
    return rows


def _records(text, name):
    result = []
    for line in re.findall(r"^" + name + r" (.+)$", text, re.MULTILINE):
        fields = {}
        for token in line.split():
            key, value = token.split("=", 1)
            if key in fields:raise RuntimeError("Duplicate timing field")
            fields[key] = value
        result.append(fields)
    if len(result) != 2:raise RuntimeError("Missing or duplicated " + name + " records")
    return result


def parse_totals(text, batches):
    policies, records = (_records(text, name) for name in ("RTCORE_STAGE", "RTCORE_TIMING"))
    rows = []
    for stage, policy, record, batch in zip(("lshape", "flt"), policies, records, batches):
        row = dict(method=record["method"], design_nets=int(policy["design_nets"]),
                   nets=int(policy["input_nets"]), timing_scope=record["scope"],
                   elapsed_ms=float(record["elapsed_ms"]), start_us=batch["start_us"], end_us=batch["end_us"])
        expected = "cached" if stage == "flt" else "rounds"
        scope = "pipeline_to_coloring_wall" if expected == "cached" else "coloring_loop_wall"
        span = (row["end_us"] - row["start_us"]) / 1000.0
        if (row["method"] != expected or policy["method"] != expected
                or policy["stage"] != stage
                or row["nets"] != batch["nets"] or row["design_nets"] < row["nets"]
                or row["nets"] < 0 or row["timing_scope"] != scope
                or not math.isfinite(row["elapsed_ms"]) or row["elapsed_ms"] < 0
                or abs(row["elapsed_ms"] - batch["elapsed_ms"]) > .001
                or abs(row["elapsed_ms"] - span) > .001
                or (row["nets"] > 0 and row["elapsed_ms"] <= 0)):
            raise RuntimeError("Invalid contiguous timestamp span, stage dispatch or timing scope")
        rows.append(row)
    preparations = _records(text, "BATCH_PREPARATION")
    for row, fields in zip(rows, preparations):
        row.update({k:float(fields[k]) for k in ("preparation_ms", "collection_ms", "call_ms")})
        if (fields["method"] != row["method"]
                or any(not math.isfinite(row[k]) or row[k] < 0 for k in ("preparation_ms", "collection_ms", "call_ms"))
                or abs(row["call_ms"]-row["preparation_ms"]-row["elapsed_ms"]-row["collection_ms"]) > .005):
            raise RuntimeError("Invalid external batching preparation/collection timing")
    if rows[0]["design_nets"] != rows[1]["design_nets"]:
        raise RuntimeError("Routing database size changed between stages")
    return rows
