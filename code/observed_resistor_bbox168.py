"""Use measured bodies only for fully evidenced rectangle recovery."""
import math
from large_resistor161 import choose_value


def use_observed_body(c, size):
    if not c.get('accepted') or c.get('type') != 'r':
        return False
    if c.get('geometry_source') != 'observed_rectangle_with_inline_leads':
        return False
    evidence = c.get('large_resistor_recovery', {})
    refs = evidence.get('reference_views', [])
    if len(refs) != 2 or not c.get('designator'):
        return False
    if any(r.get('text') != c['designator'] or not .97 <= r.get('confidence', 0) <= 1 for r in refs):
        return False
    if c.get('value') is None or choose_value(evidence.get('value_views', [])) != c['value']:
        return False
    b = c.get('body_bbox_xyxy', [])
    if len(b) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in b):
        return False
    if not (0 <= b[0] < b[2] <= size[0] and 0 <= b[1] < b[3] <= size[1]):
        return False
    w, h = b[2]-b[0], b[3]-b[1]
    if not (31 < min(w, h) <= 60 and max(w, h) <= 200):
        return False
    if c.get('axis') == 'horizontal' and w > h:
        expected = [[b[0], (b[1]+b[3])/2], [b[2], (b[1]+b[3])/2]]
    elif c.get('axis') == 'vertical' and h > w:
        expected = [[(b[0]+b[2])/2, b[1]], [(b[0]+b[2])/2, b[3]]]
    else:
        return False
    contacts = c.get('terminal_contacts_xy', [])
    return len(contacts) == 2 and all(len(p) == 2 and math.dist(p, q) <= 1 for p, q in zip(contacts, expected))
