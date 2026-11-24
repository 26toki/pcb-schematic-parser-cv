"""Require observed local ink before joining geometrically close wire segments."""
from collections import deque
import math

def existing_ink_join(a, b, mask):
    """Return whether observed pixels connect two nearby axis-aligned segments.

Segments use direction, axis, start and end in image pixels. mask is a
2-D boolean/ink array indexed as mask[y, x]. The caller must first choose
geometrically nearby pairs: this function is a local evidence check, not
a global net tracer. It neither fills gaps nor changes the input mask.

An eight-neighbor search tolerates diagonal rasterization. It cannot decide
whether a crossing represents an electrical junction; that is upstream policy."""
    if a['direction'] == b['direction']:
        low = max(a['start'], b['start'])
        high = min(a['end'], b['end'])
        position_a = (low + high) / 2 if low <= high else min(a['end'], max(a['start'], b['start']))
        position_b = min(b['end'], max(b['start'], position_a))
        point_a = [position_a, a['axis']] if a['direction'] == 'horizontal' else [a['axis'], position_a]
        point_b = [position_b, b['axis']] if b['direction'] == 'horizontal' else [b['axis'], position_b]
    else:
        horizontal_segment, vertical_segment = (a, b) if a['direction'] == 'horizontal' else (b, a)
        x, y = (vertical_segment['axis'], horizontal_segment['axis'])
        point_a = [min(horizontal_segment['end'], max(horizontal_segment['start'], x)), y]
        point_b = [x, min(vertical_segment['end'], max(vertical_segment['start'], y))]
        a, b = (horizontal_segment, vertical_segment)
    left = max(0, math.floor(min(point_a[0], point_b[0])) - 2)
    right = min(mask.shape[1] - 1, math.ceil(max(point_a[0], point_b[0])) + 2)
    top = max(0, math.floor(min(point_a[1], point_b[1])) - 2)
    bottom = min(mask.shape[0] - 1, math.ceil(max(point_a[1], point_b[1])) + 2)

    def support(point, segment):
        x, y = map(round, point)
        horizontal = segment['direction'] == 'horizontal'
        return {(neighbor_x, neighbor_y) for neighbor_y in range(max(top, y - 1), min(bottom, y + 1) + 1) for neighbor_x in range(max(left, x - 1), min(right, x + 1) + 1) if mask[neighbor_y, neighbor_x] and segment['start'] <= (neighbor_x if horizontal else neighbor_y) <= segment['end'] and (abs((neighbor_y if horizontal else neighbor_x) - segment['axis']) <= 1.5)}
    start_pixels = support(point_a, a)
    goal_pixels = support(point_b, b)
    if not start_pixels or not goal_pixels:
        return False
    seen = set(start_pixels)
    queue = deque(start_pixels)
    while queue:
        x, y = queue.popleft()
        if (x, y) in goal_pixels:
            return True
        for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
            neighbor_x, neighbor_y = (x + dx, y + dy)
            if left <= neighbor_x <= right and top <= neighbor_y <= bottom and ((neighbor_x, neighbor_y) not in seen) and mask[neighbor_y, neighbor_x]:
                seen.add((neighbor_x, neighbor_y))
                queue.append((neighbor_x, neighbor_y))
    return False
