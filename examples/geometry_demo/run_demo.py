"""Self-drawn two-resistor example; Python standard library only.

Component terminals and candidate segments are supplied by the drawing, not
inferred by a detector. Only pixel continuity and wire grouping are computed.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from wire_join_continuity import existing_ink_join


class InkMask:
    """Minimal two-dimensional mask interface, avoiding a NumPy dependency."""
    def __init__(self, width=640, height=300):
        self.shape = (height, width)
        self.pixels = set()

    def __getitem__(self, index):
        y, x = index
        return (x, y) in self.pixels


def line_segment(direction, axis, start, end):
    return dict(direction=direction, axis=axis, start=start, end=end)


def endpoints(segment):
    axis, start, end = (segment[k] for k in ('axis', 'start', 'end'))
    return ((start, axis), (end, axis)) if segment['direction'] == 'horizontal' else ((axis, start), (axis, end))


def draw_mask(segments):
    mask = InkMask()
    for segment in segments:
        for position in range(segment['start'], segment['end'] + 1):
            point = (position, segment['axis']) if segment['direction'] == 'horizontal' else (segment['axis'], position)
            mask.pixels.add(point)
    return mask


def svg(segments, colors, caption):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="300" viewBox="0 0 640 300">', '<rect width="640" height="300" fill="white"/>', f'<text x="24" y="28" font-family="sans-serif" font-size="16">{caption}</text>']
    for index, segment in enumerate(segments):
        (x1,y1),(x2,y2) = endpoints(segment)
        parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{colors[index]}" stroke-width="3"/>')
        if caption != 'Self-drawn circuit: two parallel resistors':
            parts.append(f'<text x="{(x1+x2)/2+5}" y="{(y1+y2)/2-7}" font-family="sans-serif" font-size="12">s{index}</text>')
    for y, name in [(100,'R1 1k'),(200,'R2 2k')]:
        parts.append(f'<rect x="270" y="{y-12}" width="100" height="24" fill="white" stroke="#222" stroke-width="2"/><text x="280" y="{y-22}" font-family="sans-serif" font-size="15">{name}</text>')
    parts.append('<text x="24" y="275" font-family="sans-serif" font-size="13">Supplied terminals + wire candidates; computed pixel joins and nets</text></svg>')
    return ''.join(parts)


def main():
    output = Path(__file__).parent / 'output'
    output.mkdir(exist_ok=True)
    segments = [line_segment('horizontal',100,120,270), line_segment('horizontal',200,120,270), line_segment('vertical',120,100,200), line_segment('horizontal',100,370,520), line_segment('horizontal',200,370,520), line_segment('vertical',520,100,200)]
    mask = draw_mask(segments)
    parent = list(range(len(segments)))
    def root(index):
        while parent[index] != index:
            index = parent[index]
        return index
    joins = []
    # Pair proximity is required by the module contract; no long-range search.
    for i, a in enumerate(segments):
        for j in range(i+1,len(segments)):
            b = segments[j]
            distance = min(abs(x-u)+abs(y-v) for x,y in endpoints(a) for u,v in endpoints(b))
            if distance <= 2 and existing_ink_join(a,b,mask):
                parent[root(j)] = root(i)
                joins.append([i,j])
    groups = {}
    for i in range(len(segments)):
        groups.setdefault(root(i),[]).append(i)
    terminals = {0:'R1.1',1:'R2.1',3:'R1.2',4:'R2.2'}
    nets = [dict(id=f'N{i+1}',segments=members,terminals=[terminals[s] for s in members if s in terminals]) for i,members in enumerate(groups.values())]
    assert {frozenset(n['terminals']) for n in nets} == {frozenset(['R1.1','R2.1']),frozenset(['R1.2','R2.2'])}
    # Negative control: collinear segments separated by two blank pixels.
    gap = [line_segment('horizontal',50,10,30),line_segment('horizontal',50,33,60)]
    assert not existing_ink_join(*gap,draw_mask(gap))
    touching = [line_segment('horizontal',50,10,30),line_segment('vertical',30,50,80)]
    assert existing_ink_join(*touching,draw_mask(touching))
    palette = ['#1678b5','#d06b16']
    net_colors = {s:palette[i] for i,n in enumerate(nets) for s in n['segments']}
    for name,colors,title in [('input',['#444']*6,'Self-drawn circuit: two parallel resistors'),('geometry',['#6c56a5']*6,'Geometry candidates: six supplied segments'),('nets',[net_colors[i] for i in range(6)],'Computed networks: blue N1 / orange N2')]:
        (output/(name+'.svg')).write_text(svg(segments,colors,title),encoding='utf-8')
    result = dict(scope='Synthetic geometry module demo, not end-to-end detection or competition JSON',supplied_components=['R1','R2'],supplied_segments=segments,computed_joins=joins,nets=nets,checks=dict(two_expected_nets=True,blank_gap_rejected=True,corner_contact_accepted=True))
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['checks']))


if __name__ == '__main__':
    main()
