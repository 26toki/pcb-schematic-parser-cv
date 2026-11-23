"""Strict library/evidence gates for a later runtime integration."""
def unique_numbers(record):
    pool={}
    for p in record.get('pins',[]):
        name=p.get('name');number=str(p.get('number',''))
        if not isinstance(name,str) or not name or not number or any(ch in number for ch in '.,()'):continue
        pool.setdefault(name,set()).add(number)
    return {name:next(iter(numbers)) for name,numbers in pool.items() if len(numbers)==1}
def agreed_name(name,votes):
    return len(votes)==2 and all(len(v)==1 and v[0]['text']==name and v[0]['confidence']>=.9 for v in votes)
