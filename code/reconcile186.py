"""Replace only merged low-confidence OCR names using observed numeric pairs."""
def reconcile(pairs, numeric):
    by_number={p['number']:p for p in numeric}
    result=[];events=[]
    for old in pairs:
        new=by_number.get(old['number'])
        fragments=old.get('pinname','').split()
        eligible=(new is not None and new.get('numeric_connector_evidence') is not None
                  and old.get('library_status')!='exact_match'
                  and old.get('name_confidence',1)<.85
                  and len(fragments)>1 and all(x.isdigit() for x in fragments)
                  and old.get('side')==new.get('side'))
        if eligible:
            result.append(dict(new,text_source='observed_numeric_replaces_merged_low_confidence_name'))
            events.append(dict(number=old['number'],before=old['pinname'],after=new['pinname']))
        else:result.append(old)
    return result,events
