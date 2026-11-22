"""Compile a constrained drawing question and calculate its answer from page facts."""
import re

from shapely.geometry import Point,Polygon

from .geometry import center
from .openings import touches_scope
from .reader import AREA,DOOR,ROOM_CODE,clean,evidence
from .schedule import printed_property, CERTIFIED, MARK
from .phrasing import alternative


def compile_question(question):
    q=clean(question).strip(' ?.!')
    floor=None
    if re.search(r'\s+(?:im erdgeschoss|on the ground floor)$',q,re.I):
        q=re.sub(r'\s+(?:im erdgeschoss|on the ground floor)$','',q,flags=re.I)
        floor='ground'
    door_ids = list(MARK.finditer(q))
    mark = door_ids[0] if len(door_ids) == 1 else None
    def program(op,**args): return dict(op=op,args=args,**({'floor':floor} if floor else {}))
    if len(door_ids) > 1 or re.search(r'\b(?:except|excluding|other than|außer|ausgenommen|greater than|at least|mindestens|'
                                       r'less than|höchstens|should|must|muss|müssen|required|vorgeschrieben)\b',q,re.I):
        return program('unsupported')
    if mark and (re.search(r'\d', MARK.sub('',q)) or
                 re.search(r'\b(?:first floor|upper floor|basement|obergeschoss|kellergeschoss|in inches|in feet)\b',q,re.I)):
        return program('unsupported')
    if mark and re.search(r'fire[- ]resistance|fire rating|feuerwiderstand|brandwiderstand',q,re.I):
        p = program('fire_rating',mark=mark.group().upper())
        if CERTIFIED.search(q):
            p['certified_only'] = True
        return p
    if mark and re.search(r'as[- ]built|aufgemessen|aufmaß|field[- ]measured|tatsächlich gemessen',q,re.I) and re.search(r'width|breite',q,re.I):
        p = program('as_built_width',mark=mark.group().upper())
        if re.search(r'\b(?:cm|centimet(?:er|re)s?|zentimetern?)\b',q,re.I):
            p['output_unit'] = 'cm'
        elif re.search(r'\b(?:in m|in met(?:er|re)s?|in metern?)\b',q,re.I):
            p['output_unit'] = 'm'
        return p
    if mark and re.fullmatch(r'(?:where is|locate|find|wo (?:befindet sich|ist))\s+(?:(?:the|die)\s+)?(?:door|tür)\s+(?:marked |mit (?:der )?kennung )?'+re.escape(mark.group())+r'(?: on this sheet| auf diesem (?:plan|blatt))?',q,re.I):
        return program('locate_mark',mark=mark.group().upper())
    match=re.fullmatch(r'(?:which (?:door(?:s| identifiers)?) (?:directly )?connect(?:s)?|welche türen? verbinde[nt])\s+(?:room )?(.+?)\s+(?:and|und|mit)\s+(?:room )?(.+)',q,re.I)
    if match:
        return program('adjacent_doors',rooms=[re.sub(r'^(?:den|dem|das|der|die)\s+','',x.strip(),flags=re.I).upper() for x in match.groups()])
    match=re.fullmatch(r'(?:how many (?:labeled |labelled )?doors (?:directly )?connect to|wie viele (?:beschriftete )?türen führen (?:direkt )?zu)\s+(?:room )?(.+)',q,re.I)
    if match:return program('room_door_count',room=match.group(1).upper())
    if re.fullmatch(r'(?:how many (?:distinct |different )?door (?:marks|identifiers|tags) are printed on (?:this|the) (?:floor plan|sheet|plan)|wie viele (?:verschiedene )?türkennungen sind auf diesem (?:blatt|plan) (?:aufgedruckt|gedruckt))',q,re.I):
        return program('count_marks')
    if re.fullmatch(r'(?:what is the project title (?:printed|shown) on this (?:sheet|plan)|wie lautet der projekttitel auf diesem (?:blatt|plan))',q,re.I):
        return program('project_title')
    room=ROOM_CODE.search(q)
    if room and re.fullmatch(r'(?:what floor area is printed for room|what is the printed (?:floor )?area of room|welche (?:grund)?fläche ist für raum)\s+'+re.escape(room.group())+r'(?:,? in m(?:²|2)| angegeben)?',q,re.I):
        return program('printed_area',room=room.group().upper())
    op, args = alternative(q)
    return program(op, **args)


def _title(scene):
    rows=scene['lines']
    headings=[]
    for i,line in enumerate(rows):
        text=clean(line['text'])
        match=re.fullmatch(r'(?:project(?: title)?|projekt(?:titel)?)\s*[:\-]\s*(.+)',text,re.I)
        if match:
            return match.group(1),[evidence('project_title',line['bbox'],text,f'text-{i}')]
        if re.fullmatch(r'project(?: title)?|projekt(?:titel)?',text,re.I):
            headings.append((i,line))
    if len(headings)!=1:return None,[]
    i,label=headings[0]
    box=label['bbox'];height=box[3]-box[1]
    candidates=[]
    for j,row in enumerate(rows):
        if i==j:continue
        b=row['bbox']
        gap=b[1]-box[3]
        aligned=abs(b[0]-box[0])<=3*height
        right=0<=b[0]-box[2]<5*height and abs(center(b)[1]-center(box)[1])<.7*height
        if right or (aligned and -.2*height<=gap<4*height):
            candidates.append((0 if right else max(0,gap)+height,j,row))
    candidates.sort(key=lambda x:x[0])
    if not candidates or (len(candidates)>1 and candidates[1][0]-candidates[0][0]<.25*height):
        return None,[]
    _,j,value=candidates[0]
    return clean(value['text']),[evidence('project_label',box,label['text'],f'text-{i}'),
                                 evidence('project_title',value['bbox'],value['text'],f'text-{j}')]


def _room_evidence(scene,name):
    r=scene['rooms'].get(name)
    if not r:return []
    items=[r['label']]
    if 'polygon' in r:
        items.append(dict(evidence('room_polygon',r['bbox'],name,'boundary-'+r['label']['id']),polygon=r['polygon']))
    return items


def _mark_evidence(scene,mark):
    result=scene['marks'].get(mark,[])[:]
    frame=scene['frames'].get(mark)
    if frame:
        item=evidence('door_frame',frame['bbox'],mark,'frame-'+mark)
        for key in ('jamb_centers','wall_support_points','opening_profile','swing_evidence','status'):
            if key in frame:
                item[key]=frame[key]
        result.append(item)
    return result


def _area(scene,room):
    info=scene['rooms'].get(room)
    if not info:return None,[]
    outline=Polygon(info['polygon']) if info.get('polygon') else None
    a=info['label']['bbox'];h=a[3]-a[1]
    candidates=[]
    for i,line in enumerate(scene['lines']):
        text=clean(line['text'])
        match=AREA.search(text)
        if not match:continue
        b=line['bbox']
        explicit=re.search(r'(?<!\w)'+re.escape(room)+r'(?!\w)',text,re.I)
        nearby=(-.25*h<=b[1]-a[3]<=2.5*h and min(abs(b[0]-a[0]),abs(center(b)[0]-center(a)[0]))<=3*h)
        contained=outline is not None and outline.covers(Point(center(b)))
        if explicit or (nearby and (contained or outline is None)) or contained:
            candidates.append((0 if explicit else 1 if nearby else 2,i,line,match))
    if not candidates:return None,[]
    priority=min(item[0] for item in candidates)
    items=[item for item in candidates if item[0]==priority]
    if len(items)!=1:return None,[]
    _,i,line,match=items[0]
    source=evidence('printed_area',line['bbox'],line['text'],f'text-{i}')
    return float(match.group(1).replace(',','.')),[info['label'],source]


def answer_program(scene,program,arm='relations'):
    if arm not in ('facts','relations'):
        raise ValueError('Expected facts or relations arm')
    op=program.get('op') if isinstance(program,dict) else None
    args=program.get('args',{}) if isinstance(program,dict) else {}
    result={key:scene[key] for key in ('page','width','height','coordinates','source_sha256')}
    result.update(program=program,status='unresolved',value=None,unit=None,evidence=[],calculation=None,reason='')
    def emit(status='unresolved',value=None,items=(),calculation=None,reason='',unit=None):
        deduplicated={e['id']:e for e in items}
        if floor_anchor and status in ('answered','partial'):
            deduplicated[floor_anchor['id']]=floor_anchor
        result.update(status=status,value=value,unit=unit,evidence=list(deduplicated.values()),calculation=calculation,reason=reason)
        return result
    floor_anchor=None
    if isinstance(program,dict) and 'floor' in program:
        titles=[(i,line) for i,line in enumerate(scene['lines']) if re.fullmatch(
            r'(?:(?:floor plan|grundriss)\s*[-:]?\s*)?(?:ground floor|erdgeschoss|rez-de-chaussée)(?:\s*[-:]?\s*(?:floor plan|grundriss))?',
            clean(line['text']),re.I)]
        if program['floor']!='ground' or len(titles)!=1:
            return emit(reason='The requested floor could not be uniquely verified from the printed sheet title.')
        i,line=titles[0]
        floor_anchor=evidence('floor_title',line['bbox'],line['text'],f'text-{i}')
    expected={'count_marks':set(),'project_title':set(),'printed_area':{'room'},'room_door_count':{'room'},
              'locate_mark':{'mark'},'fire_rating':{'mark'},'as_built_width':{'mark'},'adjacent_doors':{'rooms'}}
    if not isinstance(op,str) or op not in expected or not isinstance(args,dict) or set(args)!=expected[op]:
        return emit(reason='Unsupported or incomplete question program.')
    if 'rooms' in args and (not isinstance(args['rooms'],list) or len(args['rooms'])!=2 or
                            any(not isinstance(s,str) or not s.strip() for s in args['rooms'])):
        return emit(reason='Two distinct named rooms are required.')
    if any(not isinstance(v,str) or not v.strip() for k,v in args.items() if k!='rooms'):
        return emit(reason='A room or literal door identifier is required.')
    mark=args.get('mark','').upper()
    if mark and not DOOR.fullmatch(mark):return emit(reason='Invalid door identifier.')
    if op in ('fire_rating','as_built_width'):
        value, anchors, state = printed_property(scene, op, mark, certified=program.get('certified_only') is True)
        unit = program.get('output_unit', 'mm') if op == 'as_built_width' else 'min'
        factors = {'min':1, 'mm':1, 'cm':10, 'm':1000}
        if not isinstance(unit, str) or (op == 'as_built_width' and unit not in ('mm','cm','m')):
            return emit(reason='Unsupported width unit.')
        if state == 'documented':
            amount = value / factors[unit]
            return emit('answered', int(amount) if amount == int(amount) else float(amount), anchors,
                        'Read the door-specific printed schedule entry and express it in '+unit+'.', unit=unit)
        if state == 'ambiguous':
            return emit(items=anchors, reason='The door-specific entries are ambiguous, qualified or in conflict.')
        reason=('The sheet supplies no verified door-specific certified fire-resistance rating in minutes.' if op=='fire_rating'
                else 'No on-site as-built clear opening width is documented for this door. A drawn frame measures a different quantity.')
        return emit(items=_mark_evidence(scene,mark), reason=reason)
    if op=='locate_mark':
        items=_mark_evidence(scene,mark)
        if len(scene['marks'].get(mark,[]))==1:
            return emit('answered',mark,items,'Locate the printed door identifier in page coordinates.')
        return emit(reason='Door identifier missing or occurs in more than one location.')
    if op=='count_marks':
        items=[item for mark in sorted(scene['marks']) for item in scene['marks'][mark]]
        n=len(scene['marks'])
        if not n:
            return emit(reason='No positive door-mark evidence; absence is unconfirmed.')
        if scene['native_text_complete']:
            return emit('answered',n,items,'Count distinct visible native door identifiers on this page.','', 'count')
        return emit('partial',n,items,'Count distinct confirmed OCR door identifiers.','Raster text coverage is incomplete.','count')
    if op=='project_title':
        value,items=_title(scene)
        if value:return emit('answered',value,items,'Read the title printed next to the Project field.')
        return emit(reason='A unique visible project-title field and its value are missing.')
    room=args.get('room','').upper()
    if op=='printed_area':
        value,items=_area(scene,room)
        if value is not None:return emit('answered',value,items,'Read the printed room area and its square-metre unit.',unit='m²')
        return emit(reason='A unique printed square-metre quantity associated with the requested room is missing.')
    if arm!='relations':
        return emit(reason='Room-to-opening relations are unavailable in the facts-only reader.')
    names=([s.upper() for s in args['rooms']] if op=='adjacent_doors' else [room])
    if len(set(names))!=len(names) or any(n not in scene['rooms'] for n in names):
        return emit(reason='The requested distinct named rooms could not all be verified.')
    target=set(names)
    matched=sorted(m for m,rel in scene['relations'].items() if target<=set(rel['rooms']))
    unresolved=[]
    excluded={}
    polygons=[Polygon(scene['rooms'][n]['polygon']) for n in names if 'polygon' in scene['rooms'][n]]
    for m in scene['marks']:
        if m in scene['relations']:continue
        frame=scene['frames'].get(m) or scene.get('located_profiles',{}).get(m)
        if frame:
            # A bounded physical opening outside a requested room excludes this mark.
            # An unattached text label could be connected to its door by a long leader.
            if not touches_scope(frame,polygons):
                excluded[m]={'basis':frame['status'],'bbox':frame['bbox'],'jamb_centers':frame['jamb_centers']}
                continue
        unresolved.append(m)
    items=[item for n in names for item in _room_evidence(scene,n)]
    items+=[item for m in matched for item in _mark_evidence(scene,m)]
    exact=scene['native_text_complete'] and not unresolved and bool(scene['marks'])
    if not matched:
        return emit(items=items,reason='Door inventory or room-boundary contacts are incomplete in the requested scope.')
    value=matched if op=='adjacent_doors' else len(matched)
    result['relations']={m:scene['relations'][m] for m in matched}
    if excluded:
        result['outside_scope']=excluded
    for mark in matched:
        for name,(x,y) in result['relations'][mark].get('probes',{}).items():
            radius=max(.2,min(scene['width'],scene['height'])*.001)
            b=[max(0,x-radius),max(0,y-radius),min(scene['width'],x+radius),min(scene['height'],y+radius)]
            probe=evidence('room_probe',b,name,'probe-'+mark+'-'+name)
            probe.update(room=name,mark=mark,point=[x,y])
            items.append(probe)
    return emit('answered' if exact else 'partial',value,items,
                'Select distinct labelled openings with opposite-side contacts in the requested room(s).',
                '' if exact else 'Further printed openings may be unresolved in this scope.',
                'count' if op=='room_door_count' else None)


def answer(scene,question,arm='relations'):
    return answer_program(scene,compile_question(question),arm)
