#!/usr/bin/env python3
"""Export the FULL French-term inventory and within-entry candidate contrasts from OQLF.

No initial marker list is consulted; no geographic labels are inferred. Every
pair of distinct terms from the same GDT entry is exported for MANUAL review.
The officially recognized terms dataset is also exported, separately.

Optional --gdt-url retrieves a current live GDT page as a supplementary HTML
snapshot; these are NOT silently merged with the dated CSV or interpreted as
verified FI/FQ pairs.

Usage (from repository root):
  python scripts/extract_oqlf_markers.py
  python scripts/extract_oqlf_markers.py --benchmark data/qfrblima.csv \\
    --gdt-url https://vitrinelinguistique.oqlf.gouv.qc.ca/fiche-gdt/fiche/2077588/camion-a-ordures
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from html.parser import HTMLParser
from itertools import combinations

URLS = {
 'fiches': ('https://www.donneesquebec.ca/recherche/dataset/'
            '1c6567bf-8995-40b9-84a4-50faabae12f4/resource/'
            'c3ce0af4-7c0f-4dd2-b53a-6dc7fb3ea5ef/download/'
            'fiches_recentes_signees_oqlf_2026-01-19.csv'),
 'official': ('https://www.donneesquebec.ca/recherche/dataset/'
            '1c6567bf-8995-40b9-84a4-50faabae12f4/resource/'
            '882453c2-93c3-4204-b5ff-6d6297082ad9/download/'
            'termes_officialises_2026-01-14.csv'),
}
POS = re.compile(r'\s*\((?:n\.|adj\.|v\.|loc\.|adv\.|interj\.|pr[eé]p\.|pron\.|d[eé]term\.|abr\.|sigle|symb\.|nom|verbe)[^)]*\)', re.I)
LABELS = re.compile(r'\[([^\]]+)\]')
QC = re.compile(r'qu[eé]bec|canada', re.I)
EU = re.compile(r'france|europe', re.I)
INVENTORY_FIELDS = ['source_dataset','entry_id','source_url','term_index','term','geographic_labels','regional_class','other_labels','definition_OQLF','domain','date_fiche','source_terms_original']
PAIR_FIELDS = ['candidate_id','source_dataset','entry_id','source_url','term_a','term_b','region_a','region_b','evidence_a','evidence_b','pair_evidence','definition_OQLF','domain','date_fiche','source_terms_original','review_status']
OFFICIAL_FIELDS = ['source_dataset','entry_id','source_url','term','geographic_labels','region','domain','date_fiche','date_officialisation','type_officialisation','definition_OQLF','source_terms_original']


def norm(v):
    v=unicodedata.normalize('NFKD', str(v).casefold())
    v=''.join(ch for ch in v if not unicodedata.combining(ch))
    return re.sub(r'[^a-z0-9]+',' ',v).strip()


def parse_terms(s):
    results=[]
    for item in (s or '').split(';'):
        item=item.strip()
        if not item: continue
        labs=LABELS.findall(item)
        term=POS.sub('', LABELS.sub('',item)).strip(' ,.;')
        if not term: continue
        cls=geo('; '.join(labs))
        results.append({'term':term,'labels':'; '.join(labs),'region':cls})
    return results


def geo(label):
    qc=bool(QC.search(label));eu=bool(EU.search(label))
    return 'both' if qc and eu else 'QC' if qc else 'EU' if eu else 'unlabelled'


def rows(path, required):
    with path.open('r',encoding='utf-8-sig',newline='') as fp:
        reader=csv.DictReader(fp)
        missing=set(required)-set(reader.fieldnames or [])
        if missing:raise ValueError(f'{path}: missing {sorted(missing)}')
        yield from reader


def fetch(url, out, no_download=False):
    if out.is_file():return
    if no_download:raise FileNotFoundError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with urlopen(Request(url,headers={'User-Agent':'QFrBLiMa-reproducibility/1.0'}),timeout=120) as response:
        content=response.read()
    if len(content)<1000 or b'<html' in content[:400].lower() and out.suffix=='.csv':
        raise ValueError(f'Unexpected download for {url}')
    out.write_bytes(content)


def write_csv(path, fields, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)


def extract(fiches, official, output):
    inv=[];pair=[];off=[];n_f=n_o=0
    seen_terms=set();seen_pairs=set()
    for r in rows(fiches,['Termes_francais','URL_fiche','Definition','Domaines','Date_fiche']):
        n_f+=1
        url=r['URL_fiche'].strip();entry=re.search(r'/fiche/(\d+)',url)
        entry_id=entry.group(1) if entry else str(n_f)
        terms=[]; local_seen=set()
        for t in parse_terms(r['Termes_francais']):
            k=norm(t['term'])
            if not k or k in local_seen:continue
            local_seen.add(k);terms.append(t)
            key=(url,k)
            if key in seen_terms:continue
            seen_terms.add(key)
            inv.append({'source_dataset':'fiches','entry_id':entry_id,'source_url':url,
                'term_index':len(terms),'term':t['term'],'geographic_labels':t['labels'],
                'regional_class':t['region'],'other_labels':'',
                'definition_OQLF':r['Definition'],'domain':r['Domaines'],
                'date_fiche':r['Date_fiche'],'source_terms_original':r['Termes_francais']})
        for a,b in combinations(terms,2):
            key=(url,tuple(sorted((norm(a['term']),norm(b['term'])))))
            if key in seen_pairs:continue
            seen_pairs.add(key)
            regions={a['region'],b['region']}
            evidence=('explicit_QC_vs_EU' if regions=={'QC','EU'} else
                'QC_with_unlabelled_term' if regions=={'QC','unlabelled'} else
                'EU_with_unlabelled_term' if regions=={'EU','unlabelled'} else
                'other_same_entry_pair')
            pair.append({'candidate_id':f'GDT_C{len(pair)+1:06d}',
                'source_dataset':'fiches','entry_id':entry_id,'source_url':url,
                'term_a':a['term'],'term_b':b['term'],'region_a':a['region'],'region_b':b['region'],
                'evidence_a':a['labels'],'evidence_b':b['labels'],'pair_evidence':evidence,
                'definition_OQLF':r['Definition'],'domain':r['Domaines'],
                'date_fiche':r['Date_fiche'],'source_terms_original':r['Termes_francais'],
                'review_status':'not_manually_curated'})
    for r in rows(official,['Termes_Officialises','Type_Officialisation','URL_fiche','Definition','Domaines','Date_fiche']):
        n_o+=1
        url=r['URL_fiche'].strip();entry=re.search(r'/fiche/(\d+)',url)
        for t in parse_terms(r['Termes_Officialises']):
            off.append({'source_dataset':'official','entry_id':entry.group(1) if entry else str(n_o),
                'source_url':url,'term':t['term'],'geographic_labels':t['labels'],'region':t['region'],
                'domain':r['Domaines'],'date_fiche':r['Date_fiche'],
                'date_officialisation':r.get('Date_officialisation',''),
                'type_officialisation':r['Type_Officialisation'],'definition_OQLF':r['Definition'],
                'source_terms_original':r['Termes_Officialises']})
    write_csv(output/'oqlf_all_french_terms.csv',INVENTORY_FIELDS,inv)
    write_csv(output/'oqlf_all_same_entry_pairs.csv',PAIR_FIELDS,pair)
    write_csv(output/'oqlf_all_official_terms.csv',OFFICIAL_FIELDS,off)
    return {'fiches_rows':n_f,'official_rows':n_o,'all_french_terms':len(inv),
            'all_same_entry_candidate_pairs':len(pair),'official_term_rows':len(off),
            'evidence_counts':{k:sum(x['pair_evidence']==k for x in pair) for k in
                ['explicit_QC_vs_EU','QC_with_unlabelled_term','EU_with_unlabelled_term','other_same_entry_pair']}}


class VisibleText(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.hide=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style','noscript'):self.hide+=1
        if tag in ('h1','h2','h3','p','li','br'):self.parts.append('\n')
    def handle_endtag(self,tag):
        if tag in ('script','style','noscript') and self.hide:self.hide-=1
        if tag in ('h1','h2','h3','p','li'):self.parts.append('\n')
    def handle_data(self,data):
        if not self.hide:self.parts.append(data)


def pages(urls,folder,no_download):
    result=[];folder.mkdir(parents=True,exist_ok=True)
    for url in urls:
        parsed=urlparse(url)
        if parsed.scheme!='https' or parsed.hostname!='vitrinelinguistique.oqlf.gouv.qc.ca' or '/fiche-gdt/fiche/' not in parsed.path:
            raise ValueError(f'Not an official GDT page: {url}')
        m=re.search(r'/fiche/(\d+)',parsed.path)
        if not m: raise ValueError(url)
        ident=m.group(1);path=folder/f'{ident}.html'
        fetch(url,path,no_download)
        parser=VisibleText();parser.feed(path.read_text(encoding='utf-8',errors='replace'))
        cleaned='\n'.join(re.sub(r'\s+',' ',line).strip() for line in ''.join(parser.parts).splitlines() if line.strip())
        (folder/f'{ident}.txt').write_text(cleaned+'\n',encoding='utf-8')
        result.append({'entry_id':ident,'url':url,'snapshot_html':str(path),
                       'snapshot_text':str(folder/f'{ident}.txt'),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'status':'live_page_supplement_not_in_archived_CSV'})
    write_csv(folder/'live_gdt_pages_index.csv',list(result[0]) if result else ['entry_id','url','snapshot_html','snapshot_text','sha256','status'],result)
    return len(result)


def coverage(benchmark,inventory,pairs,dest):
    with inventory.open(encoding='utf-8') as f:
        index_terms={(r['source_url'],norm(r['term'])) for r in csv.DictReader(f)}
    with pairs.open(encoding='utf-8') as f:
        index_pairs={(r['source_url'],frozenset((norm(r['term_a']),norm(r['term_b'])))) for r in csv.DictReader(f)}
    with benchmark.open(encoding='utf-8-sig') as f:
        b={r['feature_id']:r for r in csv.DictReader(f)}
    out=[]
    for fid,r in sorted(b.items()):
        src=r['source'];a=norm(r['marker_FI']);z=norm(r['marker_FQ'])
        result=('both_terms_in_same_CSV_entry' if (src,frozenset((a,z))) in index_pairs else
                'both_terms_present_no_pair' if (src,a) in index_terms and (src,z) in index_terms else
                'one_term_present_in_CSV' if (src,a) in index_terms or (src,z) in index_terms else
                'neither_term_in_CSV_entry')
        out.append({'feature_id':fid,'FI':r['marker_FI'],'FQ':r['marker_FQ'],'source_url':src,'status':result})
    write_csv(dest,['feature_id','FI','FQ','source_url','status'],out)
    return {s:sum(r['status']==s for r in out) for s in sorted({r['status'] for r in out})}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fiches',type=Path,default=Path('data/raw/fiches_recentes_signees_oqlf_2026-01-19.csv'))
    ap.add_argument('--official',type=Path,default=Path('data/raw/termes_officialises_2026-01-14.csv'))
    ap.add_argument('--output-dir',type=Path,default=Path('data/intermediate'))
    ap.add_argument('--benchmark',type=Path,help='Optional audit only; never used to generate candidates')
    ap.add_argument('--gdt-url',action='append',default=[],help='Supplementary current GDT page URL; repeat as needed')
    ap.add_argument('--no-download',action='store_true')
    a=ap.parse_args()
    fetch(URLS['fiches'],a.fiches,a.no_download);fetch(URLS['official'],a.official,a.no_download)
    stats=extract(a.fiches,a.official,a.output_dir)
    if a.gdt_url:stats['live_pages_downloaded']=pages(a.gdt_url,a.output_dir/'live_gdt_pages',a.no_download)
    if a.benchmark:stats['benchmark_coverage']=coverage(a.benchmark,
        a.output_dir/'oqlf_all_french_terms.csv',a.output_dir/'oqlf_all_same_entry_pairs.csv',
        a.output_dir/'benchmark_coverage.csv')
    data={'source_urls':URLS,'source_sha256':{'fiches':hashlib.sha256(a.fiches.read_bytes()).hexdigest(),
           'official':hashlib.sha256(a.official.read_bytes()).hexdigest()},'counts':stats,
          'methodology':'Full same-entry inventory; NO FI/FQ inference for unlabelled terms; manual curation required.',
          'note':'Live GDT pages may differ from the 2026-01-19 archive and are kept separate.'}
    (a.output_dir/'oqlf_extraction.metadata.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
