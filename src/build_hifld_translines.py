"""Per-county transmission-line density from HIFLD, using envelope queries (no GIS stack)."""
import urllib.parse,urllib.request,json,math,time,sys
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
U="https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/US_Electric_Power_Transmission_Lines/FeatureServer/0/query"
cs=pd.read_csv(ROOT / 'data_static/county_static.csv',dtype={'fipsCode':str})
cs['fipsCode']=cs.fipsCode.str.zfill(5)
rows=[]
for i,r in cs.iterrows():
    lat,lon,sqmi=r.lat,r.lon,r.land_sqmi
    side=math.sqrt(sqmi); dlat=(side/69.0)/2; dlon=(side/(69.0*math.cos(math.radians(lat))))/2
    env={"xmin":lon-dlon,"ymin":lat-dlat,"xmax":lon+dlon,"ymax":lat+dlat,"spatialReference":{"wkid":4326}}
    stats=[{"statisticType":"sum","onStatisticField":"Shape__Length","outStatisticFieldName":"len_m"},
           {"statisticType":"count","onStatisticField":"OBJECTID","outStatisticFieldName":"n"},
           {"statisticType":"max","onStatisticField":"VOLTAGE","outStatisticFieldName":"maxkv"}]
    q={"f":"json","where":"1=1","geometry":json.dumps(env),"geometryType":"esriGeometryEnvelope",
       "inSR":"4326","spatialRel":"esriSpatialRelIntersects","outStatistics":json.dumps(stats),
       "returnGeometry":"false"}
    for attempt in range(3):
        try:
            req=urllib.request.Request(U+"?"+urllib.parse.urlencode(q),headers={"User-Agent":"Mozilla/5.0"})
            d=json.loads(urllib.request.urlopen(req,timeout=60).read().decode())
            a=d['features'][0]['attributes']; break
        except Exception as e:
            if attempt==2: a={'len_m':None,'n':None,'maxkv':None}; print('FAIL',r.fipsCode,e)
            time.sleep(2)
    # Shape__Length is Web-Mercator metres -> multiply by cos(lat) for true ground distance
    km=(a['len_m'] or 0)/1000.0*math.cos(math.radians(lat))
    rows.append(dict(fipsCode=r.fipsCode,tl_km=km,tl_segments=a['n'],tl_max_kv=a['maxkv'],
                     tl_km_per_1ksqmi=1000*km/sqmi))
    if i%50==0: print(i,r.fipsCode,round(km,1),flush=True)
out=pd.DataFrame(rows)
o=ROOT / 'data_static/eia861/county_transmission_hifld.csv'
out.to_csv(o,index=False); print('WROTE',o,out.shape)
print(out.describe().to_string())
print(out.head(6).to_string())
