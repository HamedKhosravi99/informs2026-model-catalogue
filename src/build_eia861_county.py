"""Build county-level EIA-861 utility/infrastructure features for IN/OH/PA/WV."""
import pandas as pd, numpy as np, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
D='/tmp/eia861/y2024/'
STATES=['IN','OH','PA','WV']

# ---- 1. county FIPS crosswalk (Census 2020 national county file) ----
fips=pd.read_csv('/tmp/county_fips.txt',sep='|',dtype=str)
fips=fips[fips.STATE.isin(STATES)].copy()
fips['fipsCode']=(fips.STATEFP+fips.COUNTYFP)
def norm(s):
    s=str(s).upper().strip()
    for suf in [' COUNTY',' PARISH',' CITY AND BOROUGH',' BOROUGH',' CENSUS AREA',' MUNICIPIO']:
        if s.endswith(suf): s=s[:-len(suf)]
    return s.replace('.','').replace("'",'').replace('-',' ').replace('  ',' ').strip()
fips['ckey']=fips.COUNTYNAME.map(norm)
xw=fips.set_index(['STATE','ckey'])['fipsCode'].to_dict()
print('crosswalk counties:',len(xw))

# ---- 2. service territory: utility x state x county ----
ter=pd.read_excel(D+'Service_Territory_2024.xlsx',sheet_name='Counties_States',header=0)
ter.columns=[str(c).strip() for c in ter.columns]
ter=ter[ter.State.isin(STATES)].copy()
ter['ckey']=ter.County.map(norm)
ter['fipsCode']=[xw.get((s,c)) for s,c in zip(ter.State,ter.ckey)]
miss=ter[ter.fipsCode.isna()]
print('territory rows:',len(ter),'unmatched county names:',len(miss), miss.County.unique()[:10])
ter=ter.dropna(subset=['fipsCode'])

# ---- 3. county population weights (from our existing static file) ----
cs=pd.read_csv(ROOT / 'data_static/county_static.csv',dtype={'fipsCode':str})
cs['fipsCode']=cs.fipsCode.str.zfill(5)
pop=cs.set_index('fipsCode')['population'].to_dict()
ter['pop']=ter.fipsCode.map(pop).fillna(10000.0)
# share of a utility's state-level totals assigned to each county, weighted by county population
ter['w']=ter.groupby(['Utility Number','State'])['pop'].transform(lambda x: x/x.sum())

# ---- 4. customers + ownership (Sales to Ultimate Customers) ----
s=pd.read_excel(D+'Sales_Ult_Cust_2024.xlsx',sheet_name='States',header=2)
s.columns=[str(c).strip().replace('\n',' ') for c in s.columns]
s=s[s.State.isin(STATES) & s['Service Type'].isin(['Bundled','Delivery'])].copy()
num=lambda c: pd.to_numeric(s[c],errors='coerce')
s['cust_tot']=num('Count.4'); s['cust_res']=num('Count'); s['sales_mwh']=num('Megawatthours.4')
s=s.groupby(['Utility Number','State','Ownership'],as_index=False)[['cust_tot','cust_res','sales_mwh']].sum()

# ---- 4b. EIA-861S short-form filers (NOT in Sales_Ult_Cust) -> union in ----
sf=pd.read_excel(D+'Short_Form_2024.xlsx',sheet_name='861S',header=0)
sf.columns=[str(c).strip() for c in sf.columns]
sf=sf[sf.State.isin(STATES)].copy()
sf['cust_tot']=pd.to_numeric(sf['Total Customers'],errors='coerce')
sf['cust_res']=np.nan
sf['sales_mwh']=pd.to_numeric(sf['Total Sales (MWh)'],errors='coerce')
sf=sf.rename(columns={'Ownership':'Ownership'})[['Utility Number','State','Ownership','cust_tot','cust_res','sales_mwh']]
sf=sf.groupby(['Utility Number','State','Ownership'],as_index=False).sum(min_count=1)
sf=sf[~sf.set_index(['Utility Number','State']).index.isin(s.set_index(['Utility Number','State']).index)]
print('short-form utilities added:',len(sf),'customers:',sf.cust_tot.sum())
s=pd.concat([s,sf],ignore_index=True)

# ---- 5. distribution circuits ----
dist=pd.read_excel(D+'Distribution_Systems_2024.xlsx',sheet_name='Distribution_Systems_States',header=0)
dist.columns=[str(c).strip() for c in dist.columns]
dist=dist[dist.State.isin(STATES)].copy()
dist['circuits']=pd.to_numeric(dist['Distribution Circuits'],errors='coerce')
dist=dist.groupby(['Utility Number','State'],as_index=False)['circuits'].sum()

# ---- 6. reliability SAIDI/SAIFI (IEEE standard) ----
rel=pd.read_excel(D+'Reliability_2024.xlsx',sheet_name='Reliability_States',header=2)
rel.columns=[str(c).strip() for c in rel.columns]
rel=rel[rel.State.isin(STATES)].copy()
ren={'SAIDI (minutes per year)':'saidi_with_med','SAIFI (times per year)':'saifi_with_med',
     'SAIDI (minutes per year).1':'saidi_no_med','SAIFI (times per year).1':'saifi_no_med'}
rel=rel.rename(columns=ren)
for c in ren.values(): rel[c]=pd.to_numeric(rel[c],errors='coerce')
rel=rel.groupby(['Utility Number','State'],as_index=False)[list(ren.values())].mean()

# ---- 6b. AMI penetration (Advanced Meters): Total.1=AMI, Total.4=all meters ----
am=pd.read_excel(D+'Advanced_Meters_2024.xlsx',sheet_name='states',header=1)
am.columns=[str(c).strip() for c in am.columns]
am=am[am.State.isin(STATES)].copy()
am['ami']=pd.to_numeric(am['Total.1'],errors='coerce')
am['meters']=pd.to_numeric(am['Total.4'],errors='coerce')
am=am.groupby(['Utility Number','State'],as_index=False)[['ami','meters']].sum(min_count=1)

# ---- 7. join utility attributes onto utility x county rows ----
m=ter.merge(s,on=['Utility Number','State'],how='left') \
     .merge(dist,on=['Utility Number','State'],how='left') \
     .merge(rel,on=['Utility Number','State'],how='left') \
     .merge(am,on=['Utility Number','State'],how='left')
for c in ['cust_tot','cust_res','sales_mwh','circuits','ami','meters']:
    m['c_'+c]=m[c]*m['w']           # county share of that utility's state total

# ---- 8. aggregate to county ----
def wavg(g,val,wt):
    v=g[val]; w=g[wt]; k=v.notna()&w.notna()&(w>0)
    return np.average(v[k],weights=w[k]) if k.any() else np.nan
rows=[]
for f,g in m.groupby('fipsCode'):
    own=g.groupby('Ownership')['c_cust_tot'].sum()
    tot=own.sum()
    rows.append(dict(
        fipsCode=f, state=g.State.iloc[0],
        eia_n_utilities=g['Utility Number'].nunique(),
        eia_n_coops=g.loc[g.Ownership=='Cooperative','Utility Number'].nunique(),
        eia_customers=g['c_cust_tot'].sum(),
        eia_cust_residential=g['c_cust_res'].sum(),
        eia_sales_mwh=g['c_sales_mwh'].sum(),
        eia_dist_circuits=g['c_circuits'].sum(),
        eia_ami_meters=g['c_ami'].sum(), eia_all_meters=g['c_meters'].sum(),
        eia_coop_share=(own.get('Cooperative',0)/tot if tot else np.nan),
        eia_muni_share=(own.get('Municipal',0)/tot if tot else np.nan),
        eia_iou_share=(own.get('Investor Owned',0)/tot if tot else np.nan),
        eia_saidi_with_med=wavg(g,'saidi_with_med','c_cust_tot'),
        eia_saidi_no_med=wavg(g,'saidi_no_med','c_cust_tot'),
        eia_saifi_with_med=wavg(g,'saifi_with_med','c_cust_tot'),
        eia_saifi_no_med=wavg(g,'saifi_no_med','c_cust_tot'),
    ))
out=pd.DataFrame(rows)
out['eia_ami_share']=out.eia_ami_meters/out.eia_all_meters.replace(0,np.nan)
out['eia_cust_per_circuit']=out.eia_customers/out.eia_dist_circuits.replace(0,np.nan)
out['eia_mwh_per_customer']=out.eia_sales_mwh/out.eia_customers.replace(0,np.nan)
out=out.merge(cs[['fipsCode','population','land_sqmi']],on='fipsCode',how='left')
out['eia_cust_per_capita']=out.eia_customers/out.population
out['eia_circuits_per_1ksqmi']=1000*out.eia_dist_circuits/out.land_sqmi
tl=pd.read_csv(ROOT / 'data_static/eia861/county_transmission_hifld.csv',dtype={'fipsCode':str})
tl['fipsCode']=tl.fipsCode.str.zfill(5)
out=out.merge(tl,on='fipsCode',how='left')
out['eia_cust_per_tl_km']=out.eia_customers/out.tl_km.replace(0,np.nan)
o=ROOT / 'data_static/eia861/county_eia861_2024.csv'
out.to_csv(o,index=False)
print('\nWROTE',o,out.shape)
print(out.groupby('state').size())
print('\nnull rate per column:'); print((out.isna().mean()*100).round(1).to_string())
print('\nSAMPLE:'); print(out.head(6).to_string())
