import math, os, datetime
from functools import lru_cache
import pandas as pd
import requests
from modules.montecarlo_mlb import simular_partido_mlb
from modules.ml_mlb import PredictorMLMLB
from modules.scanner_engine import no_vig_two_way, moneyline_candidate, total_candidate, runline_candidate
from modules.metric_quality import batting_metric, pitching_metric, row_pitching_value
from modules.game_context import park_for_team, match_odds_game, market_from_event, conservative_auto_weather

TEAMS={"New York Yankees":"NYY","Boston Red Sox":"BOS","Los Angeles Dodgers":"LAD","Houston Astros":"HOU","Atlanta Braves":"ATL","Philadelphia Phillies":"PHI","Baltimore Orioles":"BAL","Tampa Bay Rays":"TB","Toronto Blue Jays":"TOR","Chicago White Sox":"CWS","Cleveland Guardians":"CLE","Detroit Tigers":"DET","Kansas City Royals":"KC","Minnesota Twins":"MIN","Los Angeles Angels":"LAA","Oakland Athletics":"OAK","Athletics":"OAK","Sacramento Athletics":"OAK","Seattle Mariners":"SEA","Texas Rangers":"TEX","Chicago Cubs":"CHC","Cincinnati Reds":"CIN","Milwaukee Brewers":"MIL","Pittsburgh Pirates":"PIT","St. Louis Cardinals":"STL","Arizona Diamondbacks":"AZ","Colorado Rockies":"COL","San Francisco Giants":"SF","San Diego Padres":"SD","Miami Marlins":"MIA","New York Mets":"NYM","Washington Nationals":"WSH"}
CITIES={"New York Yankees":"New_York","Boston Red Sox":"Boston","Los Angeles Dodgers":"Los_Angeles","Houston Astros":"Houston","Atlanta Braves":"Atlanta","Philadelphia Phillies":"Philadelphia","Baltimore Orioles":"Baltimore","Tampa Bay Rays":"St_Petersburg","Toronto Blue Jays":"Toronto","Chicago White Sox":"Chicago","Cleveland Guardians":"Cleveland","Detroit Tigers":"Detroit","Kansas City Royals":"Kansas_City","Minnesota Twins":"Minneapolis","Los Angeles Angels":"Anaheim","Oakland Athletics":"Oakland","Athletics":"Sacramento","Seattle Mariners":"Seattle","Texas Rangers":"Arlington","Chicago Cubs":"Chicago","Cincinnati Reds":"Cincinnati","Milwaukee Brewers":"Milwaukee","Pittsburgh Pirates":"Pittsburgh","St. Louis Cardinals":"St_Louis","Arizona Diamondbacks":"Phoenix","Colorado Rockies":"Denver","San Francisco Giants":"San_Francisco","San Diego Padres":"San_Diego","Miami Marlins":"Miami","New York Mets":"New_York","Washington Nationals":"Washington"}

def dec(x):
    try:
        x=float(x)
        return round(1+x/100,3) if x>0 else round(1+100/abs(x),3)
    except: return None

def prob_normal(mu,line,kind,sigma):
    sigma=max(1.0,float(sigma or (3.5 if kind in ('over','under') else 4.2)))
    z=((mu-line) if kind in ('over','spread_loc') else (line-mu))/sigma
    return max(0,min(100,100*0.5*(1+math.erf(z/math.sqrt(2)))))

def kelly(pct,odds,push=0):
    try:
        p=float(pct)/100; r=float(push)/100; q=max(0,1-p-r); b=float(odds)-1; d=p+q
        return round(max(0,(b*p-q)/(b*d))*25,2) if b>0 and d>0 else 0
    except: return 0

class MLBCloudService:
    def __init__(self):
        self.bat=self._csv('data/mlb_batting.csv'); self.pit=self._csv('data/mlb_pitching.csv')
        self.ind=self._csv('data/mlb_pitching_individual.csv'); self.parks=self._csv('data/mlb_park_factors.csv')
        self.hist=self._csv('data/mlb_games.csv'); self.bull=self._csv('data/mlb_bullpen.csv')
        self.ml=PredictorMLMLB()
        if not self.hist.empty and not self.bat.empty and not self.pit.empty: self.ml.entrenar(self.bat,self.pit,self.hist)
    @staticmethod
    def _csv(path):
        try: return pd.read_csv(path,sep=None,engine='python',on_bad_lines='skip') if os.path.exists(path) else pd.DataFrame()
        except: return pd.DataFrame()
    def health(self):
        return {'status':'ok','runtime':'google-cloud-run','ui':'fastapi','streamlit':False,'history_rows':len(self.hist),'ml_ready':bool(getattr(self.ml,'entrenado',False))}
    @lru_cache(maxsize=2)
    def _slate(self,date):
        url=f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher,team'
        r=requests.get(url,timeout=10); r.raise_for_status(); out={}
        for d in r.json().get('dates',[]):
            for g in d.get('games',[]):
                h=g['teams']['home']['team']['name']; a=g['teams']['away']['team']['name']; pk=int(g['gamePk'])
                out[pk]={'game_pk':pk,'home':h,'away':a,'starter_home':g['teams']['home'].get('probablePitcher',{}).get('fullName','Por Anunciar'),'starter_away':g['teams']['away'].get('probablePitcher',{}).get('fullName','Por Anunciar'),'start_time_utc':g.get('gameDate'),'line':None,'odds_home':None,'odds_away':None,'odds_over':None,'odds_under':None,'spread_home':None,'spread_away':None,'odds_spread_home':None,'odds_spread_away':None}
        key=os.getenv('ODDS_API_KEY','').strip()
        if key and out:
            ro=requests.get(f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={key}&regions=us&markets=h2h,totals,spreads&oddsFormat=american',timeout=10)
            if ro.ok:
                events=ro.json()
                for x in out.values():
                    legacy={'local':x['home'],'visita':x['away'],'start_time_utc':x['start_time_utc']}
                    ev=match_odds_game(events,legacy)
                    if ev:
                        m=market_from_event(ev,dec)
                        x.update({'line':m.get('linea_carreras'),'odds_home':m.get('cuota_loc'),'odds_away':m.get('cuota_vis'),'odds_over':m.get('cuota_over'),'odds_under':m.get('cuota_under'),'spread_home':m.get('spread_loc'),'spread_away':m.get('spread_vis'),'odds_spread_home':m.get('cuota_spread_loc'),'odds_spread_away':m.get('cuota_spread_vis')})
        return out
    def games(self): return list(self._slate(datetime.date.today().isoformat()).values())
    def _starter(self,name,team):
        if not self.ind.empty and 'Name' in self.ind:
            x=self.ind[self.ind.Name.astype(str).str.casefold()==str(name).casefold()]
            if not x.empty:
                v,_=row_pitching_value(x.iloc[-1],None)
                if v is not None:return float(v)
        x=self.pit[self.pit.Team.astype(str)==team] if 'Team' in self.pit else pd.DataFrame()
        if not x.empty:
            v,_=row_pitching_value(x.iloc[-1],None)
            if v is not None:return float(v)
        return 4.2
    def _off(self,team):
        col=batting_metric(self.bat)
        if not col or 'Team' not in self.bat:return 100.0
        x=self.bat[self.bat.Team.astype(str)==team]
        if x.empty:return 100.0
        v=pd.to_numeric(x[col],errors='coerce').dropna()
        allv=pd.to_numeric(self.bat[col],errors='coerce').dropna()
        if v.empty or allv.empty:return 100.0
        center=float(allv.median()); return float(max(75,min(125,float(v.iloc[-1])/center*100))) if center else 100.0
    def _prior(self,df,team,col,fallback):
        if col not in df or 'Team' not in df:return fallback
        x=df[df.Team.astype(str)==team]; v=pd.to_numeric(x[col],errors='coerce').dropna()
        return float(v.iloc[-1]) if not v.empty else fallback
    def _weather(self,home,start):
        city=CITIES.get(home); t=w=None; direction='None'
        if city:
            try:
                r=requests.get(f'https://wttr.in/{city}?format=j1',timeout=5); c=r.json()['current_condition'][0]
                t=int(c['temp_F']); w=int(c['windspeedMiles']); direction='None'
            except: pass
        return conservative_auto_weather(home,start,t,w,direction)
    def analyze(self,game_pk):
        g=self._slate(datetime.date.today().isoformat())[int(game_pk)]; h=TEAMS.get(g['home']); a=TEAMS.get(g['away'])
        if not h or not a: raise ValueError('Equipo sin mapeo MLB')
        park=park_for_team(self.parks,h)
        if not park: raise ValueError('Parque sin datos')
        oh,oa=self._off(h),self._off(a); ph=self._starter(g['starter_home'],h); pa=self._starter(g['starter_away'],a)
        bh=self._prior(self.bull,h,'ERA',4.2); ba=self._prior(self.bull,a,'ERA',4.2)
        temp,wind,wdir,wsrc=self._weather(g['home'],g['start_time_utc']); line=g['line'] or 8.5
        mc=simular_partido_mlb(local=g['home'],visita=g['away'],pitcher_loc_xfip=ph,pitcher_vis_xfip=pa,wrc_loc=oh,wrc_vis=oa,bullpen_loc_era=bh,bullpen_vis_era=ba,park_factor=park['park_factor'],altitud_ft=park['altitude_ft'],viento_mph=wind,direccion_viento=wdir,temp_f=temp,linea_carreras_casino=line,df_games=self.hist,num_simulaciones=50000)
        bc=batting_metric(self.bat) or 'wRC+'; pc=pitching_metric(self.pit) or 'ERA'
        ml=self.ml.predecir_partido(h,a,self._prior(self.bat,h,bc,oh),self._prior(self.bat,a,bc,oa),self._prior(self.pit,h,pc,bh),self._prior(self.pit,a,pc,ba),park['park_factor'],game_date=datetime.date.today())
        runs=mc.get('Carreras',{}); mh,mv=no_vig_two_way(g['odds_home'],g['odds_away']); mo,mu=no_vig_two_way(g['odds_over'],g['odds_under'])
        pmo=prob_normal(ml.get('Proyeccion_Carreras',line),line,'over',ml.get('Sigma_Carreras')); pmu=prob_normal(ml.get('Proyeccion_Carreras',line),line,'under',ml.get('Sigma_Carreras'))
        cand=[moneyline_candidate(f"Gana {g['home']}",ml['Probabilidad_Local'],mc['Moneyline']['Gana Local'],g['odds_home'],mh),moneyline_candidate(f"Gana {g['away']}",ml['Probabilidad_Visita'],mc['Moneyline']['Gana Visita'],g['odds_away'],mv),total_candidate(f'Over {line}',pmo,runs.get(f'Over {line}',50),g['odds_over'],mo,runs.get(f'Push {line}',0)),total_candidate(f'Under {line}',pmu,runs.get(f'Under {line}',50),g['odds_under'],mu,runs.get(f'Push {line}',0))]
        picks=[]
        for c in cand:
            if c and c.accepted:picks.append({'market':c.market,'selection':c.selection,'probability':round(c.probability,2),'ml':round(c.prob_ml,2),'mc':round(c.prob_mc,2),'odds':c.odds,'edge_pp':c.edge_pp,'ev_pct':c.ev_pct,'kelly_pct':kelly(c.probability,c.odds,c.push_probability),'score':c.score})
        picks.sort(key=lambda x:x['score'],reverse=True)
        return {'game':g,'model':{'prob_home':ml['Probabilidad_Local'],'prob_away':ml['Probabilidad_Visita'],'projected_runs':ml.get('Proyeccion_Carreras'),'projected_home_margin':ml.get('Proyeccion_Handicap_Local')},'montecarlo':mc,'context':{'park_factor':park['park_factor'],'altitude_ft':park['altitude_ft'],'temperature_f':temp,'wind_mph':wind,'weather_source':wsrc},'picks':picks}
    def scan(self):
        results=[]; errors=[]
        for g in self.games():
            try:
                a=self.analyze(g['game_pk'])
                for p in a['picks']: results.append({**p,'game_pk':g['game_pk'],'game':f"{g['away']} @ {g['home']}"})
            except Exception as e: errors.append({'game_pk':g['game_pk'],'game':f"{g['away']} @ {g['home']}",'error':str(e)[:160]})
        results.sort(key=lambda x:x['score'],reverse=True)
        return {'recommendations':results[:3],'evaluated_games':len(self.games()),'errors':errors}
