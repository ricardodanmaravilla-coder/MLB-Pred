import calendar
import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import statsapi

from modules.fangraphs_mlb import fetch_team_fangraphs
from modules.team_utils import normalize_team

TEMPORADAS = [2021, 2022, 2023, 2024, 2025, 2026]


def _safe_float(value, default=None):
    try:
        if value in (None, "", "-.--"): return default
        return float(value)
    except (TypeError, ValueError): return default


def _innings_to_float(value):
    try:
        text=str(value or '0'); whole,_,frac=text.partition('.'); outs=int(frac[:1] or 0)
        if outs not in (0,1,2): return float(text)
        return float(int(whole)+outs/3.0)
    except Exception:return 0.0


def _official_team_data():
    bateo_data,pitcheo_data=[],[]
    for year in TEMPORADAS:
        print(f"📊 MLB StatsAPI oficial {year}...")
        try:
            equipos=statsapi.get("teams",{"season":year,"sportId":1})["teams"]
            for equipo in equipos:
                team_id=equipo["id"]; team_abbr=normalize_team(equipo.get("abbreviation","UNK"))
                if not team_abbr or not equipo.get("active",True):continue
                sb=statsapi.get("team_stats",{"teamId":team_id,"season":year,"group":"hitting","stats":"season"})
                if sb and sb.get("stats"):
                    splits=sb["stats"][0].get("splits",[])
                    if splits:
                        stat=dict(splits[0].get("stat",{})); stat["Team"]=team_abbr; stat["Season"]=year; stat["DataSource"]="MLB_STATSAPI_OFFICIAL"
                        ops=_safe_float(stat.get("ops")); obp=_safe_float(stat.get("obp")); slg=_safe_float(stat.get("slg")); avg=_safe_float(stat.get("avg")); pa=_safe_float(stat.get("plateAppearances"),0) or 0
                        bb=_safe_float(stat.get("baseOnBalls"),0) or 0; k=_safe_float(stat.get("strikeOuts"),0) or 0
                        stat["OPS"]=ops; stat["OPS_Index_Raw"]=None if ops is None else ops*100.; stat["ISO"]=None if slg is None or avg is None else slg-avg
                        stat["BB%_Official"]=None if pa<=0 else bb/pa*100.; stat["K%_Official"]=None if pa<=0 else k/pa*100.
                        bateo_data.append(stat)
                sp=statsapi.get("team_stats",{"teamId":team_id,"season":year,"group":"pitching","stats":"season"})
                if sp and sp.get("stats"):
                    splits=sp["stats"][0].get("splits",[])
                    if splits:
                        stat=dict(splits[0].get("stat",{})); stat["Team"]=team_abbr; stat["Season"]=year; stat["DataSource"]="MLB_STATSAPI_OFFICIAL"
                        stat["ERA"]=_safe_float(stat.get("era")); stat["IP_Float"]=_innings_to_float(stat.get("inningsPitched")); pitcheo_data.append(stat)
                time.sleep(.08)
        except Exception as exc: print(f"❌ Error StatsAPI {year}: {exc}")
    bat=pd.DataFrame(bateo_data); pit=pd.DataFrame(pitcheo_data)
    if not bat.empty:
        # Park-neutrality is not claimed. This is a transparent league-relative OPS
        # strength index built from official OBP/SLG, centered at 100 each season.
        bat['obp_num']=pd.to_numeric(bat.get('obp'),errors='coerce'); bat['slg_num']=pd.to_numeric(bat.get('slg'),errors='coerce')
        for yr,idx in bat.groupby('Season').groups.items():
            rows=bat.loc[idx]; lg_obp=float(rows['obp_num'].median()); lg_slg=float(rows['slg_num'].median())
            if lg_obp>0 and lg_slg>0: bat.loc[idx,'Offense_Index']=((rows['obp_num']/lg_obp)+(rows['slg_num']/lg_slg)-1.)*100.
        bat['OPS_Index']=bat['Offense_Index'].combine_first(pd.to_numeric(bat.get('OPS_Index_Raw'),errors='coerce'))
        bat['wRC+']=bat['OPS_Index']; bat['wRC+_Source']='MLB_OFFICIAL_OBP_SLG_INDEX_NOT_WRCPLUS'
    if not pit.empty:
        for c in ('homeRuns','baseOnBalls','hitByPitch','strikeOuts','battersFaced'):
            pit[c+'_num']=pd.to_numeric(pit.get(c),errors='coerce').fillna(0.)
        pit['IP_Float']=pd.to_numeric(pit['IP_Float'],errors='coerce').fillna(0.)
        pit['ERA']=pd.to_numeric(pit['ERA'],errors='coerce')
        for yr,idx in pit.groupby('Season').groups.items():
            rows=pit.loc[idx]; ip=float(rows['IP_Float'].sum()); hr=float(rows['homeRuns_num'].sum()); bb=float(rows['baseOnBalls_num'].sum()); hbp=float(rows['hitByPitch_num'].sum()); k=float(rows['strikeOuts_num'].sum())
            lg_era=float(np.average(rows['ERA'].dropna(),weights=rows.loc[rows['ERA'].notna(),'IP_Float'].clip(lower=.1))) if rows['ERA'].notna().any() else 4.10
            raw=(13*hr+3*(bb+hbp)-2*k)/ip if ip>0 else 0.; const=lg_era-raw; pit.loc[idx,'FIP_Constant']=const
            rip=rows['IP_Float'].replace(0,np.nan); pit.loc[idx,'FIP']=(13*rows['homeRuns_num']+3*(rows['baseOnBalls_num']+rows['hitByPitch_num'])-2*rows['strikeOuts_num'])/rip+const
            bf=rows['battersFaced_num'].replace(0,np.nan); pit.loc[idx,'K-BB%']=(rows['strikeOuts_num']-rows['baseOnBalls_num'])/bf*100.; pit.loc[idx,'HR/9']=rows['homeRuns_num']/rip*9.
        pit['xFIP']=pit['FIP']; pit['xFIP_Source']='MLB_OFFICIAL_FIP_USED_NOT_XFIP'
    return bat,pit


def extraer_estadisticas_oficiales_mlb():
    print("⚾ [INICIO] Extrayendo y derivando métricas oficiales MLB..."); os.makedirs("data",exist_ok=True)
    bat,pit=_official_team_data()
    # FanGraphs remains an opt-in enrichment, never a production dependency.
    if os.getenv('ENABLE_FANGRAPHS','').strip()=='1':
        fg_bat,fg_pit,_=fetch_team_fangraphs(min(TEMPORADAS),max(TEMPORADAS))
        if not fg_bat.empty and {'Team','Season','wRC+'}.issubset(fg_bat.columns):
            fg_bat['Team']=fg_bat['Team'].map(normalize_team); bat=fg_bat; bat['wRC+_Source']='FANGRAPHS_REAL_WRCPLUS'; print('✅ FanGraphs wRC+ habilitado.')
        if not fg_pit.empty and {'Team','Season','xFIP'}.issubset(fg_pit.columns):
            fg_pit['Team']=fg_pit['Team'].map(normalize_team); pit=fg_pit; pit['xFIP_Source']='FANGRAPHS_REAL_XFIP'; print('✅ FanGraphs xFIP habilitado.')
    if bat.empty or pit.empty: raise RuntimeError('No se pudieron construir métricas oficiales de equipos')
    bat.to_csv('data/mlb_batting.csv',index=False); pit.to_csv('data/mlb_pitching.csv',index=False)
    print(f"✅ Bateo oficial: {len(bat)} filas; índice ofensivo/ISO/K%/BB%."); print(f"✅ Pitcheo oficial: {len(pit)} filas; FIP/K-BB%/HR9/WHIP.")


def _monthly_ranges(year):
    ranges=[]
    for month in range(1,10): last=calendar.monthrange(year,month)[1]; ranges.append((f"{month:02d}/01/{year}",f"{month:02d}/{last:02d}/{year}"))
    ranges.append((f"10/01/{year}",f"12/31/{year}")); return ranges

def extraer_historico_juegos():
    print("\n🗓️ Actualizando historial de juegos..."); archivo_csv="data/mlb_games.csv"; filas_antes=0; df_existente=pd.DataFrame(); tramos_descarga=[]
    if os.path.exists(archivo_csv):
        df_existente=pd.read_csv(archivo_csv); filas_antes=len(df_existente); ultima_fecha=pd.to_datetime(df_existente["Date"],errors="coerce").max()
        if pd.notna(ultima_fecha): inicio=ultima_fecha-timedelta(days=3); fin=date.today(); tramos_descarga.append((inicio.strftime("%m/%d/%Y"),fin.strftime("%m/%d/%Y")))
    if not tramos_descarga:
        for year in TEMPORADAS:tramos_descarga.extend(_monthly_ranges(year))
    nuevos=[]; ignorados={"Scheduled","Pre-Game","Postponed","Cancelled","Delayed","Warmup","Preview"}
    for inc,fin in tramos_descarga:
        for intento in range(3):
            try:
                for game in statsapi.schedule(start_date=inc,end_date=fin):
                    if game.get('status','Unknown') in ignorados or 'away_score' not in game or 'home_score' not in game:continue
                    gd=game.get('game_date'); nuevos.append({'GameID':game.get('game_id'),'Date':gd,'Season':str(gd)[:4],'GameType':game.get('game_type') or game.get('gameType'),'Away':game.get('away_name'),'Home':game.get('home_name'),'Away_Score':game.get('away_score',0),'Home_Score':game.get('home_score',0),'Away_Starter':game.get('away_probable_pitcher') or game.get('away_pitcher'),'Home_Starter':game.get('home_probable_pitcher') or game.get('home_pitcher'),'Innings':game.get('current_inning',9),'Venue':game.get('venue_name','Unknown')})
                time.sleep(.3);break
            except Exception as exc:print(f"⚠️ Intento {intento+1}: {exc}");time.sleep(2)
    if not nuevos:return
    df_n=pd.DataFrame(nuevos); df_f=pd.concat([df_existente,df_n],ignore_index=True) if not df_existente.empty else df_n
    if 'GameID' in df_f.columns:df_f=df_f.drop_duplicates(subset=['GameID'],keep='last')
    df_f.to_csv(archivo_csv,index=False);print(f"✅ {archivo_csv}: {len(df_f)} partidos ({len(df_f)-filas_antes:+d}).")

if __name__=='__main__':
    os.makedirs('data',exist_ok=True);extraer_estadisticas_oficiales_mlb();extraer_historico_juegos();print('🎯 Minería completada.')
