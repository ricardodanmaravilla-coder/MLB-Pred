from pathlib import Path
p=Path('app_mlb.py');t=p.read_text(encoding='utf-8')

def rep(old,new,label,expected=1):
    global t
    n=t.count(old)
    if n!=expected:raise SystemExit(f'{label}: expected {expected}, found {n}')
    t=t.replace(old,new)

rep('''                                    "_Disagreement": cand.disagreement_pp,
                                    "_StarterHome": datos_partido.get('pitcher_local'), "_StarterAway": datos_partido.get('pitcher_visita'),
                                    "_Park": park_factor, "_Temp": temp_scan, "_Wind": viento_scan, "_WindDir": dir_scan,
                                    "_WeatherSource": weather_source,
''','''                                    "_Disagreement": cand.disagreement_pp, "_BlendWeightML": cand.blend_weight_ml,
                                    "_BatMetric": res_ml.get('Metrica_Bateo'), "_PitMetric": res_ml.get('Metrica_Pitcheo'),
                                    "_StarterHome": datos_partido.get('pitcher_local'), "_StarterAway": datos_partido.get('pitcher_visita'),
                                    "_StarterIPHome": starter_ip_loc, "_StarterIPAway": starter_ip_vis,
                                    "_Park": park_factor, "_Temp": temp_scan, "_Wind": viento_scan, "_WindDir": dir_scan,
                                    "_WeatherSource": weather_source,
''','recommendation audit state')
rep('''                                'market_no_vig': rr['_MarketNoVig'], 'edge_pp': rr['_Edge'], 'ev_pct': str(rr['EV+']).replace('%',''),
                                'disagreement_pp': rr['_Disagreement'], 'score': rr['_Score'],
                                'starter_away': rr['_StarterAway'], 'starter_home': rr['_StarterHome'],
                                'park_factor': rr['_Park'], 'temperature_f': rr['_Temp'], 'wind_mph': rr['_Wind'], 'wind_direction': rr['_WindDir'],
                                'model_version': 'v6', 'result_status': 'pending'
''','''                                'blend_weight_ml': rr['_BlendWeightML'], 'market_no_vig': rr['_MarketNoVig'], 'edge_pp': rr['_Edge'], 'ev_pct': str(rr['EV+']).replace('%',''),
                                'disagreement_pp': rr['_Disagreement'], 'score': rr['_Score'],
                                'batting_metric': rr['_BatMetric'], 'pitching_metric': rr['_PitMetric'],
                                'starter_away': rr['_StarterAway'], 'starter_home': rr['_StarterHome'],
                                'starter_ip_away': rr['_StarterIPAway'], 'starter_ip_home': rr['_StarterIPHome'],
                                'park_factor': rr['_Park'], 'temperature_f': rr['_Temp'], 'wind_mph': rr['_Wind'], 'wind_direction': rr['_WindDir'],
                                'weather_source': rr['_WeatherSource'], 'model_version': 'v6', 'result_status': 'pending'
''','ledger audit row')
p.write_text(t,encoding='utf-8');print('V6 ledger audit patch applied')
