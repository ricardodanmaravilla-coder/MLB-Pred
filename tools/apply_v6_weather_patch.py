from pathlib import Path
p=Path('app_mlb.py'); t=p.read_text(encoding='utf-8')

def rep(old,new,label,expected=1):
    global t
    n=t.count(old)
    if n!=expected: raise SystemExit(f'{label}: expected {expected}, found {n}')
    t=t.replace(old,new)

rep('''from modules.game_context import (
    slate_date, park_for_team, match_odds_game, market_from_event, conservative_auto_weather
)
''','''from modules.game_context import (
    slate_date, park_for_team, match_odds_game, market_from_event, conservative_auto_weather, best_auto_weather
)
''','weather import')
rep('''                            temp_scan, viento_scan, dir_scan, weather_source = conservative_auto_weather(
                                datos_partido["local"], datos_partido.get("start_time_utc"), temp_raw, viento_raw, dir_raw
                            )
''','''                            temp_scan, viento_scan, dir_scan, weather_source = best_auto_weather(
                                datos_partido["local"], datos_partido.get("start_time_utc"), temp_raw, viento_raw, dir_raw
                            )
''','scanner weather')
rep('''            temp_auto, viento_auto, dir_auto = obtener_clima_estadio(datos_partido["local"])
''','''            current_temp, current_wind, current_dir = obtener_clima_estadio(datos_partido["local"])
            temp_auto, viento_auto, dir_auto, weather_auto_source = best_auto_weather(
                datos_partido["local"], datos_partido.get("start_time_utc"), current_temp, current_wind, current_dir
            )
''','individual weather')
# UI should not try to select raw compass direction in directional multiplier choices.
rep('''            indice_dir = opciones_viento.index(dir_auto) if dir_auto in opciones_viento else 0
''','''            indice_dir = opciones_viento.index(dir_auto) if dir_auto in opciones_viento else 0
            st.caption(f"Fuente clima automática: {weather_auto_source}")
''','weather source caption')
p.write_text(t,encoding='utf-8'); print('V6 game-time weather patch applied')
