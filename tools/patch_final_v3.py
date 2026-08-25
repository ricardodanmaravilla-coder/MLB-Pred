from pathlib import Path

p = Path('app_mlb.py')
s = p.read_text(encoding='utf-8')
orig = s
old = '''            dir_str = "None"\n            if wind_dir in ['N', 'NNE', 'NNW', 'NE']: dir_str = "Infield (Hacia Adentro)"\n            elif wind_dir in ['S', 'SSW', 'SSE', 'SW']: dir_str = "Outfield (Hacia Afuera)"\n            elif wind_dir in ['E', 'ENE', 'ESE']: dir_str = "Lateral (Derecha a Izquierda)"\n            elif wind_dir in ['W', 'WNW', 'WSW']: dir_str = "Lateral (Izquierda a Derecha)"\n            \n            return temp_f, wind_mph, dir_str\n'''
new = '''            # A compass direction cannot be translated to in/outfield without the\n            # physical orientation of this specific ballpark. Preserve the raw\n            # compass reading for diagnostics, but Monte Carlo will not apply a\n            # directional wind multiplier unless the user explicitly supplies\n            # an infield/outfield direction in the individual-game controls.\n            dir_str = f"Compass {wind_dir}" if wind_dir else "None"\n            return temp_f, wind_mph, dir_str\n'''
if old not in s:
    raise SystemExit('Weather direction block not found')
s = s.replace(old, new, 1)
if s == orig or 'Compass {wind_dir}' not in s:
    raise SystemExit('Wind patch not applied')
p.write_text(s, encoding='utf-8')
print('Final V3 wind patch applied')
