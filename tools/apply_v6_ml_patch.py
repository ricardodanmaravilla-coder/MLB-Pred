from pathlib import Path

p=Path('modules/ml_mlb.py')
t=p.read_text(encoding='utf-8')

def rep(old,new,label,count=1):
    global t
    n=t.count(old)
    if n!=count:
        raise SystemExit(f'{label}: expected {count}, got {n}')
    t=t.replace(old,new)

rep('from sklearn.ensemble import GradientBoostingRegressor\n',
    'from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor\n',
    'ensemble imports')
rep('from .historical_mlb import prepare_games, team_state, h2h_state, append_game\n',
    'from .historical_mlb import prepare_games, team_state, h2h_state, append_game\nfrom .metric_quality import batting_metric, pitching_metric\n',
    'metric imports')

rep('''    def __init__(self):
        self.modelo_ganador = self._new_classifier()
        self.modelo_carreras = GradientBoostingRegressor(
''','''    def __init__(self):
        self.classifier_family = 'logistic'
        self.runs_family = 'gbr'
        self.diff_family = 'gbr'
        self.modelo_ganador = self._new_classifier(self.classifier_family)
        self.modelo_carreras = GradientBoostingRegressor(
''','init classifier')

old='''    @staticmethod
    def _new_classifier():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, max_iter=2000, solver='lbfgs', random_state=42)
        )
'''
new='''    @staticmethod
    def _new_classifier(family='logistic'):
        if family == 'histgb':
            return HistGradientBoostingClassifier(
                learning_rate=0.04, max_iter=180, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=2.0, random_state=42
            )
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, max_iter=2000, solver='lbfgs', random_state=42)
        )

    @staticmethod
    def _new_regressor(family='gbr'):
        if family == 'histgb':
            return HistGradientBoostingRegressor(
                learning_rate=0.04, max_iter=180, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=2.0, random_state=42
            )
        return GradientBoostingRegressor(
            n_estimators=140, max_depth=2, learning_rate=0.035,
            loss='huber', random_state=42
        )

    @staticmethod
    def _ewma_state(history, team, n=12, alpha=0.28):
        rows = history.get(team, [])[-int(n):]
        if not rows:
            return 0.5, 0.0
        weights = np.array([(1.0-alpha)**i for i in range(len(rows)-1, -1, -1)], dtype=float)
        weights /= weights.sum()
        wins = np.array([r[0] for r in rows], dtype=float)
        rd = np.array([r[1]-r[2] for r in rows], dtype=float)
        return float(np.dot(weights, wins)), float(np.dot(weights, rd))
'''
rep(old,new,'model factories')

old='''        w20l, rf20l, ra20l, rd20l = team_state(hist, loc, 20)
        w20v, rf20v, ra20v, rd20v = team_state(hist, vis, 20)
        hwin, hrd, hn = h2h_state(h2h, loc, vis, 12)
        return [
            w5l, w5v, w20l, w20v,
            rf5l, rf5v, ra5l, ra5v,
            rd5l, rd5v, rd20l, rd20v,
'''
new='''        w20l, rf20l, ra20l, rd20l = team_state(hist, loc, 20)
        w20v, rf20v, ra20v, rd20v = team_state(hist, vis, 20)
        w50l, rf50l, ra50l, rd50l = team_state(hist, loc, 50)
        w50v, rf50v, ra50v, rd50v = team_state(hist, vis, 50)
        ewl, ewrdl = self._ewma_state(hist, loc)
        ewv, ewrdv = self._ewma_state(hist, vis)
        hwin, hrd, hn = h2h_state(h2h, loc, vis, 12)
        return [
            w5l, w5v, w20l, w20v, w50l, w50v, ewl, ewv,
            rf5l, rf5v, ra5l, ra5v,
            rd5l, rd5v, rd20l, rd20v, rd50l, rd50v, ewrdl, ewrdv,
'''
rep(old,new,'enhanced rolling features')

rep("        self.modelo_ganador = cached['modelo_ganador']\n", "        self.modelo_ganador = cached['modelo_ganador']\n        self.classifier_family = cached.get('classifier_family','logistic')\n        self.runs_family = cached.get('runs_family','gbr')\n        self.diff_family = cached.get('diff_family','gbr')\n", 'restore families')

rep("            bat_col = 'OPS_Index' if 'OPS_Index' in df_batting.columns else 'wRC+'\n            pit_col = 'ERA' if 'ERA' in df_pitching.columns else 'xFIP'\n", "            bat_col = batting_metric(df_batting)\n            pit_col = pitching_metric(df_pitching)\n", 'quality metric choice')

old='''            # Chronological validation: fit only on the earlier block, calibrate on
            # later unseen games, then refit on all available historical games.
            self.modelo_carreras.fit(X[:cut], yr[:cut])
            self.modelo_handicap.fit(X[:cut], yd[:cut])
            val_runs_pred = self.modelo_carreras.predict(X[cut:])
            val_diff_pred = self.modelo_handicap.predict(X[cut:])
            self.sigma_runs = _calibrate_sigma(val_runs_pred, yr[cut:], market='total')
            self.sigma_diff = _calibrate_sigma(val_diff_pred, yd[cut:], market='spread')

            temp_classifier = self._new_classifier()
            temp_classifier.fit(X[:cut], yw[:cut])
            raw_cal = temp_classifier.predict_proba(X[cut:])[:, 1]
            best_alpha, best_brier = 1.0, float('inf')
            for alpha in np.linspace(0.35, 1.00, 66):
                cal = 0.5 + alpha * (raw_cal - 0.5)
                score = float(np.mean((cal - yw[cut:]) ** 2))
                if score < best_brier:
                    best_brier, best_alpha = score, float(alpha)
            self.prob_shrink = best_alpha

            self.modelo_ganador = self._new_classifier()
            self.modelo_ganador.fit(X, yw)
            self.modelo_carreras.fit(X, yr)
            self.modelo_handicap.fit(X, yd)
'''
new='''            # V6 chronological model selection. Complexity is accepted only when it
            # improves genuinely later unseen games, never by in-sample fit.
            run_candidates = {}
            diff_candidates = {}
            for family in ('gbr','histgb'):
                rm = self._new_regressor(family); rm.fit(X[:cut], yr[:cut])
                rp = rm.predict(X[cut:]); run_candidates[family] = (float(np.mean(np.abs(rp-yr[cut:]))), rp)
                dm = self._new_regressor(family); dm.fit(X[:cut], yd[:cut])
                dp = dm.predict(X[cut:]); diff_candidates[family] = (float(np.mean(np.abs(dp-yd[cut:]))), dp)
            self.runs_family = min(run_candidates, key=lambda k: run_candidates[k][0])
            self.diff_family = min(diff_candidates, key=lambda k: diff_candidates[k][0])
            val_runs_pred = run_candidates[self.runs_family][1]
            val_diff_pred = diff_candidates[self.diff_family][1]
            self.sigma_runs = _calibrate_sigma(val_runs_pred, yr[cut:], market='total')
            self.sigma_diff = _calibrate_sigma(val_diff_pred, yd[cut:], market='spread')

            best = None
            for family in ('logistic','histgb'):
                cm = self._new_classifier(family); cm.fit(X[:cut], yw[:cut])
                raw = cm.predict_proba(X[cut:])[:,1]
                for alpha in np.linspace(0.35,1.00,66):
                    cal = 0.5 + alpha*(raw-0.5)
                    brier = float(np.mean((cal-yw[cut:])**2))
                    if best is None or brier < best[0]:
                        best = (brier, family, float(alpha))
            _, self.classifier_family, self.prob_shrink = best

            self.modelo_ganador = self._new_classifier(self.classifier_family)
            self.modelo_carreras = self._new_regressor(self.runs_family)
            self.modelo_handicap = self._new_regressor(self.diff_family)
            self.modelo_ganador.fit(X, yw)
            self.modelo_carreras.fit(X, yr)
            self.modelo_handicap.fit(X, yd)
'''
rep(old,new,'temporal model selection')

rep("                'modelo_ganador': self.modelo_ganador,\n", "                'modelo_ganador': self.modelo_ganador,\n                'classifier_family': self.classifier_family,\n                'runs_family': self.runs_family,\n                'diff_family': self.diff_family,\n", 'cache families')

rep("                'Modelo_Desde_Cache': bool(self.loaded_from_cache),\n", "                'Modelo_Desde_Cache': bool(self.loaded_from_cache),\n                'Classifier_Family': self.classifier_family,\n                'Runs_Family': self.runs_family,\n                'Diff_Family': self.diff_family,\n", 'prediction family metadata')

p.write_text(t,encoding='utf-8')
print('V6 ML patch applied')
