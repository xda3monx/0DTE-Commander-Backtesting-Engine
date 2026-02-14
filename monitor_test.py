from TradeMonitor import TradeMonitor

m = TradeMonitor()
# call _get_latest_bar and _get_indicators for quick check
es_bar = m._get_latest_bar('/ES', '3min')
spx_bar = m._get_latest_bar('$SPX', '3min')
print('ES latest bar:', es_bar)
print('SPX latest bar:', spx_bar)
es_ix = m._get_indicators('/ES', '3min', limit=2)
spx_ix = m._get_indicators('$SPX', '3min', limit=2)
print('ES indicator sample:', es_ix)
print('SPX indicator sample:', spx_ix)
entry, reason = m._evaluate_entry(es_bar, spx_bar, es_ix, spx_ix)
print('Entry:', entry, reason)
exit, reason = m._evaluate_exit(es_bar, spx_bar, es_ix, spx_ix)
print('Exit:', exit, reason)
