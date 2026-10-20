"""Создание учебного набора событий авторизации для задания 1.
Не является реальным журналом ИБ; метки созданы искусственно.
"""
from pathlib import Path
import numpy as np
import pandas as pd
rng = np.random.default_rng(42)
n = 2400
night = rng.random(n) < 0.24
hour = np.where(night, rng.choice([0,1,2,3,4,5,6,7,20,21,22,23], size=n), rng.integers(8,20,size=n))
risk_probe = rng.random(n) < 0.105
failed = rng.poisson(np.where(risk_probe, 3.5, 0.65)).astype(int)
new_dev = (rng.random(n) < np.where(risk_probe, 0.32, 0.15)).astype(int)
foreign = (rng.random(n) < np.where(risk_probe, 0.21, 0.055)).astype(int)
vpn = (rng.random(n) < 0.21).astype(int)
score = (-4.15 + 0.54*np.minimum(failed, 9) + 1.32*new_dev + 1.78*foreign +
         0.65*night.astype(int) + 0.20*vpn + 0.75*new_dev*foreign + rng.normal(0,0.38,n))
probability = 1 / (1 + np.exp(-score))
labels = (rng.random(n) < probability).astype(int)
df = pd.DataFrame({
    'event_id': [f'EVT{i:05d}' for i in range(1,n+1)],
    'hour': hour,
    'failed_attempts_24h': failed,
    'new_device': new_dev,
    'unusual_country': foreign.astype(float),
    'vpn_used': vpn,
    'is_suspicious': labels
})
missing = rng.choice(n, size=37, replace=False)
df.loc[missing, 'unusual_country'] = np.nan
outfile = Path(__file__).resolve().parent / 'auth_events.csv'
df.to_csv(outfile, index=False, encoding='utf-8-sig')
print(f'Сохранено: {outfile} ({len(df)} событий)')
