import numpy as np
import pandas as pd


plastic_signatures = {
    'PE':  {'Pol0': 180, 'Pol45': 160, 'Pol90': 140, 'Pol135': 150, 'std': 10},
    'PET': {'Pol0': 300, 'Pol45': 295, 'Pol90': 290, 'Pol135': 285, 'std': 15},
    'PP':  {'Pol0': 220, 'Pol45': 215, 'Pol90': 210, 'Pol135': 205, 'std': 10},
    'PS':  {'Pol0': 250, 'Pol45': 240, 'Pol90': 230, 'Pol135': 225, 'std': 12},
    'PVC': {'Pol0': 290, 'Pol45': 280, 'Pol90': 270, 'Pol135': 260, 'std': 12},
    'PC':  {'Pol0': 260, 'Pol45': 250, 'Pol90': 240, 'Pol135': 230, 'std': 10}
}


N = 500
data = []


for plastic, sig in plastic_signatures.items():
      for _ in range(N):

          Pol0 = np.random.normal(sig['Pol0'], sig['std'])
          Pol45 = np.random.normal(sig['Pol45'], sig['std'])
          Pol90 = np.random.normal(sig['Pol90'], sig['std'])
          Pol135 = np.random.normal(sig['Pol135'], sig['std'])

          Pol_Ratio_0_90 = Pol0 / Pol90
          Pol_Diff_0_90 = Pol0 - Pol90

          data.append([Pol0, Pol45, Pol90, Pol135, Pol_Ratio_0_90, Pol_Diff_0_90, plastic])


df = pd.DataFrame(data, columns=['Pol0', 'Pol45', 'Pol90', 'Pol135', 'Pol_Ratio_0_90', 'Pol_Diff_0_90', 'PlasticType'])


df.to_csv('synthetic_microplastic_4angle.csv', index=False)
