#Iwanaga, T., Usher, W., & Herman, J. (2022). Toward SALib 2.0:
#Advancing the accessibility and interpretability of global sensitivity analyses.
# Socio-Environmental Systems Modelling, 4, 18155. doi:10.18174/sesmo.18155

#Herman, J. and Usher, W. (2017) SALib:
# An open-source Python library for sensitivity analysis. Journal of Open Source Software,2(9).
# doi:10.21105/joss.00097
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SALib.sample import morris
from SALib.analyze import morris as morris_analyze

import connector   # твой файл с моделью


# ==========================
# 1. Описание параметров
# ==========================

problem = {
    "num_vars": 3,

    "names": [
        "x1",
        "x2",
        "x3"
    ],

    "bounds": [
        [0, 1],
        [0, 1],
        [0, 1]
    ]
}


# ==========================
# 2. Генерация Morris выборки
# ==========================

N = 300

param_values = morris.sample(
    problem,
    N=N,
    num_levels=4,
    optimal_trajectories=None
)


print(
    "Количество запусков модели:",
    len(param_values)
)


# ==========================
# 3. Запуск модели
# ==========================

FD_values = []
LAC_values = []


for i, params in enumerate(param_values):

    x1, x2, x3 = params

    fd, lac = connector.do(
        [x1, x2, x3]
    )

    FD_values.append(fd)
    LAC_values.append(lac)


    if i % 50 == 0:
        print(
            f"Прогресс: {i}/{len(param_values)}"
        )


FD_values = np.array(FD_values)
LAC_values = np.array(LAC_values)



# ==========================
# 4. Morris анализ FD
# ==========================

Si_fd = morris_analyze.analyze(
    problem,
    param_values,
    FD_values,
    conf_level=0.95,
    print_to_console=True
)



# ==========================
# 5. Morris анализ lacunarity
# ==========================

Si_lac = morris_analyze.analyze(
    problem,
    param_values,
    LAC_values,
    conf_level=0.95,
    print_to_console=True
)



# ==========================
# 6. Сохраняем результаты
# ==========================


df_fd = pd.DataFrame({

    "parameter": problem["names"],

    "mu_star": Si_fd["mu_star"],

    "sigma": Si_fd["sigma"],

    "mu_star_conf":
        Si_fd["mu_star_conf"]

})


df_lac = pd.DataFrame({

    "parameter": problem["names"],

    "mu_star": Si_lac["mu_star"],

    "sigma": Si_lac["sigma"],

    "mu_star_conf":
        Si_lac["mu_star_conf"]

})


df_fd.to_csv(
    "morris_FD_results.csv",
    index=False
)


df_lac.to_csv(
    "morris_lacunarity_results.csv",
    index=False
)


print("\nFD:")
print(df_fd)

print("\nLacunarity:")
print(df_lac)



# ==========================
# 7. Простая визуализация
# ==========================

plt.figure(figsize=(7,4))

plt.bar(
    df_fd["parameter"],
    df_fd["mu_star"]
)

plt.ylabel("mu*")
plt.title("Morris sensitivity: FD")

plt.savefig(
    "morris_FD.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



plt.figure(figsize=(7,4))

plt.bar(
    df_lac["parameter"],
    df_lac["mu_star"]
)

plt.ylabel("mu*")
plt.title("Morris sensitivity: Lacunarity")

plt.savefig(
    "morris_lacunarity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()