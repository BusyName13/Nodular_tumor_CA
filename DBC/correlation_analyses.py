#Iwanaga, T., Usher, W., & Herman, J. (2022). Toward SALib 2.0:
#Advancing the accessibility and interpretability of global sensitivity analyses.
# Socio-Environmental Systems Modelling, 4, 18155. doi:10.18174/sesmo.18155

#Herman, J. and Usher, W. (2017) SALib:
# An open-source Python library for sensitivity analysis. Journal of Open Source Software,2(9).
# doi:10.21105/joss.00097
import numpy as np
import pandas as pd
from pathlib import Path

from SALib.sample.morris import sample
from SALib.analyze.morris import analyze

import differential_box_counting
import lacunarity
import simulation_generator


# ============================
# Параметры Morris
# ============================

problem = {
    "num_vars": 7,

    "names": [
        "O2_DIFFUSION_K",
        "AGE_HEALTHY_ADULT",
        "AGE_HEALTHY_G2",
        "H_TUMOUR_TO_QUISC_LIMIT",
        "AGE_PROLIF_TUMOUR_G2",
        "VOLUME_MCF10A",
        "VOLUME_MCF7"
    ],

    "bounds": [
        [1.82e-5, 2.68e-5],
        [9, 17],
        [5, 9],
        [24, 28],
        [4, 8],
        [6.78e-10, 13.17e-10],
        [33.75e-10, 168.73e-10]
    ]
}

N = 300
NUM_LEVELS = 4

OUTPUT_DIR = Path(
    r"C:\Users\Lenovo\Documents\INNO-dir\Tumors\Nodular_tumor_CA\result_pictures"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================
# Генерация выборки Morris
# ============================

param_values = sample(
    problem,
    N=N,
    num_levels=NUM_LEVELS,
)

print(f"Будет выполнено {len(param_values)} запусков.")


# ============================
# Запуск модели
# ============================

dbc_results = []
lac_results = []

for i, params in enumerate(param_values):
    # запуск модели
    image = simulation_generator.do(params)

    # вычисление характеристик
    fd = differential_box_counting.DBC(image)
    lac = lacunarity.calculate_lacunarity(image)

    dbc_results.append(fd)
    lac_results.append(lac)

    if (i + 1) % 50 == 0:
        print(f"{i+1}/{len(param_values)}")
    if i % 10 == 0:
        print(i, fd, lac)

dbc_results = np.asarray(dbc_results)
lac_results = np.asarray(lac_results)

# ============================
# Анализ Morris
# ============================

dbc_analysis = analyze(
    problem,
    param_values,
    dbc_results,
    num_levels=NUM_LEVELS,
    print_to_console=False,
)

lac_analysis = analyze(
    problem,
    param_values,
    lac_results,
    num_levels=NUM_LEVELS,
    print_to_console=False,
)

# ============================
# Формирование результатов Morris
# ============================

dbc_df = pd.DataFrame({
    "parameter": problem["names"],
    "mu": dbc_analysis["mu"],
    "mu_star": dbc_analysis["mu_star"],
    "sigma": dbc_analysis["sigma"],
})

lac_df = pd.DataFrame({
    "parameter": problem["names"],
    "mu": lac_analysis["mu"],
    "mu_star": lac_analysis["mu_star"],
    "sigma": lac_analysis["sigma"],
})


# сортировка по важности параметра (mu_star)
dbc_df = dbc_df.sort_values("mu_star", ascending=False)
lac_df = lac_df.sort_values("mu_star", ascending=False)


# ============================
# Вывод в консоль
# ============================

print("\n==============================")
print("Morris analysis for DBC")
print("==============================")
print(dbc_df.to_string(index=False))


print("\n==============================")
print("Morris analysis for Lacunarity")
print("==============================")
print(lac_df.to_string(index=False))


# ============================
# Сохранение в txt
# ============================

with open(OUTPUT_DIR / "Morris_DBC_results.txt", "w") as f:
    f.write("Morris analysis for DBC\n\n")
    f.write(
        dbc_df.to_string(index=False)
    )


with open(OUTPUT_DIR / "Morris_Lacunarity_results.txt", "w") as f:
    f.write("Morris analysis for Lacunarity\n\n")
    f.write(
        lac_df.to_string(index=False)
    )
