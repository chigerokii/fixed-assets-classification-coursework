"""Графики к README по итоговой таблице курсовой (таблица 1)."""
import matplotlib.pyplot as plt
import pandas as pd

rows = [
    (1, "Logistic Regression", 21.474, 0.999, 0.992, 0.985),
    (1, "KNN", 1.858, 0.997, 0.991, 0.983),
    (1, "Decision Tree", 1.916, 1.000, 0.997, 0.995),
    (1, "CatBoost", 138.246, 1.000, 0.999, 0.998),
    (2, "Logistic Regression", 40.943, 0.965, 0.900, 0.921),
    (2, "KNN", 1.785, 0.971, 0.917, 0.936),
    (2, "Decision Tree", 3.051, 0.938, 0.909, 0.930),
    (2, "CatBoost", 90.795, 0.982, 0.941, 0.954),
    (3, "Logistic Regression", 70.930, 0.965, 0.850, 0.855),
    (3, "KNN", 1.042, 0.986, 0.920, 0.919),
    (3, "Decision Tree", 2.074, 0.991, 0.963, 0.963),
    (3, "CatBoost", 254.263, 0.997, 0.961, 0.961),
]
df = pd.DataFrame(rows, columns=["dataset", "model", "time_s", "roc_auc", "accuracy", "f1"])
names = {1: "1. Трубопроводы (2 класса)", 2: "2. ЛЭП и кабельные линии (2 класса)", 3: "3. Энергообъекты (4 класса)"}
colors = {"Logistic Regression": "#9aa5b1", "KNN": "#2a6fdb", "Decision Tree": "#e07a1f", "CatBoost": "#7b4fc9"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
for ax, (ds, g) in zip(axes, df.groupby("dataset")):
    for _, r in g.iterrows():
        ax.scatter(r.time_s, r.f1, s=110, color=colors[r.model], zorder=3)
        ax.annotate(r.model, (r.time_s, r.f1), textcoords="offset points", xytext=(7, -4), fontsize=9)
    ax.set_xscale("log")
    ax.set_title(names[ds], fontsize=11)
    ax.set_xlabel("время обучения, с (лог. шкала)")
    ax.grid(alpha=0.3)
    lo = g.f1.min()
    ax.set_ylim(lo - 0.02, 1.005)
axes[0].set_ylabel("F1 на тесте")
fig.suptitle("CatBoost даёт максимум метрик, но обучается в 50–250 раз дольше KNN и дерева", fontsize=13)
fig.tight_layout()
fig.savefig("images/quality_vs_time.png", dpi=150)

# Прирост CatBoost над лучшей быстрой моделью
gain = []
for ds, g in df.groupby("dataset"):
    fast = g[g.model.isin(["KNN", "Decision Tree"])].sort_values("f1").iloc[-1]
    cb = g[g.model == "CatBoost"].iloc[0]
    gain.append((ds, fast.model, round((cb.f1 - fast.f1) * 100, 1), round(cb.time_s / fast.time_s)))
print(pd.DataFrame(gain, columns=["dataset", "best_fast", "catboost_f1_gain_pp", "time_ratio"]))
