import pandas as pd

# Carica il CSV
df = pd.read_csv("src/project_name/eliminateNoData/pixel_nodata_pre.csv")  # Sostituisci con il tuo nome file

yes_rows = df[df["supera_soglia"] == "YES"]

count_yes = len(yes_rows)
total = len(df)
percent_yes = (count_yes / total) * 100 if total > 0 else 0

print(f"Totale Landsat YES: {count_yes}")
print(f"Percent: {percent_yes:.2f}%")

yes_rows["file_path"].to_csv("src/project_name/eliminateNoData/ToBeDeletedPre.txt", index=False, header=False)

print("Log saved in ToBeDeleted.txt'")
