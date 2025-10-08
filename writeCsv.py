import csv

def write_csv(filename, Xs, Ys, Xname, Yname):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'{Xname}', f'{Yname}'])
        for x, y in zip(Xs, Ys):
            writer.writerow([x, y])
    print(f"✅ wrote {filename}")