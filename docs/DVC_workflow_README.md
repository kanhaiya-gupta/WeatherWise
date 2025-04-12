Perfect — here's a full, from-scratch **DVC workflow README section**, guiding users from `dvc init` to `dvc repro`, including setting up remote (like DagsHub). This is clean, beginner-friendly, and directly tailored to your WeatherWise project.

---

## 📦 Data & ML Pipeline Versioning with DVC

This project uses [**DVC (Data Version Control)**](https://dvc.org) to manage datasets, models, and reproducible ML pipelines.

---

### 🛠️ Step-by-Step DVC Setup

#### 1. ✅ Initialize DVC in Your Project
```bash
dvc init
git commit -m "Initialize DVC"
```

---

#### 2. 📁 Add Data to DVC
Let’s say you have a dataset at `data/raw/weather.csv`:
```bash
dvc add data/raw/weather.csv
```

This will:
- Track the file with DVC
- Create `data/raw/weather.csv.dvc`
- Add entry to `.gitignore`

Then commit:
```bash
git add data/raw/weather.csv.dvc .gitignore
git commit -m "Track dataset with DVC"
```

---

#### 3. 🌐 Setup DVC Remote (DagsHub Example)

Use [DagsHub](https://dagshub.com) for remote data storage:

```bash
dvc remote add -d origin https://dagshub.com/kanhaiya-gupta/WeatherWise.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user kanhaiya-gupta
dvc remote modify origin --local password <your_dagshub_token>
```

Then push data:

```bash
dvc push
```

---

#### 4. 🧱 Define the DVC Pipeline

Create a `dvc.yaml` to define pipeline stages (e.g. preprocess → train → evaluate). Here’s a sample stage:

```yaml
stages:
  preprocess:
    cmd: python src/preprocessing/preprocess.py
    deps:
      - src/preprocessing/preprocess.py
      - data/raw/weather.csv
    outs:
      - data/processed/preprocessed.csv
```

Track it:
```bash
git add dvc.yaml dvc.lock
git commit -m "Add DVC pipeline: preprocess"
```

---

#### 5. 🔁 Run the Pipeline

```bash
dvc repro
```

DVC will:
- Detect changed dependencies
- Re-run only the necessary stages
- Track new outputs (e.g. processed data, models)

---

#### 6. 📤 Push Everything to Remote

```bash
dvc push      # Push data, models, etc.
git push      # Push code, .dvc files, dvc.yaml, etc.
```

---

### ✅ Summary of Key DVC Commands

| Task                        | Command                          |
|-----------------------------|----------------------------------|
| Initialize DVC              | `dvc init`                       |
| Track a data file           | `dvc add <path>`                 |
| Set remote storage          | `dvc remote add -d origin <url>` |
| Push data to remote         | `dvc push`                       |
| Pull data from remote       | `dvc pull`                       |
| Run pipeline                | `dvc repro`                      |
| Visualize pipeline          | `dvc dag`                        |

