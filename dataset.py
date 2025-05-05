import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import json

# Authenticate Kaggle API
api = KaggleApi()
api.authenticate()

# Dataset slugs on Kaggle
datasets = {
    "sleep_health_lifestyle": "uom190346a/sleep-health-and-lifestyle-dataset",
    "sleep_data": "danagerous/sleep-data",
    "sleep_efficiency": "equilibriumm/sleep-efficiency",
    "student_sleep_patterns": "arsalanjamal002/student-sleep-patterns"
}

# Download and unzip using Kaggle API
for folder, dataset in datasets.items():
    os.makedirs(folder, exist_ok=True)
    api.dataset_download_files(dataset, path=folder, unzip=True)

# Load a sample dataset
def load_dataset(folder):
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            return pd.read_csv(os.path.join(folder, file))
    return pd.DataFrame()

# Load datasets
sleep_health_df =load_dataset("sleep_health_lifestyle")
sleep_data_df = load_dataset("sleep_data")
sleep_efficiency_df = load_dataset("sleep_efficiency")
student_sleep_df = load_dataset("student_sleep_patterns")

# Generate QA pairs
data = []

# Curated QA from Sleep Efficiency Dataset
for _, row in sleep_efficiency_df.iterrows():
    alcohol = row.get('AlcoholConsumption', 'unknown')
    efficiency = row.get('SleepEfficiency', 'unknown')
    prompt = f"Explain how alcohol consumption of {alcohol} relates to sleep efficiency."
    response = f"Alcohol consumption of {alcohol} is associated with a sleep efficiency of {efficiency}%."
    data.append({"input": prompt, "output": response})

# Instruction-style from Student Sleep Patterns
for _, row in student_sleep_df.iterrows():
    age = row.get('Age', 'unknown')
    duration = row.get('Sleep_Duration', 'unknown')
    caffeine = row.get('Caffeine_Intake', 'unknown')
    quality = row.get('Sleep_Quality', 'unknown')

    prompt = f"Analyze a student aged {age} with caffeine intake level {caffeine}."
    response = f"The student sleeps for {duration} hours and reports sleep quality as {quality}."
    data.append({"input": prompt, "output": response})

# Descriptive instruction from Sleep Health Dataset
for _, row in sleep_health_df.iterrows():
    try:
        duration = row['SleepDuration']
        age = row['Age']
        gender = row['Gender']
        stress = row.get('StressLevel', 'unknown')

        prompt = f"Describe sleep behavior of a {age}-year-old {gender} with stress level {stress}."
        response = f"The individual sleeps {duration} hours per night with a stress level of {stress}."
        data.append({"input": prompt, "output": response})
    except:
        continue

# Save the hybrid training dataset
with open("sleep_data_for_finetuning.json", "w") as f:
    json.dump(data, f, indent=4)
