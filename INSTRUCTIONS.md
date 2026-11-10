# AgriSense Cloud setup and operating instructions

This document is the detailed companion to [README.md](README.md). Follow the sections you need; Google Earth Engine, real image models, and Google Cloud are optional.

## 1. Local setup (Windows)

1. Install Git and Python 3.10–3.12. Python 3.11 or 3.12 is recommended for the TensorFlow dependency.
2. Clone the repository and open its root folder in VS Code:

   ```powershell
   git clone https://github.com/<your-username>/agrisense-cloud.git
   cd agrisense-cloud
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. If PowerShell blocks environment activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or follow Microsoft's execution-policy guidance for your machine.
4. Generate the demo data and baseline models:

   ```powershell
   python train_and_save.py
   ```

   This step can take several minutes and may use substantial memory because it trains image models as well as tabular models. Do not rerun it over existing real image models unless you intentionally want to replace them. For only the yield model, use `python retrain_yield.py`.
5. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

   Open `http://localhost:8501`.

## 2. Start the dashboard and API together (Windows)

1. Copy `start_all.example.bat` to `start_all.bat`.
2. Edit the copied file and set your own values. Do not commit the real file or credentials.
3. Run `start_all.bat`. The API is available at `http://127.0.0.1:8000`; interactive API documentation is at `/docs`.
4. To use the API from the dashboard, set `AGRISENSE_API_URL` to the API base URL as shown in the example batch file.

## 3. Google Earth Engine and daily NDVI ingestion

Earth Engine is optional. You need an eligible Google Cloud project registered for Earth Engine and the Earth Engine Python API installed (included in `requirements.txt`). Authenticate locally:

```powershell
earthengine authenticate
$env:GEE_PROJECT="your-google-cloud-project-id"
```

Register fields in the dashboard's Satellite NDVI page. The field list is stored in `data/fields.csv`; the scheduled job writes observations to `data/ndvi_observations.csv`.

To run one ingestion manually:

```powershell
python ingest_satellite.py
```

For Windows Task Scheduler, copy `run_ingestion.example.bat` to `run_ingestion.bat`, edit the project path and project ID, then create a daily task that runs the copied batch file. Keep the real batch file private; `.gitignore` excludes it.

## 4. Train image models in Google Colab

1. Open `notebooks/AgriSense_Colab_Training.ipynb` in Google Colab.
2. Select a GPU runtime if available and run the notebook cells in order. Dataset downloads may require Kaggle access/credentials and take time.
3. Download `agrisense_image_models.zip` from Colab and extract its model files into the repository's `models/` directory.
4. Run `python merge_image_meta.py` from the repository root, then restart Streamlit/API processes.

The validation metrics are from the notebook's dataset split. They are not evidence of field performance.

## 5. Generate demonstration images (optional)

Run `notebooks/colab_make_example_images.py` in a Google Colab runtime. It downloads PlantVillage and EuroSAT data via `kagglehub`, creates `examples_images.zip`, and offers it for download. Extract the `examples/` directory into the repository root so the final paths are `examples/leaf/` and `examples/landcover/`.

These images are illustrative samples from the training datasets, not independent test images. The optional packages used by this script are installed in the Colab runtime by the script itself; they are not required by the main application.

## 6. Optional crop-recommendation dataset

Place the permitted Kaggle Crop Recommendation CSV at `data/crop_recommendation.csv`. It must contain the feature columns expected by the crop recommender and a target column named `label`. Review the dataset's license and terms before using or redistributing it. If the file is absent, the project uses generated demonstration data as described in the README.

## 7. Optional Google Cloud deployment

The `deploy/deploy.sh` script is a deployment template for Google Cloud Run, Cloud Storage, and BigQuery; it is not a one-click deployment guarantee. Before running it:

- Install and configure the Google Cloud CLI (`gcloud`, `gsutil`, and `bq`).
- Select a project with billing enabled and permissions to enable services, create buckets/tables, and deploy Cloud Run services.
- Train models locally and confirm `models/meta.json` exists.
- Review the script, project, region, service account permissions, and public-access settings.
- Authenticate with `gcloud auth login`, then run from the repository root, for example:

  ```bash
  PROJECT=your-project-id REGION=asia-south1 ./deploy/deploy.sh
  ```

The script currently deploys unauthenticated Cloud Run services. Do not expose sensitive data or treat the demo password gate as production-grade security. Review IAM and application authentication before public deployment. Cloud usage may incur charges.

## 8. Run tests

```powershell
pip install -r requirements-dev.txt
python -m pytest -q tests
```

Some tests are intentionally skipped until model artifacts have been trained. A passing unit-test suite does not validate external services such as Earth Engine, Kaggle, Cloud Storage, BigQuery, or Cloud Run.

## 9. Before pushing to GitHub

- Keep `.env`, credentials, private batch files, virtual environments, generated datasets, and trained model binaries out of Git.
- Check `git status` and inspect every staged file before committing.
- Do not upload datasets unless their licenses permit redistribution.
- Update the repository URL in the README after creating the GitHub repository.
- Include screenshots and source code, but do not claim a cloud deployment or real-farm accuracy unless independently verified.
