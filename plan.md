# ParkWatch — Project plan

**Goal:** end-to-end MLOps pipeline (capture → storage → labelling → training → periodic retraining → deployment) to classify parking spot occupancy from an ESP32-CAM camera.

**Context:** second MLOps portfolio project, complementary to `tinyConvNeXt` (Azure ML + self-hosted MLflow). ESP32-CAM camera with an OV5640 sensor, placed indoors, looking through a window at an outdoor car park.

**Status:** the full loop runs on Azure ML every night (capture → pre-annotation → human review → retraining → promotion), with per-photo alignment against camera drift. Next: model optimisation (p04/p05 overlap) once the 23/10 + 3/11 holdout decides the next `champion`.

---

## Decisions taken

- **Project name:** ParkWatch
- **Capture hardware:** ESP32-CAM (OV5640), needs the Arduino IDE to flash the firmware and probably a USB-TTL adapter
- **Upload path:** ESP32 → Raspberry Pi (local HTTP, no TLS) → the Pi uploads to Azure Blob Storage
  - Reason: avoid fragile TLS/certificates on the ESP32 (limited memory); the Pi allows logging, retries, resize/compress and metadata before uploading
- **Capture cadence:** 1 photo every 3 minutes (portfolio/dataset phase, not a live product)
- **Capture schedule:** the ESP32 knows nothing about schedules — it captures every 3 min continuously (24h) and sends everything to the Pi. The Pi decides, photo by photo, whether it falls inside the daylight window using `astral` (civil twilight `dawn`/`dusk`, with a ~15 min margin before/after, computed day by day — it adapts to the seasons without touching code). Outside the window the photo is dropped and not uploaded to Blob. The check happens in the same Pi service that receives the photo from the ESP32, before uploading to Blob.
- **Labelling:** manual by the user, with Label Studio, using the model predictions as pre-annotation (review, not labelling from scratch)
- **Inference/deployment strategy (v1):**
  - **NO** live inference from the start — serving a low-quality model fast makes no sense
  - Daily process: Azure ML **inference pipeline** (a component that loads the `champion` from the registry) processes the new photos → predictions as Label Studio tasks (see Phase 4; replaces the Batch Endpoint planned initially)
  - Manual review/correction in Label Studio
  - Corrected labels feed the retraining pool
  - **Nightly retraining only if the dataset changed** (Azure ML training pipeline): retraining from scratch, evaluation against a fixed holdout, promotion to `champion` if it matches or beats the macro F1
  - Historical reprocessing: by submitting the inference pipeline with other dates
- **Live inference (v2, future):** only once the model is accurate enough — event-driven architecture (Azure Function with a Blob trigger) + a lightweight query API that reads a cached state (Table Storage), not an always-on Managed Online Endpoint
- **ROIs:** fixed per spot (PKLot/CNRPark style), defined once on a reference image after fixing the camera position
- **Privacy:** licence plate blurring is mandatory before sharing the dataset publicly

## Open points

- Field test with the camera mounted: number of spots in frame, angle/perspective, sharpness through the glass, reflections/backlight at different times of day
- Exact coordinates for `astral` (Zubiri vs. Pamplona — confirm the final camera location)
- How many days/photos of seed dataset before the first training
- Container/folder structure in Blob (raw/, labeled/, manifest)

---

## Phases

### Phase 0 — Physical setup
- Fix the final camera position with a home-made mount, at the window already chosen (it must not move once the ROIs are set)
- Field test: minimal capture firmware + individual photos at different times of day to validate the number of spots in frame, angle/perspective, sharpness and reflections/backlight
- Capture schedule: **decided** — ESP32 without schedule logic (captures 24h every 3 min), the Pi filters with `astral` (civil twilight `dawn`/`dusk` + ~15 min margin) before uploading to Blob
- Capture a reference image and define fixed ROIs per spot (JSON config with coordinates; simple rectangle or 4-point polygon depending on the angle shown by the field test)
- Validate the framing: coverage of every target spot, no obstructions (reflections, branches, etc.), at several times of day

### Phase 1 — Capture (ESP32 → Pi → Blob)
- ESP32-CAM firmware: periodic capture (every 3 min), POST to a local endpoint on the Pi
- Server on the Pi (Flask/FastAPI): receives the photo, logging, retries, resize/compress, uploads to Azure Blob with metadata (timestamp, consistent name)
- Blob structure: `raw/` container

**Status: code written, not yet tested on hardware.**
- `esp32_firmware/parkwatch_capture/parkwatch_capture.ino` — firmware without schedule logic (captures every 3 min non-stop, ESP32S3_EYE pinout already validated in earlier tests with the Freenove ESP32-S3-WROOM CAM). Retries with backoff if the POST or the WiFi fails. Requires creating `parkwatch_secrets.h` (SSID/password/Pi IP) from the `.example`.
- `pi_receiver/app.py` — FastAPI service (`uvicorn app:app --host 0.0.0.0 --port 8000`):
  - Validates that the received JPEG is not truncated/corrupt before processing it
  - Filters by daylight with `astral` (civil twilight `dawn`/`dusk` ± `PW_MARGIN_MIN`, computed day by day) — outside the window the photo is dropped and not uploaded to Blob
  - Resizes/recompresses with Pillow (`PW_MAX_WIDTH`, `PW_JPEG_QUALITY`) before uploading
  - Uploads to Blob with metadata (`captured_utc`) and a date-partitioned name (`YYYY/MM/DD/parkwatch_<timestamp>.jpg`)
  - If the Blob upload fails, stores the photo in `pi_receiver/spool/` and a background thread retries every `PW_RETRY_EVERY_S`
  - Config via `.env` (see `pi_receiver/.env.example`); default coordinates set to Zubiri — **to be confirmed** whether the camera ends up there or in Pamplona
- Dependencies: `fastapi`, `uvicorn`, `azure-storage-blob`, `astral`, `pillow`, `python-dotenv` (`pi_receiver/requirements.txt`)
- Pending before closing Phase 1: test against the real camera once mounted (Phase 0), confirm that the `raw` container exists in the Storage Account, and decide where `pi_receiver` runs persistently on the Pi (systemd service, suggested)

**ROIs v2026-10-17 (defined):** 5 spots drawn in Label Studio (`parkwatch-rois` project, `label_studio/roi_config.xml`) on a real ESP32 photo (1600×1200).
- Each ROI is the **footprint of the spot on the ground**, not the car silhouette → maximum overlap 15% (p04/p05).
- `tools/rois_from_ls.py` converts the JSON export into `label_studio/rois.json` (points in %, ids `p01..p05` from left to right, overlap report). `rois.json` is the **source of truth for the geometry**: training crops with it and only the label is read from the annotations.
- If the camera moves → new `rois.json` with another `version`.

### Phase 2 — Initial labelling + dataset
- **Label Studio** deployed on an Azure Container App (`park-watch-label-studio`, RG `AML`), `labelstudio` DB on the PostgreSQL server `mlflowtest` (legacy name; it no longer hosts MLflow), images served from Blob with pre-signed URLs (CORS configured on the Storage Account).
- Labelling project `parkwatch-labels` with `label_studio/labeling_config.xml` (`PolygonLabels`: `free` / `occupied` / `exclude`). Azure storage connected only to resolve `azure-blob://`, **without Sync**: every task comes in by script.
- `tools/make_ls_tasks.py` (removed after the hand-over to the inference pipeline) created one task per photo with the 5 spots pre-loaded (default `occupied`, the majority class) and imported them through the API. **Every** photo is imported (~250 tasks/day).
- Review: change only the wrongly labelled spots (keys 1/2/3). Labels follow whether there is a car on the ground of the spot, not what the crop shows.
- Bootstrapping: full manual labelling from day 0 with real photos in Blob (after Phase 0)
- Train/holdout definition (fixed holdout, never relabelled, as in tinyConvNeXt)
- Tool: Label Studio, free/occupied classification per ROI/spot

### Phase 3 — Initial training
- Model: binary classifier per ROI patch (not detection) — lightweight architecture
- Tracking in the MLflow integrated in the Azure ML workspace (see block 3)

**History — local pipeline v1 (2026-10-21), deleted on 2026-10-27 (block 5):** the description is kept as a record of what was learnt.
- `training/build_dataset.py`: Label Studio export + Blob photos (cache in `data/raw_cache/`) + `rois.json` → per-spot crops in `data/parkwatch_crops/` with `manifest.csv` and `dataset_info.json`. Latest non-cancelled annotation per task; `exclude` left out. Crop = polygon bbox + 15% margin (`training/crops.py`, shared with future inference).
- Split by **whole days**: holdout if `date.toordinal() % 5 == 0` (~20%, deterministic; the holdout grows with new days). The first holdout is 16/10.
- `training/train.py`: transfer learning (`mobilenet_v3_large` by default; also `resnet18`, `convnext_tiny`), inverse class weights, light colour/affine augmentation. Logs to MLflow (experiment `parkwatch`): params, dataset (manifest + rois), per-epoch metrics, F1/recall per class, accuracy per spot, confusion matrix and holdout error grid. Registers `parkwatch-spot-classifier` (pyfunc `SpotClassifier`: input (N,3,H,W) in [0,1], resizes and normalises inside).
- `training/promote.py`: evaluates new and champion on the current holdout and moves the `champion` alias if macro F1 ≥.
- v1 (5 days, mobilenet, 8 epochs): holdout macro F1 0.912, acc 0.973, `free` recall 0.78. Nearly all errors are washed-out/low-contrast photos (16/10 ~13:30–14:35Z) labelled `free` → they should be `exclude`.
- Known limitation: p02 and p03 have never been free in the dataset → their accuracy measures nothing yet.

**Pipeline redesign (2026-10-23):** pipeline v1 stays as a reference; the new one is defined decision by decision and runs **entirely in the cloud** (Azure ML). Order: 1) data and evaluation, 2) model, 3) cloud infrastructure, 4) orchestration, 5) what to do with the existing pieces.

Decisions taken — block 1 (data and evaluation):
- **Example = polygon bbox + 10% margin per side, keeping the aspect ratio** (letterboxed to a square, no stretching). Deterministic: identical in train, validation, holdout and inference. `crop_margin` is a pipeline parameter and is logged in MLflow.
- Jitter (shift/scale) only as a training augmentation and only if there is evidence it is needed.
- **Dark / washed-out photos: every labelled one goes in** (the user reviewed them one by one; if the spot could not be seen, it was marked `exclude` or the photo skipped). No quality filter.
  - `build_dataset` computes **brightness and contrast per crop** and stores them in `manifest.csv`.
  - Evaluation reports **metrics split by quality** (good light vs. dim/washed-out).
  - Their **impact on the model** is measured: metrics per quality bucket on the holdout + an experiment training with and without them. If they add noise, the decision (keep, weight or filter) is taken with that data, without touching labels.
- **Splits, always by whole days:**
  - **Validation = cross-validation by day** (grouped k-fold, ~4 folds leaving out ~2 days each time) to pick epochs and hyperparameters; the final model is retrained on every training day.
  - **Holdout: pending, decided when data collection ends.** Whole days **chosen by hand** by their content (never individual photos: their almost identical neighbours would stay in train). It will be fixed (a day that goes in never comes out) and must cover free spots in as many spots as possible; scarce days (e.g. the two with p03 free, 22 and 23/10) are split between holdout and train. Candidates as of 2026-10-26: 19+22 or 19+23. While it does not exist, there is no promotion to `champion`.
    - **Decided (2026-10-30): holdout = 2026-10-23** (a single day, the most varied; `HOLDOUT_DAYS` in `aml/pipelines/training.py`). Free spots in 4 spots (p01, p03, p04, p05), 11 free episodes, 283 low-contrast + 70 dark crops; 1,146 crops (7% of `parkwatch-crops:3`). It leaves 22/10 in train with 47 of the 60 free p03 crops. Limits: the p03 holdout metric comes from 13 free crops; p02 is not measured (never free). Days may be added later; when the holdout changes, the `champion` is re-evaluated on the new one before comparing. Generated with `tools/holdout_stats.py`.
- **Near-duplicate photos (camera every 3 min): all of them are used** in train. Reasons: negligible compute cost, it is the real production distribution, no light variation within an episode is lost, and the overfitting of v1 is explained mostly by the lack of validation.
  - `build_dataset` still computes the **episode** of each crop (consecutive run of photos of a spot with the same label, per day; `exclude` does not break it) and stores `episode_id` and its duration in `manifest.csv`.
  - Error analysis checks whether short episodes or unusual light fail more; only then is a visual-similarity deduplication tried as an experiment (keep a crop only if it differs enough from the last one kept).
  - Validation and holdout always use every photo.
- **Decision metric: macro F1** (free/occupied, over every holdout crop), higher is better.
  - No per-spot or per-quality guardrails for now (they are still logged as informative metrics).
  - **Promotion to `champion` if the new model matches or beats** the champion's macro F1, both evaluated on the same holdout.
  - No absolute minimum for now (to be set before Phase 5, with the first results).

Block 1 closed except for the holdout days (pending the end of data collection). Next: block 2 (model).

Decisions taken — block 2 (model):
- **No transfer learning: custom CNN trained from scratch, designed and built by the user.** Evidence that it is viable: mAlexNet (Amato et al., CNRPark, 2017), a small CNN trained from scratch for parking occupancy.
- **ParkNet architecture** (~583 k parameters): RGB 96×96 input → 4 blocks `[conv 3×3 → BatchNorm → ReLU] ×2 → maxpool 2×2` with 32/64/128/128 filters → global average pooling → dropout 0.3 → linear to 2 logits (`free`, `occupied`, softmax).
- **Training:** cross-entropy with **class weights** inverse to frequency; AdamW (lr 1e-3, weight decay 1e-4); cosine learning rate down to 0 over a maximum of 40 epochs; batch 128. The epochs of the final model are picked by **cross-validation by day** (best mean macro F1) and the final model uses the same LR schedule, stopping at that epoch.
- **Augmentations (train only):** brightness/contrast/saturation ±40% and horizontal flip, on from the start. Crop position/scale **jitter** as a hyperparameter, starting at 0.
- **Normalisation:** per-channel mean and std of the training crops (not ImageNet).
- **Hyperparameters to compare in MLflow:** network width (filter multiplier), input size, `crop_margin` and `jitter`.
- Code in `ml/` (self-contained so it can be uploaded as an Azure ML job's code): `parknet.py`, `crops.py`, `data.py`, `metrics.py`, `build_dataset.py`, `train.py`, `inference.py`. Training and inference crops verified identical pixel by pixel.

**Two-step pipeline (decided):** `build_dataset` on the CPU cluster (Label Studio export + Blob photos + `rois.json` → crops stored with a generous 25% margin, manifest with brightness/contrast/episodes) → **versioned data asset** in Azure ML → `train` on the GPU cluster. The final margin, letterbox, normalisation and augmentations are applied inside training, so they remain hyperparameters without rebuilding the dataset. The dataset is only rebuilt when labels or ROIs change. Compute, credentials and orchestration details: blocks 3 and 4.

Block 2 closed. Next: block 3 (cloud infrastructure).

Decisions taken — block 3 (cloud infrastructure):
- **New Azure ML workspace dedicated to ParkWatch** (the tinyConvNeXt one is not reused), in **Spain Central**, in a **new resource group**. Everything defined with **Bicep** in the repo (`infra/`). Associated resources as needed: own Storage Account and Key Vault (mandatory for the workspace), Container Registry when creating the environments; Application Insights only if the API requires it.
  - *Update 2026-10-29 (implementation):* West Europe was planned, but it does not accept new customers on this subscription (`NotAvailableForSubscription`). Spain Central does have NC4as_T4_v3; what was missing was Azure ML quota (0 T4 vCPUs, requested through a support ticket). Advantage: workspace and photos in the same region.
  - Deployed (`infra/main.bicep`): RG `parkwatch`, workspace `parkwatch-workspace`, Storage `parkwatchstorage`, Key Vault `parkwatch-keyvault` and, because Azure ML requires it when creating the workspace, Application Insights `parkwatch-appinsights` on Log Analytics `parkwatch-logs` (30-day retention).
  - Azure ML quotas in Spain Central: DASv4 10 vCPU, total dedicated 16, T4 4 (granted through a support ticket on 2026-10-29).
  - Deployed (step 3): identity `parkwatch-identity` with *Key Vault Secrets User*, *Storage Blob Data Reader* on `parkwatch` (RG `AML`) and *AzureML Data Scientist* on the workspace (added: without it the jobs cannot register data assets or models). Key Vault in **RBAC** mode; the user has *Key Vault Secrets Officer*; the workspace already received *Key Vault Administrator* when it was created. `cpu-cluster` (D4as_v4, 0–1 nodes, shuts down after 5 min). `gpu-cluster` (NC4as_T4_v3, 0–1 nodes, shuts down after 5 min), created on 2026-10-29 once the T4 quota (4 vCPU) was granted.
  - Deployed (step 4): datastore `parkwatch_raw` (`raw` container of `parkwatch`), identity-based, no stored keys. Label Studio token in the Key Vault as `label-studio-token`.
  - Deployed (step 5): Container Registry `parkwatchregistry` (Basic, no admin user) linked to the workspace; *AcrPull* for the identity; the workspace already has *Azure AI Administrator* on the RG (covers image builds). Environments in `aml/environments/`: `parkwatch-data` (`openmpi5.0-ubuntu24.04:20261018.v1`, Python 3.11, azure-identity, azure-storage-blob, azure-keyvault-secrets, azure-ai-ml) and `parkwatch-train` (`openmpi5.0-cuda12.6-ubuntu24.04:20261018.v1`, torch 2.7.1+cu126, mlflow 2.22.0 + azureml-mlflow 1.60.0); `parkwatch-train` is also used for promotion and inference on CPU.
  - Verified (2026-10-29) with `aml/jobs/smoke-*.yml`: on `cpu-cluster` the identity reads the Key Vault and lists `raw`; on `gpu-cluster` PyTorch sees the Tesla T4 and MLflow logs to the workspace.
- **Compute** (in the same workspace, both with a minimum of 0 nodes so nothing is paid while idle):
  - CPU for `build_dataset`: `Standard_D4as_v4`, dedicated, 0–1 nodes (equivalent if unavailable: `D4as_v5`).
  - GPU for `train`: `Standard_NC4as_T4_v3` (T4), dedicated, 0–1 nodes (no low-priority quota). Watch in the first training whether the 4 vCPUs slow down data loading.
- **Data:** the `parkwatch` Storage Account (Spain Central, `raw` container) stays where it is and is registered as a **datastore** of the workspace (same region as the workspace). Access through a **managed identity** with the *Storage Blob Data Reader* role (no keys). `build_dataset` publishes a **versioned data asset** and only creates a new version **if labels or ROIs change** (export fingerprint + `rois.json`). `train` receives the dataset in **download** mode.
- **Environments:** two custom environments on official Azure ML base images, with versions pinned in the repo: `parkwatch-data` (CPU: azure-storage-blob, azure-identity, pillow, numpy) and `parkwatch-train` (GPU/CUDA: torch, torchvision, mlflow, numpy, pillow, matplotlib). *Basic* Container Registry.
- **Tracking and model registry: the workspace's integrated MLflow** (not the self-hosted one, deleted in block 5). Reasons: the jobs use it without configuration, job–data asset–model lineage, and registered models in Azure ML. Pending (block 4): check whether the registry supports aliases (`champion`) or use a version tag. *Superseded for experiment tracking on 2026-11-05 (block 6): train/promote log to a self-hosted MLflow again; the model registry stays here.*
- **Secrets:** the Label Studio token in the **workspace Key Vault**. **One user-assigned identity shared** by both clusters, with *Storage Blob Data Reader* on `parkwatch` and *Key Vault Secrets User* on the Key Vault.
- **Pending code changes** (done later): `build_dataset.py` must read Blob with the managed identity instead of the connection string, read the Label Studio token from the Key Vault and compute the dataset fingerprint.

Block 3 closed. Next: block 4 (orchestration).

Decisions taken — block 4 (orchestration):
- **Two independent pipelines** in the workspace (if the training one fails, inference keeps going with the previous `champion`):
  - **Training:** `build_dataset` (CPU) → `train` (GPU) → `evaluate & promote` (CPU).
  - **Inference:** `batch_inference` (CPU, loads whatever `champion` is in the registry) → `push_to_label_studio` (CPU): creates the tasks with the model predictions as pre-annotation and backfills missing days (as the Pi did with its 3-day window).
- **The Pi only captures and uploads photos.** Transition: while there is no `champion` (no holdout yet), the Pi keeps importing tasks as before; with the first `champion` its timer is disabled and the inference pipeline takes over.
- **Triggering: Azure ML schedules** (Madrid time zone): training every night at 02:00; inference at 04:00; weekly cross-validation on Sunday night (`run_cv=true`). Any of them can be submitted by hand (Studio or `az ml job create`), also with `run_cv=true`.
- **Nightly retraining only if the dataset changed** (data asset fingerprint): without new labels `train` and promotion are skipped; inference runs anyway.
  - *Implementation (2026-10-29), option B:* a lightweight `check_dataset` step (CPU) exports Label Studio, computes the **fingerprint** (SHA-256 of final photo/spot/label labels including `exclude`, `rois.json`, `STORE_MARGIN` and `quality_margin`; no annotation timestamps) and compares it with the one stored in `dataset_info.json` of the latest version of the `parkwatch-crops` data asset. A conditional pipeline node (`dsl.condition`, Python SDK; control flow in preview) only runs `build_dataset → train → promote` if it changed. Azure ML registers the data asset as the output of `build_dataset`. If control flow caused problems: `train` and `promote` would exit early (option A).
  - *Verified (2026-10-29)*, `aml/pipelines/training.py` + `aml/components/`: first run `changed=True` → `build_dataset` (15,051 crops, 13 days) → `parkwatch-crops:1` registered automatically; second run without changes → `changed=False` and `build_dataset` skipped. Details: the control output is written to the run property `azureml.pipeline.control` (as mldesigner does, without depending on it); the SDK needs `AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true` to serialise the `if_else`; the pipeline uses `force_rerun` (otherwise Azure ML reuses `check_dataset` from a previous run and repeats its `changed`); `parkwatch-data:3` pins `azure-storage-blob==12.19.0` because `azureml-mlflow` 1.60 requires <=12.19.0.
  - *Decided (2026-10-30):* `build_dataset` **rebuilds the whole dataset** every time (13 days ≈ 7.5 min including start-up; it grows with the labelled days). Simpler and never out of sync with Label Studio. **Switch to incremental when `build_dataset` exceeds ~30 min:** compare photo by photo with the manifest of the previous version (new photos, labels changed also on old days, photos no longer there), crop only those and copy the rest from the previous version; full rebuild if `rois.json`, `STORE_MARGIN` or `quality_margin` change. Only `build_dataset` changes.
  - *First cloud training (2026-10-30)*, `train` on `gpu-cluster` inside the conditional branch (`aml/components/train.yml`): 15,079 crops over 14 days (16–29/10), no holdout. Cross-validation 4 folds × 40 epochs (~11 s/epoch on the T4, ~35 min in total): **best epoch 38, macro F1 0.935 ± 0.024** (fold 0 with 16/10 the lowest, 0.894). Final model with 38 epochs as the step output and MLflow artifact. Nights without `--run-cv` take the epochs of the last **finished** run with CV. Maximum times: 1 h per data step, 3 h `train`.
  - *Diagnosis (2026-10-31, CV out-of-fold, `crop_margin` 0.10):* acc 95.5%; 450 occupied predicted free and 228 the other way round. **p05 (87% acc) and p04 (93%) concentrate 85% of the errors**: with the camera at an angle, a car's body invades the crop of the neighbouring spot and the model does not know which one is "its" spot. p03 never recognises a free spot (only 47 free in development). Options in order: A) `crop_margin` 0; B) polygon mask channel as a 4th input; C) grey out everything outside the polygon; D) redraw the ROIs (discarded). `training.py` already accepts `--crop-margin`, `--jitter` and `--width`.
  - *Evaluate & promote (2026-11-02)*, `ml/promote.py` + `ml/registry.py` + `aml/components/promote.yml`, last step of the conditional branch (CPU). The Azure ML registry **does not support MLflow aliases** (404, tested with a throw-away job): the **champion is the `champion=true` tag** on a version of `parknet` (the new one is tagged first, then the previous one untagged). **Every trained model is registered** (option a) with tags: holdout F1 and days, training run, dataset fingerprint, ROIs, `crop_margin`, `width`, epochs. New model and champion are evaluated on the same holdout of the current dataset, each with its own `model_config.json`. **First champion: `parknet:1`, holdout (23/10) macro F1 0.970**, 31 epochs (from the CV with the holdout left out), 16,235 development crops over 14 days.
  - *Inference pipeline (2026-11-02)*, `aml/pipelines/inference.py` (independent from the training one, as in block 4): `batch_inference` (CPU, `parkwatch-train`; champion from the registry, photos of the last 3 UTC days or `--days`) → `push_to_label_studio` (CPU, `parkwatch-data`). **Decided by the user:** annotated tasks are never touched; in pending ones the **labels of the existing pre-annotation are replaced** by the champion's (same prediction, same geometry and `model_version` `rois-…`; no new versions, no project changes, no new project). Each region carries its probability and the prediction the photo's lowest confidence (to sort in Label Studio and review the doubtful ones first). Photos without a task: created by the Pi until the hand-over; then `--create-missing`. First test (31/10 and 1/11): 484 tasks updated, 503 free spots (20.8%).
  - *Closing the loop (2026-11-02)*, `aml/pipelines/schedules.py` (Madrid time): `parkwatch-training-nightly` 02:00 every day except Monday (only if the dataset changed; epochs of the last run with CV); `parkwatch-training-weekly` Monday 02:00 with full CV, **forced even without new labels** (decided by the user); `parkwatch-inference` 04:00 every day with `create_missing` (creates the tasks of the new photos: hand-over from the Pi). Each schedule stores a copy of code, components and `rois.json`: after changing them, run the script again. **Alerts:** the Azure ML SDK does not expose email on jobs → Azure Monitor alert on the workspace `Failed Runs` metric (experiments `parkwatch-training` and `parkwatch-inference`) with an email action group, in `infra/main.bicep` (`alertEmail`). **Pi:** its import timer is disabled; from now on it only captures and uploads photos. *Verified by the user (2026-11-03): the scheduled loop works end to end.*
  - **Camera drift (2026-11-03).** The camera is glued in place (it cannot be screwed) and moves. Measured with `tools/camera_drift.py` (SIFT + RANSAC on `static` regions drawn in `parkwatch-rois` → `label_studio/static_mask.json`; best photo of 10:00–11:00 Madrid time, contrast with CLAHE; ECC discarded as too sensitive to light): 17–20/10 (0, −13) px; 21/10 reference; 22–26/10 slow drift up to (+9, −9) px and +0.5°; **28–29/10 jump of ~50 px and −3.4°**; afterwards it partly comes back (rotation ~−1.5°). For p04/p05 (100–150 px wide) that is half a spot. **Decided: align every photo** — `ml/align_photos.py` measures each photo only once (incremental table, data asset `parkwatch-alignment`); `ml/alignment.py` smooths (median of reliable measurements, ≥30 inliers, within ±60 min; otherwise the nearest reliable one) and **straightens the photo to the geometry of the reference** before cropping (same ROIs; the same in `build_dataset` and in inference). The alignment goes into the dataset fingerprint (format 2): when it changes, the dataset is rebuilt and retrained automatically. `drift_alert` fails (email) if the last day exceeds 40 px, 2° or <50% measurable photos, without stopping the rest (`continue_on_step_failure`). Environment `parkwatch-data:4` (+ `opencv-python-headless` 4.12). Pending: the p04/p05 overlap (mask channel), after seeing the effect of the alignment.
  - *First aligned training (2026-11-03, `--run-cv`):* `align` measured 4,396 photos (16/10–2/11). CV 17 days / 19,529 crops: **macro F1 0.960 ± 0.005** (before 0.935 ± 0.024), out-of-fold acc **97.6%** (before 95.5%), occupied predicted free 1.7% (before 3.7%), best epoch 26. Part of the gain comes from having more days; the fair comparison is `promote` on the same holdout. `drift_alert` fired on 2/11 (41.2 px > 40; 29% of photos with ≥30 inliers) although the preview of that day (19 inliers) is well aligned → **`MIN_INLIERS` 30 → 15** (it changes the fingerprint: the next run rebuilds and retrains automatically). Alert thresholds to be reviewed with the full table.
  - *Promote (2026-11-03):* `parknet:3` (aligned) 0.9616 vs `parknet:1` 0.9661 on the aligned 23/10 holdout → **rejected, `parknet:1` stays champion** (≥ rule). The difference (~5 crops) is noise, and 23/10 has almost no drift: the holdout does not measure robustness to a moved camera. **Decided by the user: wait for a new day that no model has seen** → **holdout = 23/10 + 3/11**; 3/11 is reserved before labelling (`train.py` warns and continues if a holdout day has no labels yet). Adding to the holdout a day already seen by the champion would bias it in its favour; if that is ever needed, a rule is foreseen in `promote`: a champion trained with holdout days = not comparable (uses `trained_days`).
  - *Camera fall (3/11, glued back):* the alignment absorbs it (preview 3/11 16:00: (−14, −19) px, +2.9°, 54 inliers; p04/p05 fall over their spot again). **Decided by the user: also move the Label Studio polygons every night** — `batch_inference` stores the correction of each photo and `push_to_label_studio` places each polygon over its real spot (`alignment.transform_points`, always from `rois.json`, without accumulating), in the pending tasks the user reviews in the morning. Annotated tasks are not touched; the pipeline still reads only the labels.
  - **Decided (2026-11-02): first the complete pipeline wired end to end (promote, inference, schedules, hand-over from the Pi) and then model optimisation** (starting with option A).
- **Cross-validation:** weekly by default + manual whenever wanted. Nights without CV only train the final model, with the **number of epochs of the last cross-validation run** (to be reviewed once there is a `champion`).
- **`champion` marker:** MLflow alias if the Azure ML registry supports it; otherwise a tag on the model version. Checked at implementation time; the code that finds the champion lives in a single place.
- **Inference: a pipeline component, no Batch Endpoint or inference as a service for now** (possible future evolution). Promoting needs no redeployment: the next run loads the new `champion`.
- **Failures:** email when a job fails; maximum time per step (~1 h dataset, ~3 h training with CV, ~30 min inference); retries in the code for network calls (3 attempts with growing waits, e.g. Label Studio waking up from scale-to-zero).

Block 4 closed. Next: block 5 (what to do with the existing pieces: `training/` code, v1 in the self-hosted MLflow, Pi transition).

Decisions taken — block 5 (what to do with the existing pieces), carried out on 2026-10-27:
- **v1 code deleted:** the `training/` folder (never versioned) and the local `data/`. The new code lives in `ml/`.
- **Self-hosted MLflow deleted entirely**, v1 included (it was only a training example, no history is kept): Container App `aml-container-app`, environment `mlflow-env`, database `mlflow` and Blob container `mlflow` in `amlsblob`. **Kept:** the PostgreSQL server `mlflowtest` and its `labelstudio` database, which Label Studio depends on.
- **Nightly Pi import kept active** until the full pipeline is in place (with the first `champion` its timer is disabled, see block 4).

Block 5 closed. **Pipeline redesign defined.** Next: implement it (Bicep infra, environments, components, pipelines and schedules) and close the open points: holdout days, `champion` alias in the Azure ML registry, and epochs once there is a `champion`.

Decisions taken — block 6 (self-hosted MLflow for experiment tracking), 2026-11-05:
- **Why:** the workspace's MLflow UI is poor for comparing trainings (one run per pipeline step, no side-by-side curves across nights). A self-hosted MLflow 3.x gives the full comparison UI, as in `tinyConvNeXt`.
- **Split of responsibilities:**
  - **Self-hosted MLflow = experiment tracking of `train` + `promote`**: **one run per training** in the experiment `parkwatch-training`, named `<UTC date>_w<width>_<input_size>px`. `train` creates it (params, per-epoch curves, holdout metrics, `eval/` figures, `dataset/`, `model/`); `promote` completes the same run (params prefixed `promote/`, metrics `new/*` and `champion/*`, tags `parknet_version` and `promotion`). Tags `azureml_train_run` / `azureml_promote_run` point to the Azure ML jobs, and each Azure ML job run gets a `self_hosted_run` tag linking back.
  - **Azure ML stays for everything else:** the `parknet` model registry and the `champion` tag (`ml/registry.py`, every call inside `tracking.azureml()`; the model is uploaded to the Azure ML job run and registered from there), pipeline control outputs, `align_photos`/`drift_alert`/inference runs.
  - **Never fails because of tracking:** if the server cannot be reached (5 min of `/health` polling for the cold start), `train` logs to its Azure ML job run as before and `promote` follows it (`model_config.json` records where the run went).
- **Server** (`infra/mlflow.bicep`, `mlflow_server/`): Container App `parkwatch-mlflow` (environment `parkwatch-apps`, RG `parkwatch`), MLflow **3.18.0** with **basic-auth** (default permission READ), 1 CPU / 2 GiB, **scale to zero** (cold start ~30 s; `minReplicas: 1` if it becomes annoying). Tracking and auth databases `parkwatch_mlflow` / `parkwatch_mlflow_auth` (role `mlflow`) on the existing PostgreSQL server `mlflowtest` (B1ms, shared with Label Studio; not the bottleneck). Artifacts proxied by the server (`--serve-artifacts`) to the `mlflow` container of `parkwatchstorage`, written with its own identity `parkwatch-mlflow-identity`. Secrets as Key Vault references. Speed: `auth_cache_ttl_seconds = 300` (PBKDF2 on every request otherwise), 2 workers, schema migration only on demand (`mlflowDbUpgrade`).
- **Pipeline credentials:** user `pipeline` with EDIT on `parkwatch-training` only (`mlflow_server/create_pipeline_user.py`), password in the Key Vault (`mlflow-pipeline-password`), read by the jobs with the cluster identity. Environment `parkwatch-train:2` (+ `azure-identity`, `azure-keyvault-secrets`); clients stay on mlflow 2.22 + azureml-mlflow 1.60, compatible with the 3.18 server.
- **History:** the 7 finished trainings of 30/10–5/11 copied from Azure ML with `tools/import_mlflow_history.py` (train + promote paired through the `train_run_id` tag of each `parknet` version; full metric history, `eval/` and `dataset/` artifacts; weights stay in the registry). 30/10 and 31/10 have no promote data: they predate the registry.
- *Verified (2026-11-05):* smoke job `aml/jobs/smoke-tracking.yml` (Key Vault password, self-hosted run with curve and artifact, champion found and downloaded inside `azureml()`, back to the self-hosted server afterwards); then a manual pipeline run: run `2026-11-05_w1.0_96px` with 26 epochs taken from the last CV run (an imported one), holdout macro F1 0.9687 → `parknet:6` registered and **rejected** against `parknet:5` (0.9731). The schedules are recreated with `aml/pipelines/schedules.py` so that the nightly runs pick it up.

### Phase 4 — Daily retraining pipeline (recurring, steady state)
Defined in block 4 of the redesign (Phase 3). Summary of the loop:
1. **Inference pipeline** (04:00): the `champion` predicts the new photos and creates the Label Studio tasks with those predictions
2. **Label Studio**: manual review/correction of the predictions (active learning / human-in-the-loop)
3. The corrections change the dataset fingerprint → `build_dataset` publishes a new version of the data asset
4. **Training pipeline** (02:00, only if the dataset changed): training from scratch on the accumulated dataset (weekly or manual CV; other nights only the final model)
5. **Evaluation**: against the fixed holdout, macro F1 versus the current `champion`
6. **Promotion**: `champion` moves to the new model if it matches or beats the macro F1

### Phase 5 — Live inference (future, not v1)
- Only enable once the model is accurate enough in the Phase 4 process
- Event-driven architecture: an Azure Function with a Blob trigger runs inference on each new photo and writes the latest state to Table Storage
- Lightweight, always-available public query API (HTTP Function, consumption plan) that only reads the cached state — it does not run the model per request
- Portfolio alternative/comparison: Managed Online Endpoint with scale-to-zero, to also demonstrate that pattern

---

## Architecture notes / discarded decisions

- **ESP32 → Blob directly:** discarded because of TLS/memory fragility on the ESP32; going through the Pi is preferred
- **Always-on Managed Online Endpoint as the main piece:** discarded for v1 — hourly cost without real traffic, does not fit a 3-min cadence nor the decision not to serve live inference yet
- **Batch Endpoint as the only "live" inference piece:** initially discarded because of compute start-up overhead per individual event; but brought back as the right piece for the daily batch process (not per single event)
- **Batch Endpoint for the daily process:** replaced (2026-10-27, block 4) by an inference component inside an Azure ML pipeline that loads the `champion` from the registry: there is no deployment to update when promoting. It remains a future evolution if inference is to be exposed as a service.
- **Self-hosted MLflow for ParkWatch:** replaced (block 3) by the workspace's integrated MLflow; the self-hosted one was deleted entirely, v1 included, on 2026-10-27 (block 5). **Brought back on 2026-11-05 (block 6) for experiment tracking only**, new server and databases; the registry stays in Azure ML.
