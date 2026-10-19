# Guides: Fine-Tuning Your Own Model

> In this page: How to fine-tune a visual deepfake detector on custom datasets, folder structures, transfer learning routines, checkpoint saving, and calibration.  
> Who it's for: ML engineers customizing Guardian AI for specialized domains or novel generators.

---

## Fine-Tuning Workflow

```mermaid
graph TD
    Data[Prepare Dataset: real/ and fake/] --> Generators[Configure ImageDataGenerator]
    Generators --> Phase1[Phase 1: Train Head on Frozen Base (5 Epochs)]
    Phase1 --> Phase2[Phase 2: Unfreeze Deep Layers & Fine-Tune (15 Epochs, Low LR)]
    Phase2 --> Save[Save Checkpoint (.h5 / .keras)]
    Save --> Calibrate[Run Offline Calibration Sweep]
    Calibrate --> Deploy[Update model_runtime_config.json & Restart]
```

---

## 1. Directory Structure

Organize your labeled images into standard subdirectories:

```text
my_dataset/
├── train/
│   ├── real/    # Authentic portrait images
│   └── fake/    # Manipulated / synthetic images
├── valid/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

---

## 2. Training Script Implementation

Create a script `train_custom.py` following the two-phase transfer learning pattern established in the repository:

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255.)

train_flow = train_datagen.flow_from_directory(
    'my_dataset/train/', target_size=(224, 224), batch_size=32, class_mode='binary'
)
val_flow = val_datagen.flow_from_directory(
    'my_dataset/valid/', target_size=(224, 224), batch_size=32, class_mode='binary'
)

# 2. Build Base Model
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze for Phase 1

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 3. Phase 1: Train Custom Head
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_flow, epochs=5, validation_data=val_flow)

# 4. Phase 2: Unfreeze Deep Layers for Fine-Tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False  # Keep early feature layers frozen

model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint("models/my_finetuned_model.h5", save_best_only=True)
]
model.fit(train_flow, epochs=15, validation_data=val_flow, callbacks=callbacks)
```

---

## 3. Calibrating & Registering Your Model

Once training completes and `models/my_finetuned_model.h5` is saved:

### Step 1: Generate Validation Scores
```bash
python scripts/generate_scores_local.py \
  --real-dir my_dataset/valid/real \
  --fake-dir my_dataset/valid/fake \
  --out my_calibration.csv \
  --config model_runtime_config.json \
  --batch-size 64
```

### Step 2: Calibrate Decision Threshold
```bash
python scripts/calibrate_threshold.py \
  --csv my_calibration.csv \
  --out model_runtime_config.json
```

### Step 3: Update Runtime Configuration
Edit `model_runtime_config.json`:
```json
{
  "selected_model": "models/my_finetuned_model.h5",
  "preprocess_mode": "mobilenet_v2",
  "threshold": 0.35,
  "positive_label": "REAL",
  "negative_label": "FAKE",
  "input_size": [224, 224]
}
```

### Step 4: Restart Server & Validate
Restart `app.py` and query `GET /model-status` to verify that `models/my_finetuned_model.h5` is loaded with `preprocess_mode: mobilenet_v2`.
