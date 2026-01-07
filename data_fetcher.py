import os
import pandas as pd
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    CRS,
    BBox
)
from PIL import Image
from tqdm import tqdm

# -----------------------
# CONFIGURATION
# -----------------------
CLIENT_ID = "c6dc59f9-2274-4803-b156-c636ebd560ac"
CLIENT_SECRET = "2JKKw4SjvO19u3CyjirMKv6zTPUwdK37"

config = SHConfig()
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(IMG_DIR, exist_ok=True)

# -----------------------
# LOAD DATA (LIMITED)
# -----------------------
df = pd.read_excel(os.path.join(DATA_DIR, "train(1).xlsx"))

# ONLY FIRST 400 PROPERTIES (FAST + SAFE)
df = df.head(400)

# -----------------------
# SENTINEL REQUEST
# -----------------------
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3 }
  };
}

function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02];
}
"""

# -----------------------
# DOWNLOAD LOOP
# -----------------------
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        lat, lon = row["lat"], row["long"]

        bbox = BBox(
            bbox=[lon - 0.002, lat - 0.002, lon + 0.002, lat + 0.002],
            crs=CRS.WGS84
        )

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=("2020-01-01", "2020-12-31"),
                    maxcc=0.2
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.PNG)
            ],
            bbox=bbox,
            size=(224, 224),
            config=config
        )

        img = request.get_data()[0]
        Image.fromarray(img).save(os.path.join(IMG_DIR, f"{idx}.png"))

    except:
        continue

print("Image download complete.")
