from tensorflow import keras
from keras.applications.resnet50 import preprocess_input

MODEL_PATH = "en_iyi_resnet50_modeli.keras"
SUMMARY_OUT = "model_summary.txt"
CONFIG_OUT = "model_config.json"


def main() -> None:
    model = keras.models.load_model(
        MODEL_PATH,
        safe_mode=False,
        custom_objects={"preprocess_input": preprocess_input},
    )

    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        f.write(model.to_json())

    print(f"Wrote {SUMMARY_OUT} and {CONFIG_OUT}")


if __name__ == "__main__":
    main()
