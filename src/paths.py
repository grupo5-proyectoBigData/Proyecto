## Define rutas para el proyecto y carga de configuración

from pathlib import Path
import yaml


def is_colab() -> bool:
    try:
        import google.colab  
        return True
    except ImportError:
        return False


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_config(start_dir: Path) -> Path:
    for candidate in [start_dir, *start_dir.parents]:
        possible = candidate / "config" / "config.yaml"
        if possible.exists():
            return possible

    raise FileNotFoundError(
        "No se encontró config/config.yaml. "
        "Abre el notebook desde proyecto/notebooks/."
    )


def get_project_paths(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = find_config(Path.cwd())

    cfg = load_config(config_path)
    project_root = config_path.parent.parent

    paths = {
        "project_root": project_root,
        "data_raw": project_root / cfg["paths"]["data_raw"],
        "data_processed": project_root / cfg["paths"]["data_processed"],
        "models": project_root / cfg["paths"]["models"],
        "figures": project_root / cfg["paths"]["figures"],
    }

    for folder in [
        paths["data_raw"],
        paths["data_processed"],
        paths["models"],
        paths["figures"],
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    paths["config"] = cfg
    return paths


def get_dataset_root(paths: dict) -> Path:
    cfg = paths["config"]

    if is_colab():
        kaggle_root = Path(cfg["colab"]["kaggle_dataset_dir"])

        if kaggle_root.exists():
            return kaggle_root

        candidates = sorted(Path("/kaggle/input").glob("*"))
        if candidates:
            print("Carpetas disponibles en /kaggle/input:")
            for candidate in candidates:
                print(" -", candidate)

        raise FileNotFoundError(
            "No se encontró el dataset privado de Kaggle. "
            "Añádelo desde el panel 'Insertar datos' de Colab."
        )

    return paths["data_raw"]


def get_training_settings(cfg: dict) -> dict:
    if is_colab():
        return {
            "image_size": cfg["data"]["image_size_colab"],
            "batch_size": cfg["data"]["batch_size_colab"],
        }

    return {
        "image_size": cfg["data"]["image_size_local"],
        "batch_size": cfg["data"]["batch_size_local"],
    }