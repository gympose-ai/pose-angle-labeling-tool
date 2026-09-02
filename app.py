import os
os.environ["GRADIO_DISABLE_BROTLI"] = "1"

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import gradio as gr
import torch
from PIL import Image

from easy_ViTPose.inference import VitInference
from easy_ViTPose.vit_utils.inference import NumpyEncoder
from easy_ViTPose.vit_utils.visualization import draw_points_and_skeleton, joints_dict
from pose_editor import prepare_editor_from_path, apply_and_save_keypoints, create_editor_component

# --- paths ---
BASE_DIR = Path(__file__).parent
TEMP_INPUT = BASE_DIR / "temp" / "inputs"
TEMP_VIDEO_INPUT = BASE_DIR / "temp" / "videos"
TEMP_OUTPUT = BASE_DIR / "temp" / "outputs"
TEMP_INPUT.mkdir(parents=True, exist_ok=True)
TEMP_VIDEO_INPUT.mkdir(parents=True, exist_ok=True)
TEMP_OUTPUT.mkdir(parents=True, exist_ok=True)

POSE_MODEL = BASE_DIR / "checkpoints" / "vitpose-h-coco_25.pth"
YOLO_MODEL = BASE_DIR / "checkpoints" / "yolo11x.pt"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}
_MODEL_CACHE: Dict[str, Any] = {}


def _get_runtime_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model() -> VitInference:
    device = _get_runtime_device()
    cache_key = f"{POSE_MODEL}|{YOLO_MODEL}|{device}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = VitInference(
            str(POSE_MODEL),
            str(YOLO_MODEL),
            model_name="h",
            det_class="human",
            dataset="coco_25",
            yolo_size=320,
            device=device,
            is_video=False,
            single_pose=True,
        )
    return _MODEL_CACHE[cache_key]


def _save_outputs(
    model: VitInference,
    input_path: Path,
    frame_keypoints: Dict[Any, Any],
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    run_dir = output_dir or TEMP_OUTPUT / f"{input_path.stem}_{uuid.uuid4().hex[:10]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result_image = run_dir / f"{input_path.stem}_result.png"
    result_json = run_dir / f"{input_path.stem}_result.json"

    result_rgb = model.draw(show_yolo=False, show_raw_yolo=False, confidence_threshold=0.5)
    Image.fromarray(result_rgb).save(result_image)

    out_json = {
        "keypoints": [frame_keypoints],
        "skeleton": joints_dict()[model.dataset]["keypoints"],
    }
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, cls=NumpyEncoder)

    return result_image, result_json


def _run_inference_for_input(
    input_path: Path,
    output_dir: Optional[Path] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not input_path.exists():
        return None, None, f"Input not found: {input_path}"

    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        return None, None, f"Image read failed: {input_path}"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        model = _get_model()
        frame_keypoints = model.inference(img_rgb)
        result_image, result_json = _save_outputs(model, input_path, frame_keypoints, output_dir)
    except Exception as e:
        return None, None, f"Inference error for {input_path.name}: {e}"

    return str(result_image), str(result_json), None


def _collect_images_from_directory(folder_path: Path) -> List[Path]:
    return sorted(
        [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def _uploaded_file_path(uploaded_file: Any) -> Optional[Path]:
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file)
    if isinstance(uploaded_file, tuple) and uploaded_file:
        return Path(uploaded_file[0])
    file_name = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "path", None)
    return Path(file_name) if file_name else None


def _safe_folder_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:90] or "video"


def handle_video_upload(video_file: Any) -> Tuple[str, str]:
    video_path = _uploaded_file_path(video_file)
    if video_path is None:
        return "", "Video secilmedi."

    if not video_path.exists():
        return "", f"Video dosyasi bulunamadi: {video_path}"

    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        return "", f"Desteklenmeyen video formati: {video_path.suffix}. Desteklenenler: {supported}"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return str(video_path), f"Video yuklendi ama OpenCV ile okunamadi: {video_path}"

    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0
    capture.release()

    size_mb = video_path.stat().st_size / (1024 * 1024)
    status = [
        f"Video secildi: {video_path.name}",
        f"Kaynak video yolu: {video_path}",
        "Video temp/videos altina kopyalanmayacak; sadece pose cikarimli resimler kaydedilecek.",
        f"Boyut: {size_mb:.2f} MB",
        f"Cozunurluk: {width}x{height}",
        f"FPS: {fps:.2f}",
        f"Frame sayisi: {frame_count}",
        f"Sure: {duration:.2f} sn",
    ]
    return str(video_path), "\n".join(status)


def process_video_frames(
    video_path_text: str,
    requested_fps: float,
    progress=gr.Progress(),
):
    video_path_text = (video_path_text or "").strip()
    if not video_path_text:
        return (
            "",
            [],
            "Once bir video yukleyin.",
            [],
            0,
            gr.update(value=1, minimum=1, maximum=1, visible=False),
            "",
            "",
            "",
        )

    video_path = Path(video_path_text)
    if not video_path.exists():
        return (
            "",
            [],
            f"Video dosyasi bulunamadi: {video_path}",
            [],
            0,
            gr.update(value=1, minimum=1, maximum=1, visible=False),
            "",
            "",
            "",
        )

    try:
        target_fps = float(requested_fps)
    except (TypeError, ValueError):
        target_fps = 1.0
    target_fps = max(0.1, target_fps)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return (
            "",
            [],
            f"Video OpenCV ile acilamadi: {video_path}",
            [],
            0,
            gr.update(value=1, minimum=1, maximum=1, visible=False),
            "",
            "",
            "",
        )

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0
    if source_fps <= 0:
        source_fps = target_fps
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / source_fps if source_fps > 0 else 0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = TEMP_VIDEO_INPUT / f"{timestamp}_{_safe_folder_name(video_path.stem)}"
    if run_dir.exists():
        run_dir = TEMP_VIDEO_INPUT / f"{timestamp}_{_safe_folder_name(video_path.stem)}_{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_frame_dir = run_dir / "_raw_frames"

    gallery_items = []
    batch_items: List[Dict[str, str]] = []
    errors = []
    selected_count = 0
    frame_idx = 0
    next_sample_time = 0.0
    sample_interval = 1.0 / target_fps
    runtime_device = _get_runtime_device()
    t0 = time.perf_counter()

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break

        time_sec = frame_idx / source_fps
        if time_sec + 1e-9 >= next_sample_time:
            frame_path = raw_frame_dir / f"frame_{frame_idx:06d}_{time_sec:.3f}s.png"
            raw_frame_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_path), frame_bgr)

            progress(
                selected_count / max(1, int(duration * target_fps) or 1),
                desc=f"Frame isleniyor: {frame_idx}",
            )
            out_img, json_path, err = _run_inference_for_input(frame_path, run_dir)
            if err:
                errors.append(f"Frame {frame_idx}: {err}")
            elif out_img and json_path:
                caption = f"frame {frame_idx} | {time_sec:.2f}s"
                gallery_items.append((out_img, caption))
                has_pose, pose_error = _has_pose_keypoints(json_path)
                if has_pose:
                    batch_items.append({
                        "result_image": out_img,
                        "json_path": json_path,
                        "source_name": caption,
                        "original_image": str(frame_path),
                    })
                else:
                    errors.append(f"{caption}: {pose_error}")

            selected_count += 1
            next_sample_time += sample_interval

        frame_idx += 1

    capture.release()

    if not gallery_items:
        error_text = "\n".join(errors) if errors else "Videodan islenebilir frame uretilemedi."
        return (
            "",
            [],
            error_text,
            [],
            0,
            gr.update(value=1, minimum=1, maximum=1, visible=False),
            "",
            "",
            "",
        )

    elapsed = time.perf_counter() - t0
    first_item = batch_items[0] if batch_items else None
    first_payload, _ = prepare_editor_from_path(first_item["original_image"], first_item["json_path"]) if first_item else ("", "")
    nav_status = _batch_nav_state_text(0, len(batch_items), first_item["source_name"]) if first_item else "Pose tespit edilen frame yok."
    status_lines = [
        f"Video islendi: {video_path.name}",
        f"Kaynak FPS: {source_fps:.2f}",
        f"Istenen FPS: {target_fps:.2f}",
        f"Toplam frame: {total_frames}",
        f"Secilen frame: {selected_count}",
        f"Basarili frame: {len(gallery_items)}",
        f"Hatali frame: {len(errors)}",
        f"Device: {runtime_device}",
        f"Cikti klasoru: {run_dir}",
        f"Pose cikarimli resimler ve JSON dosyalari bu klasore kaydedildi.",
        f"Toplam sure: {elapsed:.2f}s",
    ]
    if errors:
        status_lines.append("\nHatalar:")
        status_lines.extend(errors)

    return (
        first_payload,
        gallery_items,
        "\n".join(status_lines),
        batch_items,
        0,
        gr.update(value=1, minimum=1, maximum=max(1, len(batch_items)), visible=bool(batch_items)),
        nav_status,
        first_item["original_image"] if first_item else "",
        first_item["json_path"] if first_item else "",
    )


def _batch_nav_state_text(index: int, total: int, source_name: str) -> str:
    return f"{index + 1}/{total} - {source_name}"


def _show_batch_item(batch_items: List[Dict[str, str]], index: int):
    if not batch_items:
        return "", "", 0, gr.update(value=1, minimum=1, maximum=1, visible=False), "Batch sonucu yok.", ""

    idx = max(0, min(index, len(batch_items) - 1))
    item = batch_items[idx]
    nav_text = _batch_nav_state_text(idx, len(batch_items), item.get("source_name", ""))
    editor_payload, _ = prepare_editor_from_path(
        item.get("original_image", ""),
        item["json_path"],
    )
    return (
        editor_payload,
        item["json_path"],
        idx,
        gr.update(value=idx + 1, minimum=1, maximum=len(batch_items), visible=True),
        nav_text,
        item.get("original_image", ""),
    )


def _has_pose_keypoints(json_path: str) -> Tuple[bool, str]:
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        kp_outer = data.get("keypoints", [])
        if not kp_outer or not isinstance(kp_outer, list):
            return False, "JSON'da keypoints yok"
        person_dict = kp_outer[0]
        if not isinstance(person_dict, dict) or not person_dict:
            return False, "pose/person tespit edilemedi"
        kp_list = person_dict.get("0")
        if kp_list is None:
            kp_list = person_dict[next(iter(person_dict.keys()))]
        if not isinstance(kp_list, list) or len(kp_list) != 25:
            count = len(kp_list) if isinstance(kp_list, list) else "gecersiz"
            return False, f"beklenen 25 keypoint, gelen: {count}"
        return True, ""
    except Exception as e:
        return False, f"JSON okunamadi: {e}"


def show_prev_batch(batch_items: List[Dict[str, str]], current_idx: int):
    return _show_batch_item(batch_items, current_idx - 1)


def show_next_batch(batch_items: List[Dict[str, str]], current_idx: int):
    return _show_batch_item(batch_items, current_idx + 1)


def show_batch_by_slider(batch_items: List[Dict[str, str]], slider_idx: float):
    target_idx = int(slider_idx) - 1
    return _show_batch_item(batch_items, target_idx)


def run_vitpose(image: Image.Image, folder_path: str):
    folder_path = (folder_path or "").strip()
    runtime_device = _get_runtime_device()
    t0 = time.perf_counter()

    if image is not None:
        filename = f"{uuid.uuid4()}.png"
        input_path = TEMP_INPUT / filename
        image.save(input_path)

        out_img, json_path, err = _run_inference_for_input(input_path)
        if err:
            return "", [], err, [], 0, gr.update(value=1, minimum=1, maximum=1, visible=False), gr.update(), gr.update(), "", ""

        has_pose, pose_error = _has_pose_keypoints(json_path)
        if not has_pose:
            elapsed = time.perf_counter() - t0
            return (
                "",
                [(out_img, input_path.name)] if out_img else [],
                f"{json_path}\nPose tespit edilemedi: {pose_error}\nDevice: {runtime_device} | Time: {elapsed:.2f}s",
                [],
                0,
                gr.update(value=1, minimum=1, maximum=1, visible=False),
                gr.update(),
                gr.update(),
                "Tek goruntu modu: pose yok.",
                str(input_path),
            )

        editor_payload, _ = prepare_editor_from_path(str(input_path), json_path)
        elapsed = time.perf_counter() - t0
        return (
            editor_payload,
            [(out_img, input_path.name)],
            f"{json_path}\nDevice: {runtime_device} | Time: {elapsed:.2f}s",
            [],
            0,
            gr.update(value=1, minimum=1, maximum=1, visible=False),
            gr.update(),
            gr.update(),
            "Tek görüntü modu.",
            str(input_path),
        )

    if not folder_path:
        return None, [], "Tek bir görüntü yükleyin veya görüntü klasörü yolu girin.", [], 0, gr.update(value=1, minimum=1, maximum=1, visible=False), gr.update(), gr.update(), "", ""

    dir_path = Path(folder_path)
    if not dir_path.exists() or not dir_path.is_dir():
        return None, [], f"Klasör bulunamadı veya geçersiz: {folder_path}", [], 0, gr.update(value=1, minimum=1, maximum=1, visible=False), gr.update(), gr.update(), "", ""

    image_files = _collect_images_from_directory(dir_path)
    if not image_files:
        return None, [], f"Klasörde desteklenen görüntü bulunamadı: {folder_path}", [], 0, gr.update(value=1, minimum=1, maximum=1, visible=False), gr.update(), gr.update(), "", ""

    gallery_items = []
    batch_items: List[Dict[str, str]] = []
    json_paths = []
    first_result = None
    errors = []

    for img_path in image_files:
        out_img, json_path, err = _run_inference_for_input(img_path)
        if err:
            errors.append(f"{img_path.name}: {err}")
            continue

        if out_img is not None:
            if first_result is None:
                first_result = out_img
            gallery_items.append((out_img, img_path.name))
            if json_path:
                has_pose, pose_error = _has_pose_keypoints(json_path)
                if has_pose:
                    batch_items.append({
                        "result_image": out_img,
                        "json_path": json_path,
                        "source_name": img_path.name,
                        "original_image": str(img_path),
                    })
                else:
                    errors.append(f"{img_path.name}: {pose_error}")

        if json_path:
            json_paths.append(json_path)

    if not gallery_items:
        error_text = "\n".join(errors) if errors else "Klasördeki görüntüler işlenemedi."
        return None, [], error_text, [], 0, gr.update(value=1, minimum=1, maximum=1, visible=False), gr.update(visible=False), gr.update(visible=False), "", ""

    status_lines = [
        f"Device: {runtime_device}",
        f"Toplam görüntü: {len(image_files)}",
        f"Başarılı: {len(gallery_items)}",
        f"Hatalı: {len(errors)}",
    ]
    elapsed = time.perf_counter() - t0
    status_lines.append(f"Toplam süre: {elapsed:.2f}s")
    if len(gallery_items) > 0:
        status_lines.append(f"Ortalama süre/görüntü: {elapsed / len(gallery_items):.2f}s")
    if json_paths:
        status_lines.append("\nJSON çıktıları:")
        status_lines.extend(json_paths)
    if errors:
        status_lines.append("\nHatalar:")
        status_lines.extend(errors)

    nav_status = _batch_nav_state_text(0, len(batch_items), batch_items[0]["source_name"]) if batch_items else "Pose tespit edilen goruntu yok."
    first_original = batch_items[0]["original_image"] if batch_items else ""
    first_json     = batch_items[0]["json_path"]      if batch_items else ""
    first_payload, _ = prepare_editor_from_path(first_original, first_json) if first_original else ("", "")
    return (
        first_payload,
        gallery_items,
        "\n".join(status_lines),
        batch_items,
        0,
        gr.update(value=1, minimum=1, maximum=max(1, len(batch_items)), visible=bool(batch_items)),
        gr.update(),
        gr.update(),
        nav_status,
        first_original,
    )


def parse_pose_json(data: Dict[str, Any]) -> Tuple[Dict[int, str], List[Tuple[float, float, float]]]:
    """
    Senin format:
      data["skeleton"] : {"0":"nose", ...}
      data["keypoints"] : [{"0": [[x,y,c], ... 25]}]
    Çıktı:
      idx_to_name, kps (len=25) as (x,y,c)
    """
    idx_to_name = {int(k): v for k, v in data.get("skeleton", {}).items()}

    # keypoints -> first element -> person "0"
    kp_outer = data.get("keypoints", [])
    if not kp_outer or not isinstance(kp_outer, list):
        raise ValueError("JSON'da 'keypoints' bulunamadı veya format hatalı.")

    person_dict = kp_outer[0]
    if not isinstance(person_dict, dict) or not person_dict:
        raise ValueError("JSON'da tespit edilmis kisi/keypoint yok.")
    if "0" not in person_dict:
        # fallback: ilk key'i al
        first_key = next(iter(person_dict.keys()))
        kp_list = person_dict[first_key]
    else:
        kp_list = person_dict["0"]

    if len(kp_list) != 25:
        raise ValueError(f"Beklenen 25 keypoint, gelen: {len(kp_list)}")

    kps = [(float(x), float(y), float(c)) for x, y, c in kp_list]
    return idx_to_name, kps


def draw_from_json(image: Image.Image, json_file) -> Tuple[Image.Image, str]:
    if image is None:
        return None, "Önce bir görüntü çalıştır veya yükle."

    if json_file is None:
        return None, "Bir JSON dosyası yükle."

    # gr.File -> obj has .name
    json_path = Path(json_file.name)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    try:
        idx_to_name, kps = parse_pose_json(data)
    except Exception as e:
        return None, f"JSON parse error: {e}"

    thr = 0.30  # confidence threshold
    skeleton = joints_dict()["coco_25"]["skeleton"]

    # draw_points_and_skeleton (y, x, c) bekliyor
    kps_arr = np.array([[float(y), float(x), float(c)] for x, y, c in kps], dtype=np.float32)

    img_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_np = draw_points_and_skeleton(
        img_np,
        kps_arr,
        skeleton,
        person_index=0,
        points_color_palette="gist_rainbow",
        skeleton_color_palette="jet",
        points_palette_samples=10,
        confidence_threshold=thr,
    )
    out_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

    return out_img, f"Loaded JSON: {json_path.name} | thr={thr}"


_CSS = """
.pe-hidden-trigger { display: none !important; }
"""

with gr.Blocks() as demo:
    gr.Markdown("## easy_ViTPose - Inference + JSON Overlay")

    # ── Top row: input + interactive pose canvas ─────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            input_img    = gr.Image(type="pil", label="Input Image")
            input_folder = gr.Textbox(
                label="Image Folder Path (optional)",
                placeholder="C:/path/to/images",
            )
            run_button   = gr.Button("Run ViTPose")
            json_path_box = gr.Textbox(label="Inference JSON Path", lines=2)
        with gr.Column(scale=2):
            # The pose result IS the interactive editor — no separate step needed
            pose_editor_html = create_editor_component()

    # ── Apply & Save — directly below the canvas ────────────────────────────
    # Hidden textbox: JS canvas writes keypoints here; Python reads it.
    kp_editor_output = gr.Textbox(
        elem_id="kp_editor_output", elem_classes="pe-hidden-trigger", lines=1
    )
    with gr.Row():
        apply_save_btn = gr.Button("✅ Apply & Save", variant="primary")
        editor_status  = gr.Textbox(label="Editor Status", lines=1, interactive=False)

    # ── Batch gallery + navigation ───────────────────────────────────────────
    batch_gallery    = gr.Gallery(label="Batch Results (Folder Input)", columns=4, height=260)

    # Hidden Textboxes for JS to trigger prev/next via change event
    prev_trigger = gr.Textbox(elem_classes="pe-hidden-trigger", elem_id="pe_prev_trigger")
    next_trigger = gr.Textbox(elem_classes="pe-hidden-trigger", elem_id="pe_next_trigger")
    batch_slider     = gr.Slider(
        label="Frame / Image Index", minimum=1, maximum=1, value=1, step=1, visible=False
    )
    batch_nav_status = gr.Textbox(label="Batch Navigation", lines=1, interactive=False)

    # Video upload block - placed before JSON overlay tools.
    gr.Markdown("### Video Yukleme")
    with gr.Column():
        input_video = gr.Video(
            label="Video Yukle",
            sources=["upload"],
            interactive=True,
            height=260,
        )
        video_fps = gr.Dropdown(
            choices=[1, 5, 10],
            value=1,
            label="Frame Cikarma Hizi (kare/sn)",
        )
        process_video_btn = gr.Button("Videodan Frame Cikar ve ViTPose Calistir", variant="primary")
        video_path_box = gr.Textbox(label="Uploaded Video Path", lines=2, interactive=False)
        video_status = gr.Textbox(label="Video Status", lines=7, interactive=False)
        video_gallery = gr.Gallery(label="Video Frame Overlay Results", columns=4, height=280)

    # State
    batch_items_state  = gr.State([])
    batch_index_state  = gr.State(0)
    original_img_state = gr.State("")

    input_video.upload(
        fn=handle_video_upload,
        inputs=[input_video],
        outputs=[video_path_box, video_status],
    )

    process_video_btn.click(
        fn=process_video_frames,
        inputs=[video_path_box, video_fps],
        outputs=[
            pose_editor_html,
            video_gallery,
            video_status,
            batch_items_state,
            batch_index_state,
            batch_slider,
            batch_nav_status,
            original_img_state,
            json_path_box,
        ],
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    _NAV_OUTPUTS = [
        pose_editor_html,
        json_path_box,
        batch_index_state,
        batch_slider,
        batch_nav_status,
        original_img_state,
    ]

    run_button.click(
        fn=run_vitpose,
        inputs=[input_img, input_folder],
        outputs=[
            pose_editor_html,
            batch_gallery,
            json_path_box,
            batch_items_state,
            batch_index_state,
            batch_slider,
            prev_trigger,
            next_trigger,
            batch_nav_status,
            original_img_state,
        ],
    )

    prev_trigger.change(
        fn=show_prev_batch,
        inputs=[batch_items_state, batch_index_state],
        outputs=_NAV_OUTPUTS,
    )

    next_trigger.change(
        fn=show_next_batch,
        inputs=[batch_items_state, batch_index_state],
        outputs=_NAV_OUTPUTS,
    )

    batch_slider.change(
        fn=show_batch_by_slider,
        inputs=[batch_items_state, batch_slider],
        outputs=_NAV_OUTPUTS,
    )

    # Read current keypoints directly from canvas on button click.
    _APPLY_JS = """(orig, kps, jpath, payload) => {
        var w = document.querySelector('.pe-wrap');
        if (w && typeof w._peGetKps === 'function') {
            kps = w._peGetKps();
        }
        return [orig, kps, jpath, payload];
    }"""

    apply_save_btn.click(
        fn=apply_and_save_keypoints,
        inputs=[original_img_state, kp_editor_output, json_path_box, pose_editor_html],
        outputs=[pose_editor_html, editor_status],
        js=_APPLY_JS,
    )

    # ── JSON Upload overlay (standalone, has its own image input) ────────────
    gr.Markdown("### JSON Upload → Draw keypoints + skeleton on the image")

    with gr.Row():
        overlay_src_img = gr.Image(type="pil", label="Source Image")
        json_file       = gr.File(label="Upload Pose JSON (.json)", file_types=[".json"])
        overlay_img     = gr.Image(type="pil", label="Overlay Result")
    overlay_status = gr.Textbox(label="Status", lines=2)

    draw_button = gr.Button("Draw From Uploaded JSON")
    draw_button.click(
        fn=draw_from_json,
        inputs=[overlay_src_img, json_file],
        outputs=[overlay_img, overlay_status],
    )

if __name__ == "__main__":
    demo.launch(css=_CSS)
