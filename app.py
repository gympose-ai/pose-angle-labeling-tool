import os
os.environ["GRADIO_DISABLE_BROTLI"] = "1"

import json
import time
import uuid
import tempfile
import hashlib
import queue
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import gradio as gr
import torch
from PIL import Image
from ultralytics import YOLO

from easy_ViTPose.inference import VitInference
from easy_ViTPose.vit_utils.inference import NumpyEncoder
from easy_ViTPose.vit_utils.visualization import draw_points_and_skeleton, joints_dict
from pose_editor import prepare_editor_from_path, apply_and_save_keypoints, create_editor_component
from athlete_tracking import AthleteTracker

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
_DETECTOR_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.RLock()
def _get_runtime_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_detector(device=None):
    """Load person detection independently of the pose model and reuse it."""
    device = device or _get_runtime_device()
    cache_key = f"{YOLO_MODEL}|{device}"
    with _MODEL_LOCK:
        if cache_key not in _DETECTOR_CACHE:
            if not YOLO_MODEL.is_file():
                raise FileNotFoundError(f"YOLO modeli bulunamadı: {YOLO_MODEL}")
            _DETECTOR_CACHE[cache_key] = YOLO(str(YOLO_MODEL), task="detect")
        return _DETECTOR_CACHE[cache_key]


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
            yolo_detector=_get_detector(device),
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
    bbox=None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not input_path.exists():
        return None, None, f"Input not found: {input_path}"

    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        return None, None, f"Image read failed: {input_path}"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        with _MODEL_LOCK:
            model = _get_model()
            frame_keypoints = model.inference(img_rgb, bbox=bbox) if bbox is not None else model.inference(img_rgb)
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


def _build_pose_video_from_frames(
    frame_paths: List[str],
    output_base_path: Path,
    fps: float,
) -> Tuple[Optional[str], Optional[str]]:
    if not frame_paths:
        return None, "Pose video icin islenmis frame yok."

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        return None, f"Pose video frame okunamadi: {frame_paths[0]}"

    height, width = first_frame.shape[:2]
    safe_fps = max(0.1, float(fps or 1.0))
    output_base_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        (output_base_path.with_suffix(".webm"), "VP80"),
        (output_base_path.with_suffix(".mp4"), "mp4v"),
    ]
    errors = []
    for output_path, fourcc_name in candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            safe_fps,
            (width, height),
        )
        if not writer.isOpened():
            errors.append(f"{output_path.suffix}/{fourcc_name} yazici acilamadi")
            continue

        written = 0
        try:
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                written += 1
        finally:
            writer.release()

        if written > 0 and output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path), None
        errors.append(f"{output_path.name} bos olustu")

    return None, "Pose video yazilamadi: " + "; ".join(errors)



def process_video_frames(
    video_path_text: str,
    requested_fps: float,
    progress=gr.Progress(),
    tracking=None,
):
    video_path_text = (video_path_text or "").strip()
    if not video_path_text:
        return (
            "",
            None,
            None,
            "Once bir video yukleyin.",
            [],
            0,
            gr.update(value=1, minimum=1, maximum=2, visible=False),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            0,
            [],
            0,
            "",
        )

    video_path = Path(video_path_text)
    if not video_path.exists():
        return (
            "",
            None,
            None,
            f"Video dosyasi bulunamadi: {video_path}",
            [],
            0,
            gr.update(value=1, minimum=1, maximum=2, visible=False),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            0,
            [],
            0,
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
            str(video_path),
            None,
            f"Video OpenCV ile acilamadi: {video_path}",
            [],
            0,
            gr.update(value=1, minimum=1, maximum=2, visible=False),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            0,
            [],
            0,
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
    video_editor_items: List[Dict[str, str]] = []
    pose_video_frame_paths: List[str] = []
    errors = []
    selected_count = 0
    frame_idx = 0
    next_sample_time = 0.0
    sample_interval = 1.0 / target_fps
    runtime_device = _get_runtime_device()
    t0 = time.perf_counter()
    tracker = None
    tracking_lost_count = 0
    tracking_records = []

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break

        tracked_box = None
        if tracking is not None:
            if tracker is None:
                try:
                    tracker = AthleteTracker(frame_bgr, tracking["bbox"])
                except (ValueError, KeyError, TypeError):
                    capture.release()
                    raise ValueError("Takip başlatılamadı. İlk karede atleti yeniden seçin.")
                tracked_box = tracker.bbox
            else:
                tracked_box = tracker.update(frame_bgr)

        time_sec = frame_idx / source_fps
        if time_sec + 1e-9 >= next_sample_time:
            frame_path = raw_frame_dir / f"frame_{frame_idx:06d}_{time_sec:.3f}s.png"
            raw_frame_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_path), frame_bgr)

            progress(
                0.9 * min(1, selected_count / max(1, int(duration * target_fps) or 1)),
                desc=f"Kareler işleniyor · {selected_count + 1}. kare · {time_sec:.1f} sn",
            )
            if tracking is not None and tracked_box is None:
                # Never fall back to a bystander after losing the selected athlete.
                out_img = str(run_dir / f"{frame_path.stem}_result.png")
                json_path = str(run_dir / f"{frame_path.stem}_result.json")
                cv2.imwrite(out_img, frame_bgr)
                Path(json_path).write_text(json.dumps({"keypoints": [{}],
                    "skeleton": joints_dict()["coco_25"]["keypoints"]}), encoding="utf-8")
                err = None
                tracking_lost_count += 1
            elif tracking is not None:
                out_img, json_path, err = _run_inference_for_input(frame_path, run_dir, bbox=tracked_box)
            else:
                out_img, json_path, err = _run_inference_for_input(frame_path, run_dir)
            if tracking is not None and json_path and not err:
                tracking_info = dict(status="lost" if tracked_box is None else "tracked",
                                     bbox=tracked_box, reason=tracker.reason, source_frame=frame_idx)
                tracking_records.append(tracking_info)
                frame_data = json.loads(Path(json_path).read_text(encoding="utf-8"))
                frame_data["tracking"] = tracking_info
                Path(json_path).write_text(json.dumps(frame_data, ensure_ascii=False), encoding="utf-8")
            if err:
                errors.append(f"Frame {frame_idx}: {err}")
            elif out_img and json_path:
                caption = f"frame {frame_idx} | {time_sec:.2f}s"
                has_pose, pose_error = _has_pose_keypoints(json_path)
                gallery_index = len(gallery_items)
                gallery_items.append((out_img, caption))
                editor_item = {
                    "gallery_index": gallery_index,
                    "result_image": out_img,
                    "json_path": json_path,
                    "source_name": caption,
                    "original_image": str(frame_path),
                }
                video_editor_items.append(editor_item)
                if has_pose:
                    pose_video_frame_paths.append(out_img)
                    batch_items.append(editor_item)
                elif tracking is None or tracked_box is not None:
                    errors.append(f"{caption}: {pose_error}")

            selected_count += 1
            next_sample_time += sample_interval

        frame_idx += 1

    capture.release()
    if tracking is not None:
        (run_dir / "tracking.json").write_text(json.dumps(dict(
            algorithm="LK optical flow + forward/backward check + RANSAC",
            initial_bbox=tracking["bbox"], records=tracking_records), ensure_ascii=False), encoding="utf-8")

    if not gallery_items:
        error_text = "\n".join(errors) if errors else "Videodan islenebilir frame uretilemedi."
        return (
            "",
            str(video_path),
            None,
            error_text,
            [],
            0,
            gr.update(value=1, minimum=1, maximum=2, visible=False),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            0,
            [],
            0,
            "",
        )

    elapsed = time.perf_counter() - t0
    progress(0.92, desc="Sonuç videosu hazırlanıyor…")
    pose_video_path, pose_video_error = _build_pose_video_from_frames(
        pose_video_frame_paths,
        run_dir / f"{_safe_folder_name(video_path.stem)}_pose_overlay",
        target_fps,
    )
    if pose_video_error:
        errors.append(f"Pose video: {pose_video_error}")
    progress(0.97, desc="Kareler editöre hazırlanıyor…")
    for editor_idx, editor_item in enumerate(video_editor_items):
        editor_payload, _, _, editor_status = _prepare_video_editor_from_item(
            editor_item,
            editor_idx,
            len(video_editor_items),
        )
        editor_item["editor_payload"] = editor_payload
        editor_item["editor_status"] = editor_status
    first_item = batch_items[0] if batch_items else None
    first_payload, _ = prepare_editor_from_path(first_item["original_image"], first_item["json_path"]) if first_item else ("", "")
    video_first_batch_idx = 0 if video_editor_items else None
    video_first_item = video_editor_items[0] if video_editor_items else None
    video_first_payload, video_first_original, video_first_json, video_first_status = (
        _prepare_video_editor_from_item(video_first_item, 0, len(video_editor_items))
        if video_first_item
        else ("", "", "", "Video frame sonucu yok.")
    )
    nav_status = _batch_nav_state_text(0, len(batch_items), first_item["source_name"]) if first_item else "Pose tespit edilen frame yok."
    video_result_nav_status = _batch_nav_state_text(0, len(gallery_items), gallery_items[0][1])
    status_lines = [
        f"Video işlendi: {video_path.name}",
        f"{len(gallery_items)}/{selected_count} kare işlendi · {len(batch_items)} karede poz bulundu · {elapsed:.1f} sn",
        f"Kayıt klasörü: {run_dir}",
    ]
    if not pose_video_path:
        status_lines.append("Çıktı videosu oluşturulamadı.")
    if tracking is not None:
        status_lines.append(f"Atlet takibi açık · {tracking_lost_count} kare elle kontrol bekliyor.")
    if errors:
        status_lines.append("\nHatalar:")
        status_lines.extend(errors)

    return (
        first_payload,
        pose_video_path or str(video_path),
        gallery_items[0][0] if gallery_items else None,
        "\n".join(status_lines),
        batch_items,
        0,
        gr.update(value=1, minimum=1, maximum=max(2, len(batch_items)), visible=len(batch_items) > 1),
        nav_status,
        first_item["original_image"] if first_item else "",
        first_item["json_path"] if first_item else "",
        video_first_payload,
        video_first_original,
        video_first_json,
        video_first_status,
        video_first_batch_idx or 0,
        video_editor_items,
        0,
        video_result_nav_status,
    )


def process_video_for_shared_editor(
    video_path_text: str,
    requested_fps: float,
    progress=gr.Progress(),
    tracking=None,
):
    """Map the video pipeline onto the single shared pose editor UI."""
    result = process_video_frames(video_path_text, requested_fps, progress, tracking=tracking)
    return (
        result[10],  # shared editor payload
        result[1],   # rendered pose video
        result[3],   # video processing status
        result[11],  # active original frame
        result[12],  # active frame JSON
        result[13],  # editor status
        result[14],  # active video frame index
        result[15],  # all video editor items
        result[16],  # legacy result index
        result[17],  # legacy navigation status
    )


def _status_box_updates(message: str, force_error: bool = False):
    """Split regular processing details and errors into separate UI boxes."""
    text = str(message or "").strip()
    info_text = ""
    error_text = ""

    if force_error:
        error_text = text
    else:
        lines = text.splitlines()
        error_index = next(
            (idx for idx, line in enumerate(lines) if line.strip().casefold() == "hatalar:"),
            None,
        )
        if error_index is None:
            info_text = text
        else:
            info_text = "\n".join(lines[:error_index]).strip()
            error_lines = [line for line in lines[error_index + 1:] if line.strip()]
            error_text = "\n".join(error_lines).strip()

    info_update = gr.update(value=info_text, visible=bool(info_text))
    error_update = gr.update(
        value=(f"**Hatalar**\n\n{error_text}" if error_text else ""),
        visible=bool(error_text),
    )
    return info_update, error_update


def prepare_editor_media(uploaded_file: Any):
    """Confirm that editor-dropped media is uploaded and ready for inference."""
    media_path = _uploaded_file_path(uploaded_file)
    if media_path is None:
        info_update, error_update = _status_box_updates(
            "Hazır — görsel veya videoyu editöre sürükleyebilirsiniz."
        )
        return (
            "",
            info_update,
            error_update,
        )
    if not media_path.exists() or not media_path.is_file():
        info_update, error_update = _status_box_updates(
            "Yüklenen medya dosyası bulunamadı.",
            force_error=True,
        )
        return (
            "",
            info_update,
            error_update,
        )

    extension = media_path.suffix.lower()
    supported = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS
    if extension not in supported:
        info_update, error_update = _status_box_updates(
            f"Desteklenmeyen dosya formatı: {extension or '(uzantı yok)' }.",
            force_error=True,
        )
        return (
            "",
            info_update,
            error_update,
        )

    media_kind = "video" if extension in SUPPORTED_VIDEO_EXTENSIONS else "image"
    signal = json.dumps(
        {"ready": True, "kind": media_kind, "name": media_path.name},
        ensure_ascii=False,
    )
    info_update, error_update = _status_box_updates(
        f"Medya hazır: {media_path.name}\nPoz tahminlemeyi editör içinden başlatabilirsiniz."
    )
    return (
        signal,
        info_update,
        error_update,
    )


def suggest_tracking_box(uploaded_file, request):
    """Suggest a person in the first frame; the user still confirms the target."""
    response = {"request": request, "bbox": None}
    capture = None
    try:
        target = json.loads(request)
        path = _uploaded_file_path(uploaded_file)
        if path is None or path.name != target.get("name") or path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError("Video hazır değil; yükleme tamamlandıktan sonra tekrar deneyin.")
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        if not ok:
            raise ValueError("İlk kare okunamadı; atleti elle seçin.")
        h, w = frame.shape[:2]
        with _MODEL_LOCK:
            device = _get_runtime_device()
            detector = _get_detector(device)
            detections = detector(frame, verbose=False, imgsz=320,
                                  device=device if device != 'cuda' else 0,
                                  classes=[0])[0].boxes.data.cpu().numpy()
        candidates = []
        for row in detections:
            if len(row) < 5 or not np.isfinite(row[:5]).all():
                continue
            x1, y1, x2, y2 = np.clip(row[:4], [0, 0, 0, 0], [w, h, w, h])
            if row[4] < 0.35 or x2-x1 < 12 or y2-y1 < 12:
                continue
            candidates.append(((x2-x1)*(y2-y1)*float(row[4]), [x1/w, y1/h, x2/w, y2/h]))
        if candidates:
            response["bbox"] = [float(v) for v in max(candidates, key=lambda item:item[0])[1]]
            response["message"] = "Otomatik öneri hazır. Doğru atlet değilse yeni bir kutu çizin."
        else:
            response["message"] = "Uygun kişi bulunamadı. Atleti kutuyla seçebilirsiniz."
    except Exception as exc:
        response["message"] = f"Otomatik öneri alınamadı; atleti elle seçebilirsiniz. ({exc})"
    finally:
        if capture is not None:
            capture.release()
    return json.dumps(response, ensure_ascii=False)


def process_uploaded_media_from_editor(uploaded_file, trigger_value):
    """Stream actual processing progress while keeping the video preview visible."""
    try:
        request_id = json.loads(trigger_value).get("at")
    except (TypeError, ValueError, AttributeError):
        request_id = None
    events = queue.Queue()
    def report(value, desc=""):
        events.put(("progress", (value, desc)))
    def work():
        try:
            events.put(("result", _process_uploaded_media_from_editor(uploaded_file, trigger_value, report)))
        except Exception as exc:
            events.put(("error", str(exc)))
    def signal(value, message, state="running"):
        return json.dumps(dict(request_id=request_id, value=value, message=message, state=state), ensure_ascii=False)
    yield (*[gr.skip() for _ in range(11)], signal(None, "İşlem hazırlanıyor…"))
    threading.Thread(target=work, daemon=True).start()
    while True:
        kind, data = events.get()
        if kind == "progress":
            value, message = data
            yield (*[gr.skip() for _ in range(11)], signal(value, message))
        elif kind == "result":
            success = json.loads(data[10]).get("success", False)
            yield (*data, signal(1 if success else None,
                                "İşlem tamamlandı." if success else "İşlem tamamlanamadı; hata açıklamasını kontrol edin.",
                                "complete" if success else "error"))
            break
        else:
            updates = [gr.skip() for _ in range(11)]
            updates[3] = gr.update(value=f"**Hata**\n\n{data}", visible=True)
            updates[10] = json.dumps({"complete":True,"success":False,"at":time.time_ns()})
            yield (*updates, signal(None, "İşlem tamamlanamadı. Hata açıklamasını kontrol edin.", "error"))
            break


def _process_uploaded_media_from_editor(
    uploaded_file: Any,
    trigger_value: str,
    progress=gr.Progress(),
):
    """Run shared media inference from the editor toolbar trigger."""
    requested_fps = 5.0
    tracking = None
    if isinstance(trigger_value, (int, float)):
        requested_fps = float(trigger_value)
    else:
        try:
            trigger_data = json.loads(trigger_value or "{}")
            requested_fps = float(trigger_data.get("fps", requested_fps))
            if trigger_data.get("tracking"):
                tracking = {"bbox": trigger_data.get("tracking_bbox"), "name": trigger_data.get("media_name")}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    requested_fps = max(1.0, min(15.0, requested_fps))
    progress(0, desc="Model ve medya hazırlanıyor…")
    result = process_uploaded_media(uploaded_file, requested_fps, progress, tracking=tracking)
    completion_signal = json.dumps({"complete": True, "success": bool(result[0]), "at": time.time_ns()})
    return (
        gr.update(value=result[0], visible=True) if result[0] else gr.update(),
        result[1],
        result[2],
        result[3],
        result[4],
        result[5],
        result[6],
        result[7],
        result[8],
        result[9],
        completion_signal,
    )


def process_uploaded_media(
    uploaded_file: Any,
    requested_fps: float,
    progress=gr.Progress(),
    tracking=None,
):
    """Run image or video inference from the shared upload component."""
    media_path = _uploaded_file_path(uploaded_file)

    def error_result(message: str):
        info_update, error_update = _status_box_updates(message, force_error=True)
        return (
            "",
            gr.update(value=None, visible=False),
            info_update,
            error_update,
            "",
            "",
            0,
            [],
            0,
            "",
        )

    if media_path is None:
        return error_result("Lutfen bir gorsel veya video secin.")
    if not media_path.exists() or not media_path.is_file():
        return error_result(f"Dosya bulunamadi: {media_path}")

    extension = media_path.suffix.lower()
    if extension in SUPPORTED_VIDEO_EXTENSIONS:
        if tracking is not None and tracking.get("name") != media_path.name:
            return error_result("Video değişti; takip edilecek atleti yeniden seçin.")
        try:
            result = process_video_for_shared_editor(str(media_path), requested_fps, progress, tracking=tracking)
        except ValueError as exc:
            return error_result(str(exc))
        info_update, error_update = _status_box_updates(result[2])
        return (
            result[0],
            gr.update(value=result[1], visible=True),
            info_update,
            error_update,
            result[3],
            result[4],
            result[6],
            result[7],
            result[8],
            result[9],
        )

    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        try:
            with Image.open(media_path) as source_image:
                image = source_image.convert("RGB")
        except Exception as exc:
            return error_result(f"Gorsel okunamadi: {exc}")

        result = run_vitpose(image, "")
        # The first line remains the JSON path in state for Apply & Save.
        # Present a readable summary separately from that internal value.
        message = result[2]
        if result[0]:
            json_path, _, summary = message.partition("\n")
            message = f"{summary}\nJSON kaydı: {json_path}"
        info_update, error_update = _status_box_updates(message, force_error=not bool(result[0]))
        return (
            result[0],
            gr.update(value=None, visible=False),
            info_update,
            error_update,
            result[9],
            result[2],
            0,
            [],
            0,
            "",
        )

    supported = ", ".join(
        sorted(SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS)
    )
    return error_result(
        f"Desteklenmeyen dosya formati: {extension or '(uzanti yok)'}. "
        f"Desteklenenler: {supported}"
    )


def reset_pose_workspace():
    """Return every pose-editor component and state to its initial value."""
    return (
        gr.update(value="", visible=True),
        gr.update(value=None, visible=False),
        gr.update(
            value="Hazır — görsel veya videoyu editöre sürükleyebilirsiniz.",
            visible=True,
        ),
        gr.update(value="", visible=False),
        "",
        "",
        0,
        [],
        0,
        "",
        "",
        "",
        None,
    )


def apply_and_save_for_ui(
    original_img_path: str,
    kps_json: str,
    json_path_str: str,
    current_payload: str = "",
):
    """Route Apply & Save feedback to the correct status box."""
    payload, status = apply_and_save_keypoints(
        original_img_path,
        kps_json,
        json_path_str,
        current_payload,
    )
    success = status.startswith(("Kaydedildi:", "Goruntu guncellendi"))
    info_update, error_update = _status_box_updates(status, force_error=not success)
    return payload, info_update, error_update


def _batch_nav_state_text(index: int, total: int, source_name: str) -> str:
    return f"{index + 1}/{total} - {source_name}"


def _prepare_video_editor_from_item(
    item: Dict[str, str],
    frame_index: int = 0,
    frame_count: int = 1,
) -> Tuple[str, str, str, str]:
    original_image = item.get("original_image", "")
    json_path = item.get("json_path", "")
    cached_payload = item.get("editor_payload", "")
    json_stamp = Path(json_path).stat().st_mtime_ns if json_path and Path(json_path).is_file() else None
    if cached_payload and item.get("editor_json_stamp") == json_stamp:
        return (
            cached_payload,
            original_image,
            json_path,
            item.get("editor_status", "Editor hazir."),
        )
    payload, status = prepare_editor_from_path(
        original_image,
        json_path,
        editor_role="video",
        output_id="kp_editor_output",
        prev_trigger_id="video_pe_prev_trigger",
        next_trigger_id="video_pe_next_trigger",
        frame_trigger_id="video_pe_frame_trigger",
        frame_index=frame_index,
        frame_count=frame_count,
    )
    if payload:
        item.update(editor_payload=payload, editor_status=status, editor_json_stamp=json_stamp)
    return payload, original_image, json_path, status


def _show_video_editor_item(batch_items: List[Dict[str, str]], index: int):
    if not batch_items:
        return "", "", "", 0

    idx = max(0, min(int(float(index or 0)), len(batch_items) - 1))
    payload, original_image, json_path, _ = _prepare_video_editor_from_item(
        batch_items[idx],
        idx,
        len(batch_items),
    )
    return payload, original_image, json_path, idx


def retry_video_frame(items, request, pending=None):
    """Preview a replacement pose; commit only the explicitly accepted preview."""
    response = {"request": request, "payload": ""}
    def finish(message, error=False):
        info_update, error_update = _status_box_updates(message, force_error=error)
        return json.dumps(response, ensure_ascii=False), info_update, error_update, pending

    try:
        target = json.loads(request)
        index = int(target["frame_index"])
        if not items or not 0 <= index < len(items):
            return finish("Tekrar tahminlenecek kare bulunamadı.", True)
        item = items[index]
        if item["original_image"] != target["original_image"]:
            return finish("Video değişti; açık kareden tekrar deneyin.", True)
        action = target.get("action", "preview")
        if action in ("accept", "discard"):
            if not pending or pending["token"] != target.get("token") or pending["original_image"] != item["original_image"]:
                return finish("Önizleme değişti; atleti yeniden seçin.", True)
            if action == "discard":
                pending = None
                response["stage"] = "discarded"
                return finish("Önizleme iptal edildi; önceki poz korundu.")
            output_json = Path(item["json_path"])
            if hashlib.sha256(output_json.read_bytes()).hexdigest() != pending["source_hash"]:
                return finish("Kare önizlemeden sonra değişti. Mevcut kaydı korumak için atleti yeniden seçin.", True)
            Path(item["result_image"]).write_bytes(pending["image"])
            output_json.write_bytes(pending["json"])
            response.update(payload=pending["payload"], stage="committed")
            pending = None
            return finish(f"Yeni poz kaydedildi: {item['source_name']}\nKayıt klasörü: {output_json.parent}")
        if action != "preview":
            return finish("Geçersiz önizleme işlemi.", True)
        bbox = target.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return finish("Önce atleti kapsayan bir kutu çizin.", True)

        source = Path(item["original_image"])
        output_json = Path(item["json_path"])
        source_hash = hashlib.sha256(output_json.read_bytes()).hexdigest()
        # Stage inference so a failed retry cannot overwrite previous files.
        with tempfile.TemporaryDirectory(prefix=".retry-", dir=output_json.parent) as staging:
            image_path, json_path, error = _run_inference_for_input(source, Path(staging), bbox=bbox)
            if error:
                return finish(f"Kare tekrar işlenemedi: {error}", True)
            has_pose, reason = _has_pose_keypoints(json_path)
            if not has_pose:
                return finish(f"Bu karede yine poz bulunamadı. Tekrar deneyebilirsiniz. ({reason})")
            refreshed = dict(item, json_path=json_path)
            refreshed.pop("editor_payload", None)
            payload, _, _, status = _prepare_video_editor_from_item(refreshed, index, len(items))
            if not payload:
                return finish(f"Kare editöre yüklenemedi: {status}", True)
            pending = dict(token=uuid.uuid4().hex, original_image=item["original_image"],
                           source_hash=source_hash, payload=payload,
                           image=Path(image_path).read_bytes(), json=Path(json_path).read_bytes())

        # Navigation invalidates cached payloads using the JSON file timestamp.
        # Do not return an old copy of session state if another video was loaded.
        response["payload"] = payload
        response.update(stage="preview", token=pending["token"])
        return finish("Yeni poz önizleniyor. Kaydetmek için Yeni Pozu Kullan, vazgeçmek için Önizlemeyi İptal Et seçin.")
    except Exception as exc:
        return finish(f"Kare tekrar tahminlenemedi: {exc}", True)


def _show_batch_item(batch_items: List[Dict[str, str]], index: int):
    if not batch_items:
        return "", "", 0, gr.update(value=1, minimum=1, maximum=2, visible=False), "Batch sonucu yok.", ""

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
        gr.update(value=idx + 1, minimum=1, maximum=max(2, len(batch_items)), visible=len(batch_items) > 1),
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


def _video_result_entry(video_results: List[Any], index: int) -> Tuple[Optional[str], str, int]:
    if not video_results:
        return None, "Video frame sonucu yok.", 0

    idx = max(0, min(int(index), len(video_results) - 1))
    entry = video_results[idx]
    if isinstance(entry, dict):
        result_image = entry.get("result_image") or entry.get("image")
        caption = entry.get("source_name") or entry.get("caption") or f"Frame {idx + 1}"
    else:
        result_image = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
        caption = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else f"Frame {idx + 1}"

    return str(result_image) if result_image else None, _batch_nav_state_text(idx, len(video_results), str(caption)), idx


def _show_video_result_item(
    video_results: List[Any],
    batch_items: List[Dict[str, str]],
    index: int,
):
    result_image, nav_text, result_idx = _video_result_entry(video_results, index)
    matched_batch_idx = None
    for idx, item in enumerate(batch_items or []):
        if int(item.get("gallery_index", idx)) == result_idx:
            matched_batch_idx = idx
            break

    if matched_batch_idx is None:
        return result_image, nav_text, result_idx, "", "", "", f"{nav_text} | Pose tespit edilemedi.", 0

    payload, original_image, json_path, status = _prepare_video_editor_from_item(batch_items[matched_batch_idx])
    return result_image, nav_text, result_idx, payload, original_image, json_path, status, matched_batch_idx


def _show_video_editor_and_result(
    video_results: List[Any],
    batch_items: List[Dict[str, str]],
    index: int,
):
    if not batch_items:
        return _show_video_result_item(video_results, batch_items, 0)

    batch_idx = max(0, min(int(index), len(batch_items) - 1))
    item = batch_items[batch_idx]
    result_idx = int(item.get("gallery_index", batch_idx))
    result_image, nav_text, result_idx = _video_result_entry(video_results, result_idx)
    if result_image is None:
        result_image = item.get("result_image")
        nav_text = _batch_nav_state_text(result_idx, len(video_results) or len(batch_items), item.get("source_name", ""))
    payload, original_image, json_path, status = _prepare_video_editor_from_item(item)
    return result_image, nav_text, result_idx, payload, original_image, json_path, status, batch_idx


def show_prev_video_result(video_results: List[Any], batch_items: List[Dict[str, str]], current_idx: int):
    return _show_video_result_item(video_results, batch_items, current_idx - 1)


def show_next_video_result(video_results: List[Any], batch_items: List[Dict[str, str]], current_idx: int):
    return _show_video_result_item(video_results, batch_items, current_idx + 1)


def show_prev_video_editor_item(video_results: List[Any], batch_items: List[Dict[str, str]], current_idx: int):
    return _show_video_editor_and_result(video_results, batch_items, current_idx - 1)


def show_next_video_editor_item(video_results: List[Any], batch_items: List[Dict[str, str]], current_idx: int):
    return _show_video_editor_and_result(video_results, batch_items, current_idx + 1)


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
            return "", [], err, [], 0, gr.update(value=1, minimum=1, maximum=2, visible=False), gr.update(), gr.update(), "", ""

        has_pose, pose_error = _has_pose_keypoints(json_path)
        if not has_pose:
            elapsed = time.perf_counter() - t0
            return (
                "",
                [(out_img, input_path.name)] if out_img else [],
                f"Poz tespit edilemedi: {pose_error}\nSüre: {elapsed:.1f} sn",
                [],
                0,
                gr.update(value=1, minimum=1, maximum=2, visible=False),
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
            f"{json_path}\nPoz tespiti tamamlandı · {elapsed:.1f} sn",
            [],
            0,
            gr.update(value=1, minimum=1, maximum=2, visible=False),
            gr.update(),
            gr.update(),
            "Tek görüntü modu.",
            str(input_path),
        )

    if not folder_path:
        return None, [], "Tek bir görüntü yükleyin veya görüntü klasörü yolu girin.", [], 0, gr.update(value=1, minimum=1, maximum=2, visible=False), gr.update(), gr.update(), "", ""

    dir_path = Path(folder_path)
    if not dir_path.exists() or not dir_path.is_dir():
        return None, [], f"Klasör bulunamadı veya geçersiz: {folder_path}", [], 0, gr.update(value=1, minimum=1, maximum=2, visible=False), gr.update(), gr.update(), "", ""

    image_files = _collect_images_from_directory(dir_path)
    if not image_files:
        return None, [], f"Klasörde desteklenen görüntü bulunamadı: {folder_path}", [], 0, gr.update(value=1, minimum=1, maximum=2, visible=False), gr.update(), gr.update(), "", ""

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
        return None, [], error_text, [], 0, gr.update(value=1, minimum=1, maximum=2, visible=False), gr.update(visible=False), gr.update(visible=False), "", ""

    status_lines = [
        f"{len(gallery_items)}/{len(image_files)} görüntü işlendi · {len(errors)} hata",
    ]
    elapsed = time.perf_counter() - t0
    status_lines.append(f"Toplam süre: {elapsed:.2f}s")
    if json_paths:
        status_lines.append(f"Kayıt klasörü: {Path(json_paths[0]).parent}")
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
        gr.update(value=1, minimum=1, maximum=max(2, len(batch_items)), visible=len(batch_items) > 1),
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
.pe-hidden-media-input {
  position: fixed !important;
  left: -10000px !important;
  top: 0 !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
}
.pe-internal-output { display: none !important; }
#media-status-messagebox,
#media-error-messagebox {
  border: 1px solid #60a5fa !important;
  border-left: 5px solid #3b82f6 !important;
  border-radius: 9px !important;
  background: rgba(59, 130, 246, 0.10) !important;
  padding: 12px 14px 12px 46px !important;
  position: relative;
  min-height: 48px;
  max-width: 100%;
  overflow: hidden !important;
  box-sizing: border-box;
}
#media-status-messagebox::before,
#media-error-messagebox::before {
  content: "i";
  position: absolute;
  left: 15px;
  top: 13px;
  width: 21px;
  height: 21px;
  border-radius: 50%;
  background: #3b82f6;
  color: white;
  font: 700 14px/21px sans-serif;
  text-align: center;
}
#media-error-messagebox {
  border-color: #f87171 !important;
  border-left-color: #ef4444 !important;
  background: rgba(239, 68, 68, 0.10) !important;
  margin-top: 10px;
}
#media-error-messagebox::before {
  content: "!";
  background: #ef4444;
}
#media-status-messagebox .prose,
#media-error-messagebox .prose {
  margin: 0 !important;
  white-space: pre-wrap;
  max-width: 100%;
  overflow: hidden !important;
  overflow-wrap: anywhere;
  word-break: break-word;
}
#media-status-messagebox .prose *,
#media-error-messagebox .prose * {
  max-width: 100%;
  overflow-wrap: anywhere;
  word-break: break-word;
}
#media-status-messagebox .prose > :first-child,
#media-error-messagebox .prose > :first-child { margin-top: 0 !important; }
#media-status-messagebox .prose > :last-child,
#media-error-messagebox .prose > :last-child { margin-bottom: 0 !important; }
"""

with gr.Blocks() as demo:
    gr.Markdown("## easy_ViTPose Studio")

    with gr.Tabs():
        with gr.Tab("Pose Editörü"):
            input_media = gr.File(
                label="Editör Medya Girişi",
                file_types=sorted(
                    SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS
                ),
                type="filepath",
                elem_id="pose_media_input",
                elem_classes=["pe-hidden-media-input"],
                container=False,
            )
            pose_editor_html = create_editor_component()

            with gr.Accordion("İşlem Durumu", open=True):
                media_status = gr.Markdown(
                    value="Hazır — görsel veya videoyu editöre sürükleyebilirsiniz.",
                    line_breaks=True,
                    elem_id="media-status-messagebox",
                )
                media_error = gr.Markdown(
                    value="",
                    line_breaks=True,
                    visible=False,
                    elem_id="media-error-messagebox",
                )

            processed_video = gr.Video(
                interactive=False,
                visible=False,
                elem_classes=["pe-internal-output"],
            )
            json_path_box = gr.Textbox(
                elem_classes="pe-hidden-trigger",
                elem_id="inference_json_path",
            )

            # Hidden textbox: JS canvas writes keypoints here; Python reads it.
            kp_editor_output = gr.Textbox(
                elem_id="kp_editor_output", elem_classes="pe-hidden-trigger", lines=1
            )
            video_frame_trigger = gr.Textbox(
                elem_classes="pe-hidden-trigger", elem_id="video_pe_frame_trigger"
            )
            editor_payload_update = gr.Textbox(
                elem_classes="pe-hidden-trigger", elem_id="pose_editor_payload_update"
            )
            pose_inference_trigger = gr.Textbox(
                elem_classes="pe-hidden-trigger", elem_id="pose_inference_trigger"
            )
            media_ready_signal = gr.Textbox(
                elem_classes="pe-hidden-trigger", elem_id="media_ready_signal"
            )
            inference_complete_signal = gr.Textbox(
                elem_classes="pe-hidden-trigger", elem_id="inference_complete_signal"
            )
            pose_clear_trigger = gr.Textbox(
                elem_classes="pe-hidden-trigger", elem_id="pose_clear_trigger"
            )
            retry_frame_trigger = gr.Textbox(
                elem_classes="pe-hidden-trigger", elem_id="pose_retry_frame_trigger"
            )
            retry_frame_result = gr.Textbox(elem_classes="pe-hidden-trigger")
            tracking_suggestion_trigger = gr.Textbox(elem_classes="pe-hidden-trigger", elem_id="tracking_suggestion_trigger")
            tracking_suggestion_result = gr.Textbox(elem_classes="pe-hidden-trigger")
            processing_progress_signal = gr.Textbox(elem_classes="pe-hidden-trigger")
            apply_save_btn = gr.Button("Değişiklikleri JSON'a Kaydet", variant="primary", elem_id="pose_apply_save")

        with gr.Tab("JSON İskelet Görüntüleyici"):
            gr.Markdown("### JSON dosyasındaki keypointleri ve iskeleti görsel üzerinde gösterin")
            with gr.Row():
                overlay_src_img = gr.Image(type="pil", label="Kaynak görsel")
                json_file = gr.File(label="Poz JSON dosyası (.json)", file_types=[".json"])
                overlay_img = gr.Image(type="pil", label="İskelet çizilmiş görsel")
            overlay_status = gr.Textbox(label="İşlem durumu", lines=2)
            draw_button = gr.Button("JSON'dan İskeleti Çiz", variant="primary")

    # State
    original_img_state = gr.State("")
    video_frame_nav_status = gr.State("")
    video_frame_index_state = gr.State(0)
    video_results_state = gr.State([])
    retry_preview_state = gr.State(None)
    video_result_index_state = gr.State(0)

    pose_inference_trigger.change(
        fn=process_uploaded_media_from_editor,
        inputs=[input_media, pose_inference_trigger],
        outputs=[
            pose_editor_html,
            processed_video,
            media_status,
            media_error,
            original_img_state,
            json_path_box,
            video_frame_index_state,
            video_results_state,
            video_result_index_state,
            video_frame_nav_status,
            inference_complete_signal,
            processing_progress_signal,
        ],
        show_progress="hidden",
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    tracking_suggestion_trigger.change(
        fn=suggest_tracking_box, inputs=[input_media, tracking_suggestion_trigger],
        outputs=[tracking_suggestion_result], show_progress="hidden",
    )
    tracking_suggestion_result.change(
        fn=None, inputs=[tracking_suggestion_result], outputs=[],
        js="""(signal) => { const w=document.querySelector('.pe-wrap');
          if(w && w._peTrackingSuggestion) w._peTrackingSuggestion(signal); return []; }""",
    )
    processing_progress_signal.change(
        fn=None, inputs=[processing_progress_signal], outputs=[],
        js="""(signal) => { const w=document.querySelector('.pe-wrap');
          if(w && w._peSetProcessingProgress) w._peSetProcessingProgress(signal); return []; }""",
    )
    _VIDEO_FRAME_OUTPUTS = [
        editor_payload_update,
        original_img_state,
        json_path_box,
        video_frame_index_state,
    ]

    input_media.change(
        fn=prepare_editor_media,
        inputs=[input_media],
        outputs=[
            media_ready_signal,
            media_status,
            media_error,
        ],
        show_progress="hidden",
        queue=False,
    )

    pose_clear_trigger.change(
        fn=reset_pose_workspace,
        inputs=[],
        outputs=[
            pose_editor_html,
            processed_video,
            media_status,
            media_error,
            original_img_state,
            json_path_box,
            video_frame_index_state,
            video_results_state,
            video_result_index_state,
            video_frame_nav_status,
            kp_editor_output,
            media_ready_signal,
            input_media,
        ],
        show_progress="hidden",
        queue=False,
    )

    video_frame_trigger.change(
        fn=_show_video_editor_item,
        inputs=[video_results_state, video_frame_trigger],
        outputs=_VIDEO_FRAME_OUTPUTS,
        show_progress="hidden",
        trigger_mode="always_last",
        queue=False,
    )

    retry_frame_trigger.change(
        fn=retry_video_frame,
        inputs=[video_results_state, retry_frame_trigger, retry_preview_state],
        outputs=[retry_frame_result, media_status, media_error, retry_preview_state],
        show_progress="hidden",
        trigger_mode="once",
    )
    retry_frame_result.change(
        fn=None,
        inputs=[retry_frame_result],
        outputs=[],
        js="""(result) => {
            var w = document.querySelector('.pe-wrap');
            if (w && w._peRetryComplete) w._peRetryComplete(result);
            return [];
        }""",
    )

    _LOAD_EDITOR_JS = """(payload) => {
        var w = document.querySelector('.pe-wrap');
        if (w && payload && typeof w._peLoadPayload === 'function') {
            w._peLoadPayload(payload);
        }
        return [];
    }"""

    editor_payload_update.change(
        fn=None,
        inputs=[editor_payload_update],
        outputs=[],
        js=_LOAD_EDITOR_JS,
        show_progress="hidden",
    )

    _MEDIA_READY_JS = """(signal) => {
        var w = document.querySelector('.pe-wrap');
        if (w && typeof w._peSetMediaReady === 'function') {
            w._peSetMediaReady(signal || '');
        }
        return [];
    }"""

    media_ready_signal.change(
        fn=None,
        inputs=[media_ready_signal],
        outputs=[],
        js=_MEDIA_READY_JS,
        show_progress="hidden",
    )

    _INFERENCE_COMPLETE_JS = """(signal) => {
        var w = document.querySelector('.pe-wrap');
        if (w && signal && typeof w._peInferenceComplete === 'function') {
            w._peInferenceComplete();
        }
        return [];
    }"""

    inference_complete_signal.change(
        fn=None,
        inputs=[inference_complete_signal],
        outputs=[],
        js=_INFERENCE_COMPLETE_JS,
        show_progress="hidden",
    )

    # Read current keypoints directly from canvas on button click.
    _APPLY_JS = """(orig, kps, jpath, payload) => {
        var w = document.querySelector('.pe-wrap');
        if (w && typeof w._peGetKps === 'function') {
            kps = w._peGetKps();
        }
        if (w && typeof w._peGetPayload === 'function') {
            payload = w._peGetPayload();
        }
        return [orig, kps, jpath, payload];
    }"""

    apply_save_btn.click(
        fn=apply_and_save_for_ui,
        inputs=[original_img_state, kp_editor_output, json_path_box, pose_editor_html],
        outputs=[editor_payload_update, media_status, media_error],
        js=_APPLY_JS,
        show_progress="hidden",
    )

    draw_button.click(
        fn=draw_from_json,
        inputs=[overlay_src_img, json_file],
        outputs=[overlay_img, overlay_status],
    )

if __name__ == "__main__":
     demo.launch(css=_CSS, share=True)
