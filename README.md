# Vietnamese News Summarization with ViT5-base

Dự án fine-tune mô hình `VietAI/vit5-base` cho bài toán tóm tắt văn bản tin tức tiếng Việt theo hướng abstractive summarization. Pipeline hiện tại sử dụng dataset `ithieund/VietNews-Abs-Sum`, trong đó đầu vào là cột `article` và nhãn tóm tắt là cột `abstract`.

## Mục tiêu

- Fine-tune ViT5-base cho task tóm tắt tin tức tiếng Việt.
- Hỗ trợ train thử trên một phần nhỏ dữ liệu trước khi train toàn bộ.
- Đánh giá mô hình bằng ROUGE.
- Chạy suy luận trực tiếp từ command line với văn bản tiếng Việt bất kỳ.
- Tối ưu sẵn cho máy có GPU NVIDIA thông qua PyTorch CUDA.

## Cấu trúc dự án

```text
TextSummarization/
  data/
  models/
  outputs/
  src/
    __init__.py
    config.py
    data.py
    evaluate.py
    inference.py
    train.py
  requirements.txt
  README.md
```

Trong đó:

- `src/config.py`: cấu hình mặc định cho model, dataset, độ dài input/output và thư mục lưu model.
- `src/data.py`: tải dataset và tokenize dữ liệu.
- `src/train.py`: fine-tune ViT5-base bằng `Seq2SeqTrainer`.
- `src/evaluate.py`: đánh giá model đã train trên tập test.
- `src/inference.py`: sinh bản tóm tắt cho một đoạn văn bản đầu vào.

## Yêu cầu môi trường

- Python 3.10 hoặc mới hơn.
- GPU NVIDIA được khuyến nghị nếu muốn train nhanh.
- CUDA tương thích với bản PyTorch trong `requirements.txt`.

File `requirements.txt` hiện đang dùng PyTorch CUDA 11.8:

```text
torch==2.7.1+cu118
torchvision==0.22.1+cu118
torchaudio==2.7.1+cu118
```

Nếu máy không có GPU hoặc CUDA không tương thích, bạn cần đổi bản PyTorch phù hợp theo hướng dẫn cài đặt chính thức của PyTorch.

## Cài đặt

Tạo virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài thư viện:

```powershell
pip install -r requirements.txt
```

Kiểm tra PyTorch có nhận GPU không:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Cấu hình mặc định

Cấu hình chính nằm trong `src/config.py`:

```python
model_name = "VietAI/vit5-base"
dataset_name = "ithieund/VietNews-Abs-Sum"
source_column = "article"
target_column = "abstract"
max_source_length = 512
max_target_length = 128
output_dir = "models/vit5-vietnews-summarization"
generation_num_beams = 4
```

## Train thử

Nên chạy một lần nhỏ để kiểm tra toàn bộ pipeline tải dữ liệu, tokenize, train, evaluate và lưu model:

```powershell
python -m src.train --max-train-samples 1000 --max-eval-samples 200 --epochs 1 --batch-size 2
```

Mặc định script train đã bật `fp16` nếu máy có CUDA. Model sau khi train sẽ được lưu tại:

```text
models/vit5-vietnews-summarization
```

## Train nhiều dữ liệu hơn

Sau khi train thử thành công, có thể tăng số mẫu:

```powershell
python -m src.train --max-train-samples 20000 --max-eval-samples 2000 --epochs 3 --batch-size 2 --gradient-accumulation-steps 8
```

Nếu muốn train toàn bộ tập train:

```powershell
python -m src.train --full-dataset --max-eval-samples 2000 --epochs 3 --batch-size 2 --gradient-accumulation-steps 8
```

Với GPU ít VRAM, hãy giảm `--batch-size` hoặc bật gradient checkpointing:

```powershell
python -m src.train --max-train-samples 20000 --max-eval-samples 2000 --epochs 3 --batch-size 1 --gradient-accumulation-steps 16 --gradient-checkpointing
```

## Đánh giá

Đánh giá model đã fine-tune trên tập test:

```powershell
python -m src.evaluate --model-path models/vit5-vietnews-summarization --max-test-samples 2000 --batch-size 2
```

Kết quả sẽ in ra các chỉ số như:

- `eval_rouge1`
- `eval_rouge2`
- `eval_rougeL`
- `eval_rougeLsum`
- `eval_loss`

## Chạy inference

Sinh tóm tắt cho một đoạn tin tức tiếng Việt:

```powershell
python -m src.inference --model-path models/vit5-vietnews-summarization --text "Nội dung bài báo tiếng Việt cần tóm tắt đặt ở đây..."
```

Có thể điều chỉnh độ dài và beam search:

```powershell
python -m src.inference --model-path models/vit5-vietnews-summarization --text "Nội dung bài báo..." --min-length 30 --max-length 128 --num-beams 4
```

## Gợi ý workflow

1. Cài dependencies.
2. Chạy train thử với 1.000 mẫu.
3. Nếu không lỗi, tăng lên 20.000 mẫu.
4. Đánh giá bằng `src.evaluate`.
5. Dùng `src.inference` để kiểm tra chất lượng tóm tắt trên tin tức thực tế.
6. Khi cấu hình ổn, train với `--full-dataset`.

## Lưu ý

- Dataset được tải tự động từ Hugging Face bằng thư viện `datasets`.
- Lần chạy đầu tiên sẽ mất thời gian để tải model và dataset.
- ViT5-base tương đối nặng, nên train CPU sẽ rất chậm.
- `max_source_length=512` nghĩa là bài báo dài hơn 512 token sẽ bị cắt bớt.
- `max_target_length=128` giới hạn độ dài bản tóm tắt sinh ra.
- Thư mục `models/` dùng để lưu checkpoint và model cuối cùng.
- Thư mục `outputs/` dùng cho kết quả đánh giá và output phụ.
